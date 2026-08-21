"""In-process gym-pusht session for the P2 MPC test-time loop.

Unlike PLDM (heavy, needs an out-of-process FastAPI server), gym-pusht is a
lightweight gymnasium env that runs in-process — so this is a thin wrapper, no
HTTP. It exposes the minimal session interface the generalised ``eval_mpc``
uses: ``start(seed) / step(action_sim) / close()``. All coordinate conversion
lives in ``pusht_adapter`` so the MPC loop stays coordinate-agnostic.

Zero-shot protocol: each ``seed`` randomises the initial block + agent
configuration and this wrapper assigns a deterministic seed-specific target
pose, giving a fresh test config the P1 simulator never saw.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import numpy as np
from PIL import Image, ImageDraw

from vdaworld.eval.pusht_adapter import (
    env_obs_to_sim_frame,
    non_occluding_pusht_agent,
    sim_action_to_native,
    strip_green_target,
)


def _transform_t_vertices(
    vertices: list[tuple[float, float]],
    *,
    pose: np.ndarray,
    frame_size: tuple[int, int],
    scale: float = 1.0,
) -> list[tuple[float, float]]:
    """Map Push-T native local T vertices into simulator image pixels."""
    x0, y0, theta = [float(v) for v in pose]
    c, s = np.cos(theta), np.sin(theta)
    sx = float(frame_size[0]) / 512.0
    sy = float(frame_size[1]) / 512.0
    out = []
    for x, y in vertices:
        x *= scale
        y *= scale
        wx = x0 + c * x - s * y
        wy = y0 + s * x + c * y
        out.append((wx * sx, (512.0 - wy) * sy))
    return out


def _draw_t_at_pose(
    draw: ImageDraw.ImageDraw,
    *,
    pose: np.ndarray,
    frame_size: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    """Draw the same two-rectangle T geometry used by gym-pusht."""
    base = 30.0
    length = 4.0
    rects = [
        [
            (-length * base / 2, base),
            (length * base / 2, base),
            (length * base / 2, 0.0),
            (-length * base / 2, 0.0),
        ],
        [
            (-base / 2, base),
            (-base / 2, length * base),
            (base / 2, length * base),
            (base / 2, base),
        ],
    ]
    for vertices in rects:
        draw.polygon(
            _transform_t_vertices(vertices, pose=pose, frame_size=frame_size, scale=scale),
            fill=color,
        )


def _expert_style_goal_image(
    start_frame: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Return a per-instance Push-T target frame from the real start image.

    ``image_A`` already contains the correct seed-specific green target. Build
    ``image_B`` by removing the current grey block and overlaying grey on that
    exact green target mask, so the target cannot be shifted/flipped by a
    coordinate conversion mistake.
    """
    arr = np.asarray(start_frame, dtype=np.uint8).copy()
    rgb = arr.astype(np.int16)
    green = (
        (rgb[..., 1] > rgb[..., 0] + 20)
        & (rgb[..., 1] > rgb[..., 2] + 20)
        & (rgb[..., 1] > 100)
    )
    blue = (
        (rgb[..., 2] > rgb[..., 0] + 40)
        & (rgb[..., 2] > rgb[..., 1] + 20)
        & (rgb[..., 2] > 100)
    )
    grey = (
        (np.abs(rgb[..., 0] - rgb[..., 1]) < 35)
        & (np.abs(rgb[..., 1] - rgb[..., 2]) < 35)
        & (rgb[..., 0] > 70)
        & (rgb[..., 0] < 190)
        & (~green)
        & (~blue)
    )

    # Remove the start-state block while keeping pusher and target.
    arr[grey] = np.array([255, 255, 255], dtype=np.uint8)

    # Grey block covers the target, leaving a thin green border when possible.
    padded = np.pad(green, 1, constant_values=False)
    eroded = green.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        eroded &= padded[1 + dy : 1 + dy + green.shape[0], 1 + dx : 1 + dx + green.shape[1]]
    cover = eroded if int(eroded.sum()) >= 0.35 * int(green.sum()) else green
    arr[cover] = np.array([128, 128, 128], dtype=np.uint8)

    if arr.shape[:2] != (int(frame_size[1]), int(frame_size[0])):
        arr = np.asarray(Image.fromarray(arr).resize(tuple(frame_size), Image.Resampling.NEAREST))
    return arr


class GymPushTSession:
    """Thin in-process wrapper over ``gym_pusht/PushT-v0`` pixel observations."""

    ENV_ID = "gym_pusht/PushT-v0"

    def __init__(
        self,
        *,
        sim_frame_size: tuple[int, int] = (96, 96),
        render_px: int | None = None,
        max_episode_steps: int = 300,
        randomize_goal_pose: bool = True,
        goal_image_mode: str = "covered_goal",
        manifest_case: dict[str, Any] | None = None,
        strip_green_target_from_sim: bool = False,
    ) -> None:
        if goal_image_mode not in {"covered_goal", "start"}:
            raise ValueError(f"unsupported Push-T goal_image_mode={goal_image_mode!r}")
        self.sim_frame_size = (int(sim_frame_size[0]), int(sim_frame_size[1]))
        self.render_size = (
            (int(render_px), int(render_px))
            if render_px is not None
            else self.sim_frame_size
        )
        self.max_episode_steps = max_episode_steps
        self.randomize_goal_pose = randomize_goal_pose
        self.goal_image_mode = goal_image_mode
        self.manifest_case = manifest_case
        self.strip_green_target_from_sim = bool(strip_green_target_from_sim)
        self._env = None
        self._goal_image_sim = None
        self._goal_render_info = {}

    def _make_env(self):
        return gym.make(
            self.ENV_ID,
            obs_type="pixels",
            render_mode="rgb_array",
            observation_width=self.render_size[0],
            observation_height=self.render_size[1],
            max_episode_steps=self.max_episode_steps,
        )

    def _goal_pose_for_seed(self, seed: Optional[int]) -> np.ndarray:
        if not self.randomize_goal_pose:
            return np.array([256.0, 256.0, np.pi / 4.0], dtype=np.float64)
        rng = np.random.default_rng(0 if seed is None else int(seed) + 104729)
        x = rng.uniform(150.0, 362.0)
        y = rng.uniform(150.0, 362.0)
        theta = rng.uniform(-np.pi, np.pi)
        return np.array([x, y, theta], dtype=np.float64)

    def _render_goal_state(
        self,
        seed: Optional[int],
        info: dict[str, Any],
        start_frame: np.ndarray,
    ) -> np.ndarray:
        """Render image_B with the grey T at the green target pose.

        Use a separate temporary env so constructing image_B cannot perturb the
        live P2 environment state that will be stepped by MPC.
        """
        goal_pose = np.asarray(info.get("goal_pose"), dtype=np.float64).reshape(3)
        requested_goal_agent_xy = np.asarray(
            info.get("goal_pos_agent", info.get("pos_agent")), dtype=np.float64
        ).reshape(2)
        goal_agent_xy, pusher_relocated = non_occluding_pusht_agent(
            goal_pose, requested_goal_agent_xy
        )
        goal_env = self._make_env()
        try:
            _, goal_info = goal_env.reset(seed=seed)
            unwrapped = goal_env.unwrapped
            unwrapped.goal_pose = goal_pose.copy()
            unwrapped.agent.position = goal_agent_xy.tolist()
            unwrapped.agent.velocity = (0.0, 0.0)
            unwrapped.block.angle = float(goal_pose[2])
            unwrapped.block.position = goal_pose[:2].tolist()
            unwrapped.block.velocity = (0.0, 0.0)
            unwrapped.block.angular_velocity = 0.0
            # Direct Pymunk pose writes update coverage immediately, but the
            # debug renderer can still use the previous shape transform until
            # the space is stepped once.
            unwrapped.space.step(unwrapped.dt)
            # Option 2: construct image_B as a real Push-T env render of the
            # same task instance with the grey block placed at the target pose.
            # Important: gym-pusht's body convention requires setting angle
            # before position; the reverse visibly shifts the rendered T.
            obs = unwrapped._render()
            self._goal_render_info = {
                **dict(goal_info),
                "pos_agent": np.array(unwrapped.agent.position),
                "block_pose": np.array(list(unwrapped.block.position) + [unwrapped.block.angle]),
                "goal_pose": goal_pose.copy(),
                "coverage": float(unwrapped._get_coverage()),
                "is_success": bool(unwrapped._get_coverage() > unwrapped.success_threshold),
                "visual_target_source": "real_env_angle_first_goal_state_after_space_step",
                "goal_pusher_requested_native": requested_goal_agent_xy.copy(),
                "goal_pusher_rendered_native": goal_agent_xy.copy(),
                "goal_pusher_relocated_for_visibility": bool(pusher_relocated),
            }
            return env_obs_to_sim_frame(obs, self.sim_frame_size)
        finally:
            goal_env.close()

    def _apply_manifest_start_goal(self, info: dict[str, Any]) -> dict[str, Any]:
        if self.manifest_case is None:
            return info
        start = dict(self.manifest_case.get("start") or {})
        goal = dict(self.manifest_case.get("goal") or {})
        unwrapped = self._env.unwrapped

        if start.get("pos_agent") is not None:
            unwrapped.agent.position = np.asarray(start["pos_agent"], dtype=np.float64).reshape(2).tolist()
        if start.get("vel_agent") is not None:
            unwrapped.agent.velocity = np.asarray(start["vel_agent"], dtype=np.float64).reshape(2).tolist()
        if start.get("block_pose") is not None:
            block_pose = np.asarray(start["block_pose"], dtype=np.float64).reshape(3)
            unwrapped.block.angle = float(block_pose[2])
            unwrapped.block.position = block_pose[:2].tolist()
        if start.get("block_velocity") is not None:
            unwrapped.block.velocity = np.asarray(start["block_velocity"], dtype=np.float64).reshape(2).tolist()
        else:
            unwrapped.block.velocity = (0.0, 0.0)
        unwrapped.block.angular_velocity = float(start.get("block_angular_velocity", 0.0))

        if goal.get("goal_pose") is not None:
            unwrapped.goal_pose = np.asarray(goal["goal_pose"], dtype=np.float64).reshape(3)
        elif goal.get("block_pose") is not None:
            unwrapped.goal_pose = np.asarray(goal["block_pose"], dtype=np.float64).reshape(3)

        unwrapped.space.step(unwrapped.dt)
        updated = dict(unwrapped._get_info())
        if goal.get("pos_agent") is not None:
            updated["goal_pos_agent"] = np.asarray(goal["pos_agent"], dtype=np.float64).reshape(2)
        updated["manifest_case_id"] = self.manifest_case.get("case_id")
        updated["manifest_goal_state"] = goal
        updated["manifest_start_state"] = start
        return updated

    def start(self, seed: Optional[int] = None) -> dict[str, Any]:
        self._env = self._make_env()
        obs, info = self._env.reset(seed=seed)
        if self.manifest_case is None:
            goal_pose = self._goal_pose_for_seed(seed)
            self._env.unwrapped.goal_pose = goal_pose.copy()
        info = self._apply_manifest_start_goal(dict(info))
        # Re-render image_A after assigning the per-seed goal, so the green T
        # shown to the VLM matches the live environment success target.
        frame = env_obs_to_sim_frame(self._env.unwrapped._render(), self.sim_frame_size)
        info = {**dict(info), **dict(self._env.unwrapped._get_info())}
        if self.manifest_case is not None:
            goal = dict(self.manifest_case.get("goal") or {})
            if goal.get("pos_agent") is not None:
                info["goal_pos_agent"] = np.asarray(goal["pos_agent"], dtype=np.float64).reshape(2)
            info["manifest_case_id"] = self.manifest_case.get("case_id")
        if self.goal_image_mode == "start":
            # Diagnostic mode: give the simulator the reset frame as image_B,
            # while the live environment still evaluates against goal_pose.
            self._goal_image_sim = frame.copy()
            self._goal_render_info = {
                **dict(info),
                "coverage": float(self._env.unwrapped._get_coverage()),
                "is_success": bool(
                    self._env.unwrapped._get_coverage() > self._env.unwrapped.success_threshold
                ),
                "visual_target_source": "start_frame_with_visible_green_target",
            }
        else:
            # image_B is the intended final state: render a separate target
            # frame where the grey T covers the green target pose.
            self._goal_image_sim = self._render_goal_state(seed, dict(info), frame)
        green_target_mask = (
            (self._goal_image_sim[..., 1] > self._goal_image_sim[..., 0])
            & (self._goal_image_sim[..., 1] > self._goal_image_sim[..., 2])
        )
        frame_sim = (
            strip_green_target(frame)
            if self.strip_green_target_from_sim
            else frame
        )
        goal_sim = (
            strip_green_target(self._goal_image_sim)
            if self.strip_green_target_from_sim
            else self._goal_image_sim
        )
        return {
            "image": frame_sim,
            "image_raw": frame,
            "goal_image": goal_sim,
            "goal_image_raw": self._goal_image_sim,
            "info": {
                **dict(info),
                "goal_render_info": self._goal_render_info,
                "goal_target_visible_pixels": int(green_target_mask.sum()),
            },
            "reward": 0.0,
            "done": False,
            "truncated": False,
        }

    def step(self, action_sim) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("call start() before step()")
        native = sim_action_to_native(action_sim, self.sim_frame_size)
        obs, reward, terminated, truncated, info = self._env.step(native)
        frame = env_obs_to_sim_frame(obs, self.sim_frame_size)
        frame_sim = (
            strip_green_target(frame)
            if self.strip_green_target_from_sim
            else frame
        )
        goal_sim = (
            strip_green_target(self._goal_image_sim)
            if self.strip_green_target_from_sim
            else self._goal_image_sim
        )
        return {
            "image": frame_sim,
            "image_raw": frame,
            "goal_image": goal_sim,
            "goal_image_raw": self._goal_image_sim,
            "info": dict(info),
            "reward": float(reward),
            # gym-pusht terminates on success (coverage ≥ 0.95).
            "done": bool(terminated or info.get("is_success", False)),
            "truncated": bool(truncated),
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
