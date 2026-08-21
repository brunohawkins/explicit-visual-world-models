"""Stable-WorldModel TwoRoom session for LeWM-protocol VDA evaluation.

This mirrors the public ``swm/TwoRoom-v1`` mechanics used by LeWM closely
enough to run VDA's MPC loop without importing the LeWM virtualenv. The direct
comparison path should feed it manifest cases sampled from LeWM's ``tworoom.h5``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image


SWM_TWO_ROOM_FRAME_SIZE = (224, 224)
SWM_TWO_ROOM_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float64)
SWM_TWO_ROOM_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float64)
SWM_TWO_ROOM_SUCCESS_RADIUS = 16.0


def _resize(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape[:2] == (frame_size[1], frame_size[0]):
        return arr
    return np.asarray(Image.fromarray(arr).resize(frame_size, Image.NEAREST), dtype=np.uint8)


class SWMTwoRoomSession:
    """Session implementing the minimal P2 ``start/step/close`` interface."""

    img_size = 224
    border_size = 14
    wall_center = 112.0
    wall_thickness = 10
    agent_radius = 7.0
    speed = 5.0
    default_door_center = 49.0
    default_door_half_extent = 14.0

    def __init__(
        self,
        *,
        sim_frame_size: tuple[int, int] = SWM_TWO_ROOM_FRAME_SIZE,
        max_episode_steps: int = 50,
        manifest_case: dict[str, Any] | None = None,
    ) -> None:
        self.sim_frame_size = sim_frame_size
        self.max_episode_steps = int(max_episode_steps)
        self.manifest_case = manifest_case
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        self.door_center = float(self.default_door_center)
        self.steps = 0
        self._goal_raw: np.ndarray | None = None

    def _case_value(self, section: str, *keys: str):
        payload = (self.manifest_case or {}).get(section) or {}
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _reset_from_case(self, seed: Optional[int]) -> None:
        if self.manifest_case is not None:
            start = self._case_value("start", "state", "proprio", "pos_agent")
            goal = self._case_value("goal", "goal_state", "state", "proprio", "pos_target")
            door = self._case_value("metadata", "door_center")
            if start is None or goal is None:
                raise ValueError(f"SWM TwoRoom manifest case missing start/goal: {self.manifest_case}")
            self.agent_position = np.asarray(start, dtype=np.float32).reshape(2)
            self.target_position = np.asarray(goal, dtype=np.float32).reshape(2)
            if door is not None:
                self.door_center = float(door)
            return

        rng = np.random.default_rng(seed)
        self.agent_position = np.array([60.0, rng.uniform(35.0, 185.0)], dtype=np.float32)
        self.target_position = np.array([164.0, rng.uniform(35.0, 185.0)], dtype=np.float32)
        self.door_center = float(self.default_door_center)

    def _gaussian_dot(self, pos_xy: np.ndarray, radius: float) -> np.ndarray:
        y = np.arange(self.img_size, dtype=np.float32)[:, None]
        x = np.arange(self.img_size, dtype=np.float32)[None, :]
        dx = x - float(pos_xy[0])
        dy = y - float(pos_xy[1])
        dot = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        m = float(dot.max())
        return dot / m if m > 0 else dot

    @staticmethod
    def _alpha_blend(img: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        out = img.astype(np.float32)
        a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        for c, value in enumerate(color):
            out[..., c] = out[..., c] * (1.0 - a) + float(value) * a
        return out.astype(np.uint8)

    def _render(self, agent_pos: np.ndarray) -> np.ndarray:
        img = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        half = self.wall_thickness // 2
        c = int(self.wall_center)
        wall_mask = np.zeros((self.img_size, self.img_size), dtype=bool)
        wall_mask[:, c - half : c + half + 1] = True

        yy = np.arange(self.img_size, dtype=np.float32)[:, None]
        door_span = (yy >= self.door_center - self.default_door_half_extent) & (
            yy <= self.door_center + self.default_door_half_extent
        )
        wall_mask &= ~np.broadcast_to(door_span, wall_mask.shape)

        bs = self.border_size
        t = 4
        wall_mask[:, bs - t : bs] = True
        wall_mask[:, self.img_size - bs : self.img_size - bs + t] = True
        wall_mask[bs - t : bs, :] = True
        wall_mask[self.img_size - bs : self.img_size - bs + t, :] = True
        img[wall_mask] = (0, 0, 0)

        return self._alpha_blend(img, self._gaussian_dot(agent_pos, self.agent_radius), (255, 0, 0))

    def _in_door(self, y_value: float) -> bool:
        margin = 1.75
        return (self.door_center - self.default_door_half_extent - margin) <= y_value <= (
            self.door_center + self.default_door_half_extent + margin
        )

    def _apply_collisions(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        bs = float(self.border_size)
        x2, y2 = float(pos2[0]), float(pos2[1])
        x2 = min(max(x2, bs + self.agent_radius), self.img_size - bs - self.agent_radius)
        y2 = min(max(y2, bs + self.agent_radius), self.img_size - bs - self.agent_radius)
        out = np.array([x2, y2], dtype=np.float32)

        half = self.wall_thickness // 2
        wall_left = self.wall_center - half
        wall_right = self.wall_center + half
        effective_left = wall_left - self.agent_radius
        effective_right = wall_right + self.agent_radius
        started_left = float(pos1[0]) < self.wall_center

        if started_left:
            if float(out[0]) > effective_left and not self._in_door(float(out[1])):
                out[0] = effective_left - 0.5
        else:
            if float(out[0]) < effective_right and not self._in_door(float(out[1])):
                out[0] = effective_right + 0.5
        return out

    def _info(self) -> dict[str, Any]:
        distance = float(np.linalg.norm(self.agent_position - self.target_position))
        return {
            "env_name": "SWMTwoRoom",
            "state": self.agent_position.astype(float).tolist(),
            "proprio": self.agent_position.astype(float).tolist(),
            "goal_state": self.target_position.astype(float).tolist(),
            "distance_to_target": distance,
            "success_radius": SWM_TWO_ROOM_SUCCESS_RADIUS,
            "door_center": float(self.door_center),
            "manifest_case_id": (self.manifest_case or {}).get("case_id"),
        }

    def _pack(self, *, done: bool, truncated: bool, reward: float = 0.0) -> dict[str, Any]:
        image_raw = self._render(self.agent_position)
        goal_raw = self._goal_raw if self._goal_raw is not None else self._render(self.target_position)
        return {
            "image": _resize(image_raw, self.sim_frame_size),
            "image_raw": image_raw,
            "goal_image": _resize(goal_raw, self.sim_frame_size),
            "goal_image_raw": goal_raw,
            "info": self._info(),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
        }

    def start(self, seed: Optional[int] = None) -> dict[str, Any]:
        self.steps = 0
        self._reset_from_case(seed)
        self._goal_raw = self._render(self.target_position)
        done = float(np.linalg.norm(self.agent_position - self.target_position)) < SWM_TWO_ROOM_SUCCESS_RADIUS
        return self._pack(done=done, truncated=False, reward=0.0)

    def step(self, action_sim) -> dict[str, Any]:
        action = np.asarray(action_sim, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        next_pos = self.agent_position + action * self.speed
        self.agent_position = self._apply_collisions(self.agent_position, next_pos)
        self.steps += 1
        done = float(np.linalg.norm(self.agent_position - self.target_position)) < SWM_TWO_ROOM_SUCCESS_RADIUS
        truncated = self.steps >= self.max_episode_steps and not done
        return self._pack(done=done, truncated=truncated, reward=0.0)

    def close(self) -> None:
        return None
