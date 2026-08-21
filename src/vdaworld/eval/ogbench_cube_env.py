"""OGBench cube session for the shared P2 MPC harness."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


CUBE_ACTION_LOW = -np.ones(5, dtype=np.float64)
CUBE_ACTION_HIGH = np.ones(5, dtype=np.float64)


class OGBenchCubeSession:
    """In-process OGBench cube wrapper exposing start/step/close."""

    def __init__(
        self,
        *,
        env_type: str = "single",
        task_id: int = 1,
        sim_frame_size: tuple[int, int] = (200, 200),
        manifest_case: dict[str, Any] | None = None,
    ) -> None:
        self.env_type = env_type
        self.task_id = task_id
        self.sim_frame_size = sim_frame_size
        self.manifest_case = dict(manifest_case) if manifest_case is not None else None
        self._env = None
        self._goal_image = None

    def _make_env(self):
        from ogbench.manipspace.envs.cube_env import CubeEnv

        width, height = self.sim_frame_size
        return CubeEnv(
            self.env_type,
            ob_type="pixels",
            height=int(height),
            width=int(width),
            visualize_info=False,
            terminate_at_goal=True,
        )

    @staticmethod
    def _frame(obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.uint8)

    def _reset_options(self) -> dict[str, Any]:
        return {"render_goal": True, "task_id": int(self.task_id)}

    @staticmethod
    def _array(case: dict[str, Any], key: str) -> np.ndarray:
        if key not in case:
            raise KeyError(f"manifest cube case missing required key {key!r}")
        return np.asarray(case[key], dtype=np.float64)

    def _set_target_from_case(self, case: dict[str, Any]) -> None:
        if self._env is None:
            raise RuntimeError("cube environment is not initialized")
        cube_id = int(case.get("cube_id", 0))
        target_pos = self._array(case, "target_pos")
        target_quat = np.asarray(
            case.get("target_quat", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        self._env._data.mocap_pos[self._env._cube_target_mocap_ids[cube_id]] = target_pos
        self._env._data.mocap_quat[self._env._cube_target_mocap_ids[cube_id]] = target_quat

    def _render_goal_from_case(self, case: dict[str, Any]) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("cube environment is not initialized")

        saved_qpos = self._env._data.qpos.copy()
        saved_qvel = self._env._data.qvel.copy()
        saved_target_pos = self._env._data.mocap_pos.copy()
        saved_target_quat = self._env._data.mocap_quat.copy()

        try:
            self._env.initialize_arm()
            self._set_target_from_case(case)
            cube_id = int(case.get("cube_id", 0))
            target_pos = self._array(case, "target_pos")
            target_quat = np.asarray(
                case.get("target_quat", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            self._env._data.joint(f"object_joint_{cube_id}").qpos[:3] = target_pos
            self._env._data.joint(f"object_joint_{cube_id}").qpos[3:] = target_quat
            self._env._data.qvel[:] = 0.0
            self._env.set_state(self._env._data.qpos.copy(), self._env._data.qvel.copy())
            return self._frame(self._env.render())
        finally:
            self._env._data.mocap_pos[:] = saved_target_pos
            self._env._data.mocap_quat[:] = saved_target_quat
            self._env.set_state(saved_qpos, saved_qvel)

    def _attach_target_info(self, info: dict[str, Any]) -> dict[str, Any]:
        if self._env is None:
            return info
        cube_id = int((self.manifest_case or {}).get("cube_id", 0))
        info["target_pos"] = self._env._data.mocap_pos[self._env._cube_target_mocap_ids[cube_id]].copy()
        info["target_quat"] = self._env._data.mocap_quat[self._env._cube_target_mocap_ids[cube_id]].copy()
        return info

    def _start_from_manifest_case(self, seed: Optional[int]) -> dict[str, Any]:
        if self.manifest_case is None:
            raise RuntimeError("manifest_case is required")
        case = self.manifest_case
        self._env = self._make_env()
        self._env.reset(seed=seed, options=self._reset_options())
        self._set_target_from_case(case)
        self._goal_image = self._render_goal_from_case(case)
        self._set_target_from_case(case)
        self._env.set_state(self._array(case, "qpos"), self._array(case, "qvel"))
        frame = self._frame(self._env.compute_observation())
        info = self._env.get_step_info()
        info["success"] = bool(getattr(self._env, "_success", False))
        info["manifest_case_id"] = case.get("case_id")
        info = self._attach_target_info(info)
        return {
            "image": frame,
            "image_raw": frame,
            "goal_image": self._goal_image,
            "goal_image_raw": self._goal_image,
            "info": dict(info),
            "reward": 0.0,
            "done": False,
            "truncated": False,
        }

    def start(self, seed: Optional[int] = None) -> dict[str, Any]:
        self.close()
        if self.manifest_case is not None:
            return self._start_from_manifest_case(seed)
        self._env = self._make_env()
        obs, info = self._env.reset(seed=seed, options=self._reset_options())
        frame = self._frame(obs)
        goal = info.get("goal_rendered")
        if goal is None:
            goal = info.get("goal", frame)
        # Distinct goal-frame case: OGBench returns a separate rendered goal image.
        self._goal_image = self._frame(goal)
        return {
            "image": frame,
            "image_raw": frame,
            "goal_image": self._goal_image,
            "goal_image_raw": self._goal_image,
            "info": dict(info),
            "reward": 0.0,
            "done": False,
            "truncated": False,
        }

    def step(self, action_sim) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("call start() before step()")
        action = np.asarray(action_sim, dtype=np.float64).reshape(5)
        action = np.clip(action, CUBE_ACTION_LOW, CUBE_ACTION_HIGH)
        obs, reward, terminated, truncated, info = self._env.step(action)
        frame = self._frame(obs)
        success = bool(info.get("success", False))
        info["success"] = success
        info = self._attach_target_info(info)
        return {
            "image": frame,
            "image_raw": frame,
            "goal_image": self._goal_image,
            "goal_image_raw": self._goal_image,
            "info": dict(info),
            "reward": float(reward),
            "done": bool(terminated or success),
            "truncated": bool(truncated),
        }

    def close(self) -> None:
        if self._env is None:
            return
        renderer = getattr(self._env, "_renderer", None)
        if renderer is not None:
            renderer.close()
            self._env._renderer = None
        if hasattr(self._env, "close_passive_viewer"):
            self._env.close_passive_viewer()
        self._env.close()
        self._env = None

