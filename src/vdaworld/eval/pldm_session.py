"""PLDM `DotWall` as a P2 test-env session.

Wraps the out-of-process FastAPI session server (`PLDMServer`) + HTTP client
(`WallSessionClient`) + image adapter behind the same minimal session interface
the generalised `eval_mpc` uses (`start/step/close`). This keeps `mpc.py`
env-agnostic; the gym-pusht path (`GymPushTSession`) implements the same
interface in-process.

PLDM actions are already in the simulator's action space (±2.45 per dim), so no
action scaling is needed here — `step` forwards the planned action verbatim.
"""

from __future__ import annotations

import socket
from typing import Any, Optional

import numpy as np

from pldm_envs.wall.sequential_eval import WallSessionClient

from vdaworld.eval.pldm_adapter import adapt_pldm_to_sim
from vdaworld.eval.pldm_server import PLDMServer

PLDM_FRAME_SIZE = (64, 64)
PLDM_ACTION_LOW = np.array([-2.45, -2.45], dtype=np.float64)
PLDM_ACTION_HIGH = np.array([2.45, 2.45], dtype=np.float64)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class PLDMSession:
    """Session over Felix's PLDM DotWall (eval mode by default)."""

    def __init__(
        self,
        *,
        mode: str = "eval",
        sim_frame_size: tuple[int, int] = (65, 65),
        port: Optional[int] = None,
        server: Optional[PLDMServer] = None,
        manifest_case: dict[str, Any] | None = None,
    ) -> None:
        self.mode = mode
        self.sim_frame_size = sim_frame_size
        self._port = port
        self._server = server
        self._own_server = server is None
        self._client: Optional[WallSessionClient] = None
        self._session_id: Optional[str] = None
        self._goal_sim = None
        self._goal_raw = None
        self._target_dot_position = [None, None]
        self.manifest_case = manifest_case

    def _pack(self, payload: dict, *, reward: float = float("nan")) -> dict[str, Any]:
        image_raw = adapt_pldm_to_sim(payload["image"], PLDM_FRAME_SIZE)
        image_sim = adapt_pldm_to_sim(payload["image"], self.sim_frame_size)
        info = dict(payload.get("info", {}))
        info["target_dot_position"] = self._target_dot_position
        if self.manifest_case is not None:
            info["manifest_case_id"] = self.manifest_case.get("case_id")
            info["manifest_start_state"] = self.manifest_case.get("start")
            info["manifest_goal_state"] = self.manifest_case.get("goal")
        return {
            "image": image_sim,
            "image_raw": image_raw,
            "goal_image": self._goal_sim,
            "goal_image_raw": self._goal_raw,
            "info": info,
            "reward": float(payload.get("reward", reward)) if payload.get("reward") is not None else reward,
            "done": bool(payload.get("done", False)),
            "truncated": bool(payload.get("truncated", False)),
        }

    def start(self, seed: Optional[int] = None) -> dict[str, Any]:
        if self.manifest_case is not None and self.manifest_case.get("env_seed") is not None:
            seed = int(self.manifest_case["env_seed"])
        if self._own_server:
            self._port = self._port or _free_port()
            self._server = PLDMServer(mode=self.mode, port=self._port)
            self._server.__enter__()
        self._client = WallSessionClient(base_url=self._server.base_url)
        session = self._client.start(seed=seed)
        self._session_id = session["session_id"]

        # Distinct goal-frame case: PLDM exposes the target frame directly.
        self._goal_raw = adapt_pldm_to_sim(session["info"]["target_position"], PLDM_FRAME_SIZE)
        self._goal_sim = adapt_pldm_to_sim(session["info"]["target_position"], self.sim_frame_size)

        # Locate the goal dot (coords usually absent → red-dot search in goal_raw).
        tdp = list(session["info"].get("target_position_xy")
                   or session["info"].get("target_position") or [])
        if not (isinstance(tdp, list) and len(tdp) == 2 and all(isinstance(v, (int, float)) for v in tdp)):
            g = self._goal_raw
            red = (g[..., 0] > 200) & (g[..., 1] < 80) & (g[..., 2] < 80)
            ys, xs = np.where(red)
            tdp = [float(xs.mean()), float(ys.mean())] if len(ys) else [None, None]
        self._target_dot_position = tdp

        return self._pack(session, reward=0.0)

    def step(self, action_sim) -> dict[str, Any]:
        if self._client is None or self._session_id is None:
            raise RuntimeError("call start() before step()")
        payload = self._client.step(self._session_id, np.asarray(action_sim).tolist())
        return self._pack(payload)

    def close(self) -> None:
        try:
            if self._client is not None and self._session_id is not None:
                self._client.close(self._session_id)
        except Exception:
            pass
        if self._own_server and self._server is not None:
            self._server.__exit__(None, None, None)
            self._server = None
