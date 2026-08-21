"""Benchmark registry for the shared smoke-test entrypoints.

The registry is the only place where benchmark descriptors live. Shared P1/P2
smoke scripts consume :class:`BenchmarkSpec` without branching on benchmark
identity; adding a future benchmark should be an append-only descriptor change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vdaworld.eval.pldm_session import (
    PLDM_ACTION_HIGH,
    PLDM_ACTION_LOW,
    PLDMSession,
)
from vdaworld.eval.ogbench_cube_env import (
    CUBE_ACTION_HIGH,
    CUBE_ACTION_LOW,
    OGBenchCubeSession,
)
from vdaworld.eval.pusht_adapter import sim_action_bounds
from vdaworld.eval.pusht_env import GymPushTSession
from vdaworld.eval.reacher_env import (
    REACHER_ACTION_HIGH,
    REACHER_ACTION_LOW,
    ReacherDMControlSession,
)
from vdaworld.eval.swm_two_room_env import (
    SWM_TWO_ROOM_ACTION_HIGH,
    SWM_TWO_ROOM_ACTION_LOW,
    SWM_TWO_ROOM_FRAME_SIZE,
    SWMTwoRoomSession,
)

SessionFactory = Callable[[], Any]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_project_root(repo_root: Path) -> Path:
    override = os.environ.get("VDA_PROJECT_ROOT") or os.environ.get("PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    parent = repo_root.resolve().parent
    if (parent / "Datasets").is_dir():
        return parent
    if parent.name == "vda_world_snapshots" and (parent.parent / "Datasets").is_dir():
        return parent.parent.resolve()
    return parent


PROJECT_ROOT = _resolve_project_root(REPO_ROOT)

SHARED_GATE_MAX_WORST_RATIO = 0.5
SANDBOX_FIT_TRAJECTORY = 0


@dataclass(frozen=True)
class BenchmarkSpec:
    """Declared benchmark facts consumed by generic smoke-test code."""

    dataset_dir: Path
    frame_size: tuple[int, int]
    n_frames: int
    sandbox_fit_frame_a: int
    sandbox_fit_frame_b: int
    deployed_action_bounds: tuple[np.ndarray, np.ndarray]
    held_out_traj: int
    p2_session_factory: SessionFactory
    p2_action_bounds: tuple[np.ndarray, np.ndarray]

    def sandbox_fit_paths(self) -> tuple[Path, Path]:
        """Return the training-frame paths used as sandbox fit(A, B) inputs."""
        traj = self.dataset_dir / f"trajectory_{SANDBOX_FIT_TRAJECTORY:04d}"
        return (
            traj / f"frame_{self.sandbox_fit_frame_a:04d}.png",
            traj / f"frame_{self.sandbox_fit_frame_b:04d}.png",
        )

    def deployed_bounds_copy(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.deployed_action_bounds
        return low.astype(np.float64).copy(), high.astype(np.float64).copy()

    def p2_bounds_copy(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.p2_action_bounds
        return low.astype(np.float64).copy(), high.astype(np.float64).copy()


PUSHT_FRAME_SIZE = (224, 224)
_pusht_low, _pusht_high = sim_action_bounds(PUSHT_FRAME_SIZE)


def _p2_not_wired(name: str):
    def _factory():
        raise NotImplementedError(f"{name} P2 not wired yet")

    return _factory


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "two_room": BenchmarkSpec(
        dataset_dir=PROJECT_ROOT
        / "Datasets"
        / "archive"
        / "two_room_curated_ablations"
        / "expert_only",
        frame_size=(65, 65),
        n_frames=11,
        sandbox_fit_frame_a=0,
        sandbox_fit_frame_b=10,
        deployed_action_bounds=(PLDM_ACTION_LOW.copy(), PLDM_ACTION_HIGH.copy()),
        held_out_traj=9,
        p2_session_factory=lambda: PLDMSession(mode="eval", sim_frame_size=(65, 65)),
        p2_action_bounds=(PLDM_ACTION_LOW.copy(), PLDM_ACTION_HIGH.copy()),
    ),
    "two_room_lewm": BenchmarkSpec(
        dataset_dir=PROJECT_ROOT
        / "Datasets"
        / "two_room_pldm_mirrored_swm_ablations_v1"
        / "expert_only",
        frame_size=SWM_TWO_ROOM_FRAME_SIZE,
        n_frames=51,
        sandbox_fit_frame_a=0,
        sandbox_fit_frame_b=25,
        deployed_action_bounds=(
            SWM_TWO_ROOM_ACTION_LOW.copy(),
            SWM_TWO_ROOM_ACTION_HIGH.copy(),
        ),
        held_out_traj=9,
        p2_session_factory=lambda: SWMTwoRoomSession(
            sim_frame_size=SWM_TWO_ROOM_FRAME_SIZE
        ),
        p2_action_bounds=(
            SWM_TWO_ROOM_ACTION_LOW.copy(),
            SWM_TWO_ROOM_ACTION_HIGH.copy(),
        ),
    ),
    "pusht": BenchmarkSpec(
        dataset_dir=PROJECT_ROOT
        / "Datasets"
        / "pusht_no_green_target_224"
        / "expert_only",
        frame_size=PUSHT_FRAME_SIZE,
        n_frames=49,
        sandbox_fit_frame_a=0,
        sandbox_fit_frame_b=48,
        deployed_action_bounds=(_pusht_low.copy(), _pusht_high.copy()),
        held_out_traj=6,
        p2_session_factory=lambda: GymPushTSession(sim_frame_size=PUSHT_FRAME_SIZE),
        p2_action_bounds=(_pusht_low.copy(), _pusht_high.copy()),
    ),
    "reacher": BenchmarkSpec(
        dataset_dir=PROJECT_ROOT
        / "Datasets"
        / "reacher_torque_target_expert_v1"
        / "expert_only",
        frame_size=(224, 224),
        n_frames=41,
        sandbox_fit_frame_a=0,
        sandbox_fit_frame_b=40,
        deployed_action_bounds=(
            np.array([-1.0, -1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        ),
        held_out_traj=9,
        p2_session_factory=lambda: ReacherDMControlSession(sim_frame_size=(224, 224)),
        p2_action_bounds=(
            REACHER_ACTION_LOW.copy(),
            REACHER_ACTION_HIGH.copy(),
        ),
    ),
    "cube": BenchmarkSpec(
        dataset_dir=PROJECT_ROOT
        / "Datasets"
        / "cube_curated_ablations_opaque"
        / "expert_only",
        frame_size=(200, 200),
        n_frames=11,
        sandbox_fit_frame_a=0,
        sandbox_fit_frame_b=10,
        deployed_action_bounds=(
            CUBE_ACTION_LOW.copy(),
            CUBE_ACTION_HIGH.copy(),
        ),
        held_out_traj=9,
        p2_session_factory=lambda: OGBenchCubeSession(sim_frame_size=(200, 200)),
        p2_action_bounds=(
            CUBE_ACTION_LOW.copy(),
            CUBE_ACTION_HIGH.copy(),
        ),
    ),
}


def available_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)


def get_benchmark(name: str) -> BenchmarkSpec:
    key = str(name)
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        choices = ", ".join(available_benchmarks())
        raise ValueError(f"unknown benchmark {key!r}; choose one of: {choices}") from exc
