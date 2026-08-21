"""Frozen evaluation profiles used for reproducible benchmark comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationProfile:
    """Resolved controller and case contract for one evaluation profile."""

    profile_id: str
    num_eval: int
    goal_offset_steps: int
    horizon: int
    apply_steps: int
    max_episode_steps: int
    cem_population: int
    cem_iters: int
    cem_elite_frac: float
    cem_init_std_scale: float
    backend: str
    solver_implementation: str
    alignment_scope: str

    @property
    def cem_topk(self) -> int:
        return max(1, int(round(self.cem_population * self.cem_elite_frac)))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cem_topk"] = self.cem_topk
        return payload


LEWM_ALIGNED_N50_V1 = EvaluationProfile(
    profile_id="lewm_aligned_n50_v1",
    num_eval=50,
    goal_offset_steps=25,
    horizon=25,
    apply_steps=25,
    max_episode_steps=50,
    cem_population=300,
    cem_iters=30,
    cem_elite_frac=0.1,
    cem_init_std_scale=1.0,
    backend="swm_reference",
    solver_implementation="vdaworld_cem_v1",
    alignment_scope=(
        "LeWM cases, SWM environments, success predicates, action budget, and "
        "CEM hyperparameters; VDA solver implementation and explicit-model "
        "action representation remain method-specific."
    ),
)

# Two-Room and Cube use the task-qualified sampler
# (lewm_matched_task_qualified_n50_v1): the LeWM valid-start rule followed by a
# task filter that drops cases which cannot test the intended capability
# (same-room Two-Room pairs; Cube pairs already inside the success radius).
# Push-T and Reacher use the unfiltered sampler (lewm_aligned_n50_v1).
LEWM_ALIGNED_MANIFEST_SHA256 = {
    "pusht": "c37c84d5ae164a1b339f366c9bbdb18a44f4336da3a2417ba1924b747b05bebb",
    "two_room_swm": "195f1991a12f2fafddd4e14b4c59e74d1026721255fdbb5e282420f733ccdcb3",
    "reacher": "e36c85c1c9e2c7c8b700716147525688a3216ad9e99353aa51e0788137a9c3e1",
    "cube": "66a6970987eef928a2b9a93b3a5b39d9f7cd98fd70c84207a051b42a2802c5d6",
}


__all__ = [
    "EvaluationProfile",
    "LEWM_ALIGNED_MANIFEST_SHA256",
    "LEWM_ALIGNED_N50_V1",
]
