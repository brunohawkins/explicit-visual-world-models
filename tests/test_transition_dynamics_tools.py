"""Focused safety and one-step calibration tests for dynamics codegen tools."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vdaworld.core.planning_agentic_generator import PlanningAgenticGenerator
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox


FRAME = 32
TRUE_GAIN = 0.75
DEFAULT_GAIN = 0.2


def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _render_state(position: np.ndarray, theta: float) -> Image.Image:
    """Encode compact continuous state in one RGB pixel."""
    image = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    image[0, 0, 0] = np.uint8(round(float(position[0]) * 4.0))
    image[0, 0, 1] = np.uint8(round(float(position[1]) * 4.0))
    phase = (_wrap_angle(theta) + np.pi) / (2.0 * np.pi)
    image[0, 0, 2] = np.uint8(round(np.clip(phase, 0.0, 1.0) * 255.0))
    return Image.fromarray(image)


def _build_dataset(root: Path, n_trajectories: int = 4, n_steps: int = 10) -> None:
    rng = np.random.default_rng(7)
    starts = [
        np.array([12.0, 12.0]),
        np.array([18.0, 13.0]),
        np.array([14.0, 19.0]),
        np.array([20.0, 20.0]),
    ]
    for trajectory_index in range(n_trajectories):
        trajectory_dir = root / f"trajectory_{trajectory_index:04d}"
        trajectory_dir.mkdir(parents=True)
        position = starts[trajectory_index].copy()
        theta = _wrap_angle(2.9 - 0.3 * trajectory_index)
        actions = rng.integers(-3, 4, size=(n_steps, 2)).astype(np.float64)
        _render_state(position, theta).save(trajectory_dir / "frame_0000.png")
        recorded = []
        for step, action in enumerate(actions):
            position = position + TRUE_GAIN * action
            theta = _wrap_angle(theta + 0.1 * TRUE_GAIN * action[0])
            _render_state(position, theta).save(
                trajectory_dir / f"frame_{step + 1:04d}.png"
            )
            recorded.append({"step": step, "action": action.tolist()})
        (trajectory_dir / "actions.json").write_text(
            json.dumps(recorded), encoding="utf-8"
        )


_COMPACT_SIMULATOR = f"""
import numpy as np


class Simulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=({FRAME}, {FRAME}), api=None, fps=10):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {{"gain": {DEFAULT_GAIN}}}

    @staticmethod
    def _parse(image):
        pixel = np.asarray(image, dtype=np.uint8)[0, 0]
        position = pixel[:2].astype(float) / 4.0
        theta = float(pixel[2]) / 255.0 * (2.0 * np.pi) - np.pi
        return position, theta

    def fit(self, image_A, image_B):
        position, theta = self._parse(image_A)
        target_position, target_theta = self._parse(image_B)
        self.state = {{
            "p": position,
            "theta": np.array([theta]),
            "velocity": np.zeros(2),
            "static_marker": np.array([3.0]),
        }}
        self.target_state = {{
            "p": target_position,
            "theta": np.array([target_theta]),
        }}

    def update(self, action):
        action = np.asarray(action, dtype=float)
        gain = self.params["gain"]
        self.state["p"] = self.state["p"] + gain * action
        theta = self.state["theta"][0] + 0.1 * gain * action[0]
        self.state["theta"] = np.array([np.arctan2(np.sin(theta), np.cos(theta))])
        self.state["velocity"] = action.copy()

    def loss_to_target(self):
        return float(np.linalg.norm(self.state["p"] - self.target_state["p"]))

    def render_frame(self):
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
"""


_GIANT_STATE_SIMULATOR = _COMPACT_SIMULATOR.replace(
    '            "static_marker": np.array([3.0]),',
    '            "static_marker": np.array([3.0]),\n'
    '            "mask": np.ones((1024, 1024), dtype=bool),\n'
    '            "image_state": np.zeros((256, 256), dtype=np.float32),\n'
    '            "oversized_vector": np.zeros(4096, dtype=np.float64),',
)


def _make_sandbox(
    tmp_path: Path,
    dataset: Path,
    code: str,
    *,
    with_logs: bool = True,
) -> PlanningCriticSandbox:
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0010.png"
    return PlanningCriticSandbox(
        code=code,
        fps=10,
        n_frames=10,
        frame_size=(FRAME, FRAME),
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        simulator_class_name="Simulator",
        sandbox_dir=str(tmp_path / "sandbox"),
        tool_calls_log_dir=str(tmp_path / "logs") if with_logs else None,
        no_api=True,
        gate_trajectories=[0, 1, 2, 3],
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


def test_transition_calibration_recovers_gain_with_held_out_errors(
    tmp_path: Path, dataset: Path
) -> None:
    sandbox = _make_sandbox(tmp_path, dataset, _COMPACT_SIMULATOR)
    source_path = Path(sandbox._sandbox_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    summary = sandbox.calibrate_transition_parameters(
        bounds={"gain": [0.1, 1.2]},
        trajectory_indices=[0, 1, 2, 3],
        max_transitions=24,
        budget=72,
    )

    assert "teacher-forced train transitions" in summary
    assert "HELD-OUT total" in summary
    assert "theta" in summary and "[wrapped angle]" in summary
    assert "velocity=latent_or_unobservable_rate" in summary
    assert "static_marker=static_in_sampled_transitions" in summary
    fitted_match = re.search(r"fitted params:\s+gain=([0-9.eE+-]+)", summary)
    assert fitted_match, summary
    assert float(fitted_match.group(1)) == pytest.approx(TRUE_GAIN, abs=0.04)

    totals = re.findall(r"(?:TRAIN|HELD-OUT) total:\s+([0-9.]+) -> ([0-9.]+)", summary)
    assert len(totals) == 2, summary
    assert all(float(after) < float(before) for before, after in totals)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash

    artifact = tmp_path / "logs" / "0_calibrate_transition_parameters"
    report_path = artifact / "transition_calibration.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["train_transition_count"] > 0
    assert report["held_out_transition_count"] > 0
    assert report_path.stat().st_size < 60_000


def test_estimator_skips_giant_state_without_large_output(
    tmp_path: Path, dataset: Path
) -> None:
    sandbox = _make_sandbox(tmp_path, dataset, _GIANT_STATE_SIMULATOR)

    summary = sandbox.estimate_dynamics_models(
        trajectory_indices=[0, 1],
        max_trajectories=2,
    )

    assert "boolean_or_mask" in summary
    assert "image_shaped" in summary
    assert "oversized" in summary
    assert '"state_key": "p"' in summary
    assert len(summary) < 14_000
    report_path = (
        tmp_path / "logs" / "0_estimate_dynamics_models" / "dynamics_models.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    skipped = report["skipped_components"]
    assert skipped["mask"]["reason"] == "boolean_or_mask"
    assert skipped["image_state"]["reason"] == "image_shaped"
    assert skipped["oversized_vector"]["reason"] == "oversized"
    assert report_path.stat().st_size < 60_000


def test_transition_calibration_is_exposed_by_generator(
    tmp_path: Path, dataset: Path
) -> None:
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0010.png"
    generator = PlanningAgenticGenerator(
        vlm=None,
        prompts_path="",
        max_turns=2,
        n_frames=10,
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        fps=10,
        frame_size=(FRAME, FRAME),
        simulator_class_name="Simulator",
        output_dir="",
        cache_dir=None,
        available_tools="",
        caption="",
        gate_trajectories=[0, 1],
    )
    sandbox = generator._build_sandbox(
        sandbox_dir=str(tmp_path / "generator_sandbox"),
        tool_calls_log_dir=str(tmp_path / "generator_logs"),
        world_api_log_dir=str(tmp_path / "api_logs"),
    )
    names = [tool.__name__ for tool in generator._build_tool_callables(sandbox)]
    assert "calibrate_transition_parameters" in names
    assert "calibrate_rigid_geometry" in names
    sandbox.cleanup()
