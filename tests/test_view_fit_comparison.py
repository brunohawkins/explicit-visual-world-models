"""Tests for the view_fit_comparison perception self-check tool (2026-07-07).

The tool runs the generated sim's ``fit()`` on a real training frame, renders,
and returns a labelled [TRUE | render | blend] side-by-side image so the VLM
can SEE whether its parse matches reality (motivated by mis-parsed Reacher
links / Push-T block poses shipping through fidelity-only feedback).
"""

from __future__ import annotations

import pytest
from PIL import Image

from tests.test_calibrate_parameters import (
    TRUE_GAIN,
    _GOOD_SIM,
    _build_dataset,
    _make_sandbox,
)

_TRUE_GAIN_SIM = _GOOD_SIM.replace(
    'self.params = {"gain": 0.2}',
    f'self.params = {{"gain": {TRUE_GAIN}}}',
)

_BROKEN_FIT_SIM = _TRUE_GAIN_SIM.replace(
    "    def fit(self, image_A, image_B):",
    "    def fit(self, image_A, image_B):\n"
    '        raise RuntimeError("fit exploded")',
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


def test_returns_labelled_comparison_image(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    result = sandbox.view_fit_comparison(0)
    assert isinstance(result, Image.Image), result
    # Three panels side by side, upscaled: wider than tall, and at least
    # 3x the raw frame width.
    assert result.width > result.height
    assert result.width >= 3 * 64


def test_defaults_goal_to_last_frame_and_accepts_cross_traj(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    result = sandbox.view_fit_comparison(
        1, frame_index=2, goal_trajectory_index=0, goal_frame_index=8
    )
    assert isinstance(result, Image.Image), result


def test_out_of_range_frame_returns_error_string(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    result = sandbox.view_fit_comparison(0, frame_index=999)
    assert isinstance(result, str)
    assert "[view_fit_comparison]" in result
    assert "out of range" in result


def test_broken_fit_surfaces_error_not_crash(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _BROKEN_FIT_SIM, dataset)
    result = sandbox.view_fit_comparison(0)
    assert isinstance(result, str)
    assert "fit()/render_frame() failed" in result
    assert "fit exploded" in result
