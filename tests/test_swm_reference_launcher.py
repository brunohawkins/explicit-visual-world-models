"""Unit tests for the SWM reference subprocess launcher."""

from __future__ import annotations

from pathlib import Path

from vdaworld.eval.swm_reference_server_launcher import (
    DEFAULT_SERVER_SCRIPT,
    DEFAULT_SWM_PYTHON,
    SWMReferenceServer,
)


def test_launcher_uses_sibling_lewm_python_by_default(monkeypatch):
    monkeypatch.delenv("VDA_SWM_PYTHON", raising=False)
    launcher = SWMReferenceServer(port=43123)

    expected = (
        Path(__file__).resolve().parents[2] / "le-wm" / ".venv-moge" / "bin" / "python"
    )
    assert DEFAULT_SWM_PYTHON == expected
    assert launcher.python_path == expected
    assert launcher.server_script == DEFAULT_SERVER_SCRIPT
    assert launcher.command() == [
        str(expected),
        str(DEFAULT_SERVER_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        "43123",
    ]


def test_launcher_honours_python_override_and_sets_egl(monkeypatch, tmp_path):
    overridden = tmp_path / "reference-python"
    monkeypatch.setenv("VDA_SWM_PYTHON", str(overridden))
    launcher = SWMReferenceServer(port=43124)

    assert launcher.python_path == overridden
    child_env = launcher.process_environment()
    assert child_env["MUJOCO_GL"] == "egl"
    assert child_env["PYTHONUNBUFFERED"] == "1"


def test_explicit_python_path_precedes_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VDA_SWM_PYTHON", str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"
    launcher = SWMReferenceServer(port=43125, python_path=explicit)

    assert launcher.python_path == explicit
