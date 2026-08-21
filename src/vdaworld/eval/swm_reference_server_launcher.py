"""Subprocess launcher for the isolated Stable WorldModel reference server."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_SWM_PYTHON = PROJECT_ROOT / "le-wm" / ".venv-moge" / "bin" / "python"
DEFAULT_SERVER_SCRIPT = Path(__file__).with_name("swm_reference_server.py")


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return an ephemeral TCP port suitable for a local server launch."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class SWMReferenceServer:
    """Launch and clean up the evaluation-only SWM FastAPI service.

    An explicit ``python_path`` takes precedence over ``VDA_SWM_PYTHON``;
    otherwise the launcher uses the ``le-wm/.venv-moge`` interpreter next to
    this repository.  ``port=None`` or ``port=0`` selects an available port.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        python_path: str | Path | None = None,
        server_script: str | Path | None = None,
        startup_timeout_s: float = 120.0,
        poll_interval_s: float = 0.5,
        shutdown_grace_s: float = 5.0,
        request_timeout_s: float = 2.0,
    ) -> None:
        env_python = os.environ.get("VDA_SWM_PYTHON")
        selected_python = python_path or env_python or DEFAULT_SWM_PYTHON
        self.python_path = Path(selected_python).expanduser()
        self.server_script = Path(server_script or DEFAULT_SERVER_SCRIPT).expanduser()
        self.host = host
        self.port = find_free_port(host) if port in (None, 0) else int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        self.startup_timeout_s = float(startup_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.shutdown_grace_s = float(shutdown_grace_s)
        self.request_timeout_s = float(request_timeout_s)
        self._proc: subprocess.Popen[Any] | None = None
        self._log_file: Any = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def process(self) -> subprocess.Popen[Any] | None:
        return self._proc

    def command(self) -> list[str]:
        """Return the exact command used to launch the service."""

        return [
            str(self.python_path),
            str(self.server_script),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

    def process_environment(self) -> dict[str, str]:
        """Return the child environment, including headless MuJoCo rendering."""

        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _read_log(self) -> str:
        if self._log_file is None:
            return ""
        try:
            self._log_file.flush()
            return Path(self._log_file.name).read_text(
                encoding="utf-8", errors="replace"
            )
        except Exception:
            return ""

    def _poll_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    "SWM reference server exited early with code "
                    f"{self._proc.returncode}.\nServer log:\n{self._read_log()}"
                )
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=self.request_timeout_s,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if (
                        payload.get("status") == "ok"
                        and payload.get("backend") == "stable_worldmodel"
                    ):
                        return
                    last_error = RuntimeError(f"unexpected health payload: {payload!r}")
                else:
                    last_error = RuntimeError(
                        f"health probe returned HTTP {response.status_code}"
                    )
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            "SWM reference server did not become healthy in "
            f"{self.startup_timeout_s}s: {last_error}.\n"
            f"Server log:\n{self._read_log()}"
        )

    def start(self) -> "SWMReferenceServer":
        if self._proc is not None:
            raise RuntimeError("SWM reference server launcher is already started")
        if not self.python_path.is_file():
            raise FileNotFoundError(
                f"SWM Python interpreter does not exist: {self.python_path}"
            )
        if not self.server_script.is_file():
            raise FileNotFoundError(
                f"SWM reference server script does not exist: {self.server_script}"
            )

        self._log_file = tempfile.NamedTemporaryFile(
            prefix="swm_reference_server_",
            suffix=".log",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        self._proc = subprocess.Popen(
            self.command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=self.process_environment(),
            start_new_session=True,
        )
        try:
            self._poll_ready()
        except Exception:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        try:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=self.shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=2.0)
            self._proc = None
        finally:
            if self._log_file is not None:
                log_path = Path(self._log_file.name)
                try:
                    self._log_file.close()
                    log_path.unlink(missing_ok=True)
                finally:
                    self._log_file = None

    close = stop

    def __enter__(self) -> "SWMReferenceServer":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


# Descriptive alias for callers that prefer to make launcher ownership explicit.
SWMReferenceServerLauncher = SWMReferenceServer


__all__ = [
    "DEFAULT_SERVER_SCRIPT",
    "DEFAULT_SWM_PYTHON",
    "SWMReferenceServer",
    "SWMReferenceServerLauncher",
    "find_free_port",
]
