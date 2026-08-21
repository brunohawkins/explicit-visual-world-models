"""Subprocess launcher for Felix's PLDM session server.

Usage:
    with PLDMServer(mode="eval", port=8001) as server:
        client = WallSessionClient(base_url=server.base_url)
        session = client.start(seed=42)
        ...
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional

import requests


class PLDMServer:
    """Context manager that spawns Felix's `pldm_envs.wall.session_server` and
    tears it down on exit. Polls `POST /start` until the server is healthy."""

    def __init__(
        self,
        mode: str = "eval",
        host: str = "127.0.0.1",
        port: int = 8001,
        startup_timeout_s: float = 120.0,
        poll_interval_s: float = 0.5,
        shutdown_grace_s: float = 5.0,
    ):
        if mode not in {"eval", "val", "train"}:
            raise ValueError(f"mode must be one of eval/val/train, got {mode!r}")
        self.mode = mode
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.shutdown_grace_s = shutdown_grace_s
        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _read_log(self) -> str:
        if self._log_file is None:
            return ""
        try:
            self._log_file.flush()
            with open(self._log_file.name, "r") as f:
                return f.read()
        except Exception:
            return ""

    def _poll_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"PLDM server exited early with code {self._proc.returncode}.\n"
                    f"Server log:\n{self._read_log()}"
                )
            try:
                r = requests.post(f"{self.base_url}/start", json={}, timeout=2.0)
                if r.status_code == 200:
                    # Clean up the probe session so we don't leak it.
                    sid = r.json().get("session_id")
                    if sid:
                        try:
                            requests.post(
                                f"{self.base_url}/close",
                                json={"session_id": sid},
                                timeout=2.0,
                            )
                        except Exception:
                            pass
                    return
                last_err = RuntimeError(f"probe got status {r.status_code}")
            except requests.exceptions.RequestException as e:
                last_err = e
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"PLDM server did not become healthy in {self.startup_timeout_s}s: {last_err}.\n"
            f"Server log:\n{self._read_log()}"
        )

    def __enter__(self) -> "PLDMServer":
        cmd = [
            sys.executable,
            "-m",
            "pldm_envs.wall.session_server",
            "--mode",
            self.mode,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self._log_file = tempfile.NamedTemporaryFile(
            prefix="pldm_server_", suffix=".log", delete=False, mode="w"
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._poll_ready()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._proc is None:
                return
            if self._proc.poll() is not None:
                self._proc = None
                return
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                self._proc = None
                return
            try:
                self._proc.wait(timeout=self.shutdown_grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=2.0)
            self._proc = None
        finally:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                    os.unlink(self._log_file.name)
                except Exception:
                    pass
                self._log_file = None
