"""Integration test for PLDMServer subprocess launcher.

Marked `integration` because it spawns a real uvicorn process and makes HTTP
calls. Slow (~5s) but the only way to verify the launch/health-check/teardown
contract end-to-end.
"""

from __future__ import annotations

import socket

import pytest
import requests

from vdaworld.eval.pldm_server import PLDMServer


def _free_port() -> int:
    """Bind to an ephemeral port, close, return the number. Race-prone but fine
    for a one-shot integration test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.integration
def test_server_starts_serves_and_shuts_down():
    port = _free_port()
    with PLDMServer(mode="eval", port=port, startup_timeout_s=20.0) as server:
        # Server should be reachable on the advertised base_url.
        r = requests.post(f"{server.base_url}/start", json={"seed": 0}, timeout=5.0)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "session_id" in body
        assert "image" in body
        assert "info" in body and "target_position" in body["info"]
        # Close the session cleanly.
        requests.post(
            f"{server.base_url}/close",
            json={"session_id": body["session_id"]},
            timeout=5.0,
        )

    # After context exit the server should no longer respond.
    with pytest.raises(requests.exceptions.RequestException):
        requests.post(f"http://127.0.0.1:{port}/start", json={}, timeout=1.0)
