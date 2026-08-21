"""Adapters between gym-pusht native conventions and the P1 Push-T simulator.

gym-pusht is the test-time ground-truth env. Its action space and internal
state live in native pixel units ``[0, 512]``; the P1-generated simulator plans
in its configured image space. Every action the simulator plans must therefore
be scaled from that frame into native coordinates before stepping gym-pusht.

All coordinate conversion lives here so the MPC loop and session stay
coordinate-agnostic.
"""

from __future__ import annotations

import numpy as np

NATIVE_RANGE = 512.0
SIM_FRAME_PX = 96.0
SIM_PER_NATIVE = SIM_FRAME_PX / NATIVE_RANGE  # 0.1875
NATIVE_PER_SIM = NATIVE_RANGE / SIM_FRAME_PX  # 5.3333…


def _frame_extent(frame_size=(96, 96)) -> np.ndarray:
    values = np.asarray(frame_size, dtype=np.float64).reshape(-1)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"frame_size must contain two positive values, got {frame_size!r}")
    return values


def non_occluding_pusht_agent(
    target_pose,
    requested_agent,
) -> tuple[np.ndarray, bool]:
    """Preserve a safe pusher pose or move it clear of the target T."""
    target_xy = np.asarray(target_pose, dtype=np.float64).reshape(3)[:2]
    requested = np.asarray(requested_agent, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(requested - target_xy)) >= 145.0:
        return requested.copy(), False
    candidates = np.array(
        [[30.0, 30.0], [482.0, 30.0], [30.0, 482.0], [482.0, 482.0]],
        dtype=np.float64,
    )
    distances = np.linalg.norm(candidates - target_xy[None, :], axis=1)
    return candidates[int(np.argmax(distances))].copy(), True


def green_target_mask(frame) -> np.ndarray:
    """Return pixels belonging to Push-T's light-green target rendering."""
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Push-T frame must have shape (H, W, 3+), got {rgb.shape}")
    values = rgb[..., :3].astype(np.int16)
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    return (g >= 100) & (g >= r + 10) & (g >= b + 10)


def strip_green_target(frame) -> np.ndarray:
    """Remove only the green target from a simulator input frame.

    Push-T's background is white, so green-dominant target pixels—including
    antialiased edge pixels—can be replaced with white without modifying the
    grey-blue block or blue pusher. The input array is never mutated.
    """
    output = np.asarray(frame, dtype=np.uint8).copy()
    output[green_target_mask(output)] = np.array([255, 255, 255], dtype=np.uint8)
    return output


def sim_action_to_native(a, frame_size=(96, 96)) -> np.ndarray:
    """Simulator-frame agent target -> gym-pusht native ``[0,512]`` action."""
    scale = NATIVE_RANGE / _frame_extent(frame_size)
    return (np.asarray(a, dtype=np.float64) * scale).astype(np.float32)


def native_xy_to_sim(xy, *, frame_size=(96, 96), flip_y: bool = False) -> np.ndarray:
    """Native ``[0,512]`` coordinates -> simulator-frame coordinates.

    ``flip_y`` mirrors about the vertical axis (image row 0 at top vs a world
    y-up convention). Which one is correct is determined empirically by the
    Phase-1 sanity check; the caller passes the verified value.
    """
    xy = np.asarray(xy, dtype=np.float64)
    extent = _frame_extent(frame_size)
    out = xy * (extent / NATIVE_RANGE)
    if flip_y:
        out[..., 1] = extent[1] - out[..., 1]
    return out


def env_obs_to_sim_frame(obs, frame_size=(96, 96)) -> np.ndarray:
    """gym-pusht pixel observation -> uint8 RGB at the simulator's frame_size.

    Identity when the env already renders at frame_size (our case: 96x96).
    Nearest-neighbour resize otherwise, to preserve hard edges.
    """
    arr = np.asarray(obs, dtype=np.uint8)
    w, h = int(frame_size[0]), int(frame_size[1])
    if arr.shape[:2] != (h, w):
        from PIL import Image

        arr = np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST), dtype=np.uint8)
    return arr


def sim_action_bounds(frame_size=(96, 96)) -> tuple[np.ndarray, np.ndarray]:
    """CEM bounds in simulator-frame coordinates.

    The env's native action space is ``Box([0,512]^2)``. We expose the full
    physically reachable image extent, not the narrower training-action subset.
    """
    low = np.zeros(2, dtype=np.float64)
    high = _frame_extent(frame_size)
    return low, high
