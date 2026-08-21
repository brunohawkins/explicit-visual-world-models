"""
Standalone subprocess script used by :class:`~vdaworld.core.critic_toolbox.CriticSandbox`
to run a simulator module in isolation.

Invocation::

    python -m vdaworld.core.sandbox_runner \\
        --simulator-path /tmp/critic_sandbox_xxx/simulator_sandbox.py \\
        --simulator-class VideoSimulation \\
        --fps 30 \\
        --n-frames 150 \\
        --frame-size 1024 576 \\
        --image-path /path/to/target.png \\
        --frames-dir /tmp/critic_sandbox_xxx/frames

All stdout/stderr produced here (including debug ``print`` statements injected
into the simulator by the critic) are captured by the parent process.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import linecache
import os
import sys

import numpy as np
from PIL import Image

from vdaworld.core.api import WorldAPI
from vdaworld.core.simulator import SimulatorBase


def _load_simulator_module(simulator_path: str):
    spec = importlib.util.spec_from_file_location("simulator_sandbox", simulator_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot create module spec from {simulator_path}")
    module = importlib.util.module_from_spec(spec)
    module.SimulatorBase = SimulatorBase
    module.WorldAPI = WorldAPI
    sys.modules["simulator_sandbox"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator-path", required=True)
    parser.add_argument("--simulator-class", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument(
        "--frame-size", type=int, nargs=2, required=True, metavar=("W", "H")
    )
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--frames-npy", required=True)
    parser.add_argument("--debug-view-npy", default=None)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Pass None instead of WorldAPI to simulator",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--api-calls-dir", default=None)
    args = parser.parse_args()

    world_api = (
        None
        if args.no_api
        else WorldAPI(cache_dir=args.cache_dir, api_calls_dir=args.api_calls_dir)
    )
    module = _load_simulator_module(args.simulator_path)
    sim_class = getattr(module, args.simulator_class)

    simulator: SimulatorBase = sim_class(
        api=world_api,
        fps=args.fps,
        frame_size=tuple(args.frame_size),
    )

    input_img = np.array(Image.open(args.image_path).convert("RGB"))
    simulator.fit(input_img)

    with open(args.simulator_path, encoding="utf-8") as _f:
        _code = _f.read()
    if "mujoco" in _code.lower():
        next(simulator)  # discard the blank MuJoCo initialisation frame

    w, h = args.frame_size
    arr = np.lib.format.open_memmap(
        args.frames_npy, mode="w+", dtype=np.uint8, shape=(args.n_frames, h, w, 3)
    )
    simulator.run_simulation(args.n_frames, out_arr=arr)
    arr.flush()
    print(
        f"[sandbox_runner] saved {args.n_frames} frames to {args.frames_npy}",
        flush=True,
    )

    if args.debug_view_npy and simulator._debug_view is not None:
        np.save(args.debug_view_npy, simulator._debug_view)
        print(
            f"[sandbox_runner] saved debug view ({simulator._debug_view.shape}) to {args.debug_view_npy}",
            flush=True,
        )

    sys.modules.pop("simulator_sandbox", None)
    gc.collect()
    linecache.clearcache()


if __name__ == "__main__":
    main()
