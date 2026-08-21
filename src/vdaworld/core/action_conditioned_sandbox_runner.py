"""Subprocess runner for action-conditioned simulators.

Parallel to :mod:`vdaworld.core.sandbox_runner` but for simulators that
subclass :class:`vdaworld.core.simulator.ActionConditionedSimulatorBase`.

Differences from the original runner:

* Loads ``ActionConditionedSimulatorBase`` into the module's globals
  (alongside ``SimulatorBase`` and ``WorldAPI``) so the generated code
  can ``import`` or just reference the base class as a module-global.
* Takes TWO image paths (start, goal) instead of one.
* Calls ``sim.fit(image_A, image_B)`` (2-arg).
* Drives the rollout via ``sim.update(a) + sim.render_frame()`` with a fixed
  test action (dataset-derived when the caller provides one, so its
  dimensionality matches the training actions). Goal: verify the sim compiles,
  fit works, update doesn't raise, render produces valid frames. CEM-driven
  planning is run separately, outside the agentic loop.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import linecache
import sys

import numpy as np
from PIL import Image

from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI
from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase


def _load_simulator_module(simulator_path: str):
    spec = importlib.util.spec_from_file_location("simulator_sandbox", simulator_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot create module spec from {simulator_path}")
    module = importlib.util.module_from_spec(spec)
    module.SimulatorBase = SimulatorBase
    module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase
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
    parser.add_argument("--start-image-path", required=True)
    parser.add_argument("--goal-image-path", required=True)
    parser.add_argument("--frames-npy", required=True)
    parser.add_argument("--debug-view-npy", default=None)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Pass None instead of WorldAPI to simulator",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--api-calls-dir", default=None)
    parser.add_argument(
        "--test-action",
        default=None,
        help=(
            "JSON list for the fixed per-step test action. Should be a real "
            "dataset action so its dimensionality matches training data. "
            "Defaults to the legacy 2-D [0.5, 0.5]."
        ),
    )
    args = parser.parse_args()

    world_api = (
        None
        if args.no_api
        else GeometryOnlyWorldAPI(
            cache_dir=args.cache_dir,
            api_calls_dir=args.api_calls_dir,
        )
    )
    module = _load_simulator_module(args.simulator_path)
    sim_class = getattr(module, args.simulator_class)

    simulator: ActionConditionedSimulatorBase = sim_class(
        api=world_api,
        fps=args.fps,
        frame_size=tuple(args.frame_size),
    )
    if not isinstance(simulator, ActionConditionedSimulatorBase):
        raise TypeError(
            f"{args.simulator_class} must subclass ActionConditionedSimulatorBase; "
            f"got {type(simulator).__mro__}"
        )

    image_A = np.array(Image.open(args.start_image_path).convert("RGB"))
    image_B = np.array(Image.open(args.goal_image_path).convert("RGB"))
    simulator.fit(image_A, image_B)

    if simulator.state is None or simulator.target_state is None:
        raise RuntimeError(
            "After fit(A, B), both self.state and self.target_state must be set."
        )

    w, h = args.frame_size
    arr = np.lib.format.open_memmap(
        args.frames_npy, mode="w+", dtype=np.uint8, shape=(args.n_frames, h, w, 3)
    )

    # Non-zero deterministic test action — exercises update(a) so degenerate
    # implementations (e.g. `def update(a): pass`) show as "loss does not change".
    # The caller passes a real dataset action so the dimensionality matches the
    # training action space; the legacy 2-D fallback only applies when absent.
    if args.test_action:
        test_action = np.asarray(json.loads(args.test_action), dtype=np.float64).reshape(-1)
    else:
        test_action = np.array([0.5, 0.5], dtype=np.float64)

    initial_loss = float(simulator.planning_objective())
    frame0 = simulator.render_frame()
    np.copyto(arr[0], frame0)
    losses = [initial_loss]
    for i in range(1, args.n_frames):
        simulator.update(test_action)
        np.copyto(arr[i], simulator.render_frame())
        losses.append(float(simulator.planning_objective()))
    arr.flush()

    print(
        f"[ac_sandbox_runner] saved {args.n_frames} frames to {args.frames_npy}",
        flush=True,
    )
    print(
        f"[ac_sandbox_runner] terminal_cost trajectory (test_action={test_action.tolist()}): "
        f"initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
        f"range=({min(losses):.4f}, {max(losses):.4f})",
        flush=True,
    )
    if losses[0] == losses[-1] and min(losses) == max(losses):
        print(
            "[ac_sandbox_runner] WARNING: terminal_cost is constant across the rollout. "
            "This suggests either update(a) does not mutate self.state, "
            "or terminal_cost does not depend on self.state. Both are degenerate.",
            flush=True,
        )

    if args.debug_view_npy and simulator._debug_view is not None:
        np.save(args.debug_view_npy, simulator._debug_view)
        print(
            f"[ac_sandbox_runner] saved debug view ({simulator._debug_view.shape}) "
            f"to {args.debug_view_npy}",
            flush=True,
        )

    sys.modules.pop("simulator_sandbox", None)
    gc.collect()
    linecache.clearcache()


if __name__ == "__main__":
    main()
