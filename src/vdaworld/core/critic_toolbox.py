"""
Sandboxed tool environment for the agentic generator.

:class:`CriticSandbox` exposes tool methods that the VLM can call during
an interactive generation loop:

* :meth:`get_api_documentation` — retrieve WorldAPI method signatures and docstrings.
* :meth:`write_code_from_scratch` — replace the entire sandbox file.
* :meth:`check_compilation` — fast syntax check via py_compile.
* :meth:`run_simulation` — execute the sandbox simulator and capture output.
* :meth:`read_terminal` — return stdout/stderr from the last simulation run.
* :meth:`edit_code` — apply a SEARCH/REPLACE patch to the current code.
* :meth:`view_rendered_frame` — retrieve a specific frame image from the last run.
* :meth:`view_motion_history_image` — compute an MHI from the last run's frames.
* :meth:`view_debug` — return the last image logged via ``SimulatorBase.log_debug_view``.
* :meth:`read_tool_call_metadata` — inspect metadata from previous tool calls.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from PIL import Image

import numpy as np

from vdaworld.agent.patching import PatchError, UnifiedPatcher
from vdaworld.api.docs import get_api_documentation as _get_api_documentation
from vdaworld.utils import time_freq
from vdaworld.utils.error_handling import AbortLoopError

logger = logging.getLogger(__name__)

_RUNNER_MODULE = "vdaworld.core.sandbox_runner"


def _world_api_call_warning(call_name: str, folder_path: str) -> str:
    """Return a warning/error suffix for a WorldAPI call summary line.

    Implements warnings A (error), C (ground-plane inliers), D (cuboid fit),
    E (empty polygon).
    """
    metadata_path = os.path.join(folder_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return ""

    try:
        import json as _json

        with open(metadata_path, encoding="utf-8") as _f:
            meta = _json.load(_f)
    except Exception:
        return ""

    if meta.get("error"):
        return " [ERROR: raised exception]"

    if call_name == "predict_ground_plane":
        pct = meta.get("inliers_percentage")
        if pct is not None and pct < 0.1:
            return f" [WARNING: low plane inlier rate ({pct:.0%})]"
    elif call_name == "fit_3d_cuboid":
        frac = meta.get("inlier_fraction")
        if frac is not None and frac < 0.2:
            return f" [WARNING: poor cuboid fit ({frac:.0%} inliers)]"
    elif call_name == "generate_polygon_from_mask":
        if meta.get("points_count") == 0:
            return " [WARNING: empty polygon]"

    return ""


class CriticSandbox:
    """Manages a sandboxed copy of simulator code for the agentic critic.

    Args:
        code: The original simulator source code (read-only within the sandbox).
        fps: Frames per second used by the simulator.
        n_frames: Number of frames to render per simulation run.
        frame_size: ``(width, height)`` tuple for the simulator.
        input_image_path: Path to the target/reference image used by ``sim.fit``.
        simulator_class_name: The class name to instantiate from the module.
        no_api: If True, pass None instead of WorldAPI to the simulator.
    """

    def __init__(
        self,
        code: str,
        fps: int,
        n_frames: int,
        frame_size: tuple[int, int],
        input_image_path: str,
        simulator_class_name: str,
        sandbox_dir: str,
        tool_calls_log_dir: str | None = None,
        cache_dir: str | None = None,
        world_api_log_dir: str | None = None,
        no_api: bool = False,
    ) -> None:
        self._fps = fps
        self._n_frames = n_frames
        self._frame_size = frame_size
        self._input_image_path = input_image_path
        self._simulator_class_name = simulator_class_name
        self._tool_calls_log_dir = tool_calls_log_dir
        self._cache_dir = cache_dir
        self._world_api_log_dir = world_api_log_dir
        self._no_api = no_api

        self._sandbox_dir = sandbox_dir
        os.makedirs(self._sandbox_dir, exist_ok=True)
        self._sandbox_path = os.path.join(self._sandbox_dir, "simulator_sandbox.py")
        self._frames_npy = os.path.join(self._sandbox_dir, "frames.npy")
        self._debug_view_npy = os.path.join(self._sandbox_dir, "debug_view.npy")

        with open(self._sandbox_path, "w", encoding="utf-8") as f:
            f.write(code)

        self._last_stdout: str = ""
        self._last_stderr: str = ""
        self._tool_call_idx: int = 0
        self._tool_call_log: list[dict] = (
            []
        )  # in-memory fallback for read_tool_call_metadata

        logger.debug("CriticSandbox created at %s", self._sandbox_dir)

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------

    def _log_critic_tool_call(
        self,
        tool_name: str,
        args: dict,
        result_summary: str,
        extra_files: dict | None = None,
    ) -> None:
        """Persist a critic tool call to a numbered subfolder under tool_calls_log_dir.

        Each call gets its own ``{idx}_{tool_name}/`` directory containing:
        * ``metadata.json`` — tool name, arguments, result summary
        * Any caller-supplied extra files (text, bytes, or PIL Images)

        When ``self._suppress_tool_call_log`` is set (harness-internal quiet
        scoring, e.g. keep-best), the call is neither logged nor counted, so
        tool-call indices keep referring to VLM-visible calls only.
        """
        import json

        if getattr(self, "_suppress_tool_call_log", False):
            return

        metadata = {
            "index": self._tool_call_idx,
            "tool": tool_name,
            "args": {k: str(v)[:1000] for k, v in args.items()},
            "result_summary": result_summary[:500] if result_summary else "",
        }
        self._tool_call_log.append(metadata)
        self._tool_call_idx += 1

        if not self._tool_calls_log_dir:
            return

        folder_name = f"{self._tool_call_idx - 1}_{tool_name}"
        folder_path = os.path.join(self._tool_calls_log_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        with open(
            os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2)

        for filename, content in (extra_files or {}).items():
            path = os.path.join(folder_path, filename)
            if hasattr(content, "save"):  # PIL Image
                content.save(path)
            elif isinstance(content, bytes):
                with open(path, "wb") as f:
                    f.write(content)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(content))

    # ------------------------------------------------------------------
    # Tool methods exposed to the VLM
    # ------------------------------------------------------------------

    def run_simulation(self) -> str:
        """Execute the current working copy of the simulator and return terminal output.

        Runs the sandbox simulator for the configured number of frames and saves
        each frame to disk so they can be retrieved with view_rendered_frame.
        Returns the combined stdout and stderr, which will contain any print
        statements that were injected via edit_code.
        """
        # Remove stale frames and debug view from a previous run
        if os.path.exists(self._frames_npy):
            os.remove(self._frames_npy)
        if os.path.exists(self._debug_view_npy):
            os.remove(self._debug_view_npy)

        # Refresh the WorldAPI (subprocess) tool calls dir so each run starts clean
        if self._world_api_log_dir:
            if os.path.isdir(self._world_api_log_dir):
                shutil.rmtree(self._world_api_log_dir)
            os.makedirs(self._world_api_log_dir)

        cmd = [
            sys.executable,
            "-m",
            _RUNNER_MODULE,
            "--simulator-path",
            self._sandbox_path,
            "--simulator-class",
            self._simulator_class_name,
            "--fps",
            str(self._fps),
            "--n-frames",
            str(self._n_frames),
            "--frame-size",
            str(self._frame_size[0]),
            str(self._frame_size[1]),
            "--image-path",
            self._input_image_path,
            "--frames-npy",
            self._frames_npy,
        ]
        cmd += ["--debug-view-npy", self._debug_view_npy]
        if self._no_api:
            cmd += ["--no-api"]
        if self._cache_dir:
            cmd += ["--cache-dir", self._cache_dir]
        if self._world_api_log_dir:
            cmd += ["--api-calls-dir", self._world_api_log_dir]

        import time as _time

        logger.debug("CriticSandbox: running subprocess: %s", " ".join(cmd))
        _t = _time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"[sandbox] run_simulation subprocess: {_time.perf_counter() - _t:.1f}s")
        self._last_stdout = result.stdout
        self._last_stderr = result.stderr

        combined = ""
        if result.stdout:
            combined += result.stdout
        if result.stderr:
            combined += "\n--- stderr ---\n" + result.stderr

        if result.returncode != 0:
            combined += (
                f"\n[run_simulation] Process exited with code {result.returncode}"
            )

        extra_files: dict = {
            "terminal_output.txt": combined.strip() or "No output produced."
        }
        if os.path.exists(self._frames_npy):
            _frames_log = np.load(self._frames_npy)
            if len(_frames_log) > 0:
                extra_files["first_frame.png"] = Image.fromarray(_frames_log[0])
            if len(_frames_log) > 1:
                extra_files["last_frame.png"] = Image.fromarray(_frames_log[-1])

        self._log_critic_tool_call(
            tool_name="run_simulation",
            args={"n_frames": self._n_frames, "fps": self._fps},
            result_summary=combined[:300] if combined else "No output.",
            extra_files=extra_files,
        )

        if "APIInternalError" in self._last_stderr:
            logger.error(
                "APIInternalError detected in sandbox stderr — aborting agentic loop."
            )
            raise AbortLoopError(
                "APIInternalError in sandbox stderr — this is an infrastructure failure, "
                "not a code error. Agentic loop aborted."
            )

        return combined.strip() or "[run_simulation] No output produced."

    def read_terminal(self) -> str:
        """Return stdout and stderr from the most recent run_simulation call.

        Use this to inspect printed variable values or error messages without
        re-running the simulation.
        """
        combined = self._last_stdout
        if self._last_stderr:
            combined += "\n--- stderr ---\n" + self._last_stderr
        result = (
            combined.strip()
            or "[read_terminal] No terminal output available. Call run_simulation first."
        )

        self._log_critic_tool_call(
            tool_name="read_terminal",
            args={},
            result_summary=result[:300],
            extra_files={"terminal_output.txt": result},
        )

        return result

    def read_code(self) -> str:
        """Return the full current source of the sandbox simulator file.

        Use this to inspect the exact current state of the code — particularly
        useful before calling edit_code, to confirm the precise whitespace and
        indentation of the block you want to replace.

        Returns:
            The complete Python source code currently in the sandbox.
        """
        with open(self._sandbox_path, encoding="utf-8") as f:
            code = f.read()
        self._log_critic_tool_call(
            tool_name="read_code",
            args={},
            result_summary=f"Returned {len(code)} chars, {code.count(chr(10)) + 1} lines.",
        )
        return code

    def edit_code(self, search: str, replace: str) -> str:
        """Apply a targeted SEARCH/REPLACE patch to the current simulator code.

        Use this for incremental fixes and tuning — changing a parameter,
        fixing a bug, or injecting a print statement to inspect a value.
        For larger rewrites, prefer write_code_from_scratch.

        After editing, call run_simulation to execute the updated code and
        read_terminal to inspect any printed output.

        Args:
            search: The exact block of code to replace (must match verbatim,
                including whitespace and indentation).
            replace: The new code to insert in place of the search block.

        Returns:
            A confirmation message with the first line of the replacement, or
            an error message if the search block was not found.
        """
        with open(self._sandbox_path, encoding="utf-8") as f:
            current_code = f.read()

        patch_text = f"<<<<\n{search}\n====\n{replace}\n>>>>"
        try:
            new_code = UnifiedPatcher.apply_all_patches(current_code, patch_text)
        except PatchError as exc:
            error_result = f"[edit_code] Error: {exc}"
            self._log_critic_tool_call(
                tool_name="edit_code",
                args={"search": search, "replace": replace},
                result_summary=error_result,
                extra_files={"search.txt": search, "replace.txt": replace},
            )
            return error_result

        with open(self._sandbox_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        first_line = replace.splitlines()[0] if replace.strip() else "(empty)"
        result = f"[edit_code] Applied. Replacement starts with: {first_line!r}"

        self._log_critic_tool_call(
            tool_name="edit_code",
            args={"search": search, "replace": replace},
            result_summary=result,
            extra_files={"search.txt": search, "replace.txt": replace},
        )

        return result

    def read_tool_call_metadata(self, tool_call_index: int | None = None) -> str:
        """Return metadata from previous tool calls made during this evaluation.

        Use this to review what tools were called, what arguments were passed,
        and what results were returned — without re-running anything.

        Args:
            tool_call_index: Zero-based index of the specific tool call to
                inspect. If omitted, returns a summary of all tool calls made
                so far.

        Returns:
            A JSON-formatted string with metadata for the requested tool call(s),
            or an error string if the index is out of range.
        """
        import json

        if not self._tool_call_log:
            return "[read_tool_call_metadata] No tool calls have been made yet."

        if tool_call_index is not None:
            try:
                tool_call_index = int(tool_call_index)
            except ValueError:
                return f"[read_tool_call_metadata] Invalid tool_call_index: {tool_call_index}. Must be an integer."

        if tool_call_index is None:
            result = json.dumps(self._tool_call_log, indent=2)
        else:
            if tool_call_index < 0 or tool_call_index >= len(self._tool_call_log):
                return (
                    f"[read_tool_call_metadata] Index {tool_call_index} out of range. "
                    f"{len(self._tool_call_log)} call(s) available "
                    f"(indices 0–{len(self._tool_call_log) - 1})."
                )
            result = json.dumps(self._tool_call_log[tool_call_index], indent=2)

        self._log_critic_tool_call(
            tool_name="read_tool_call_metadata",
            args={"tool_call_index": str(tool_call_index)},
            result_summary=result[:300],
        )
        return result

    def read_world_api_call_metadata(self, call_index: int | None = None) -> str:
        """Return metadata from WorldAPI calls made during the last run_simulation.

        Each WorldAPI method call (segment, estimate_intrinsics, fit_3d_sphere, …)
        is logged to a numbered subfolder. Use this to inspect the actual return
        values — mask counts, 3D coordinates, fitted parameters, etc. — without
        re-running the simulation.

        Args:
            call_index: Zero-based index of the specific WorldAPI call to inspect.
                If omitted, returns a summary listing all calls made in the last run.

        Returns:
            JSON-formatted metadata for the requested call(s), or an error string.
        """
        if not self._world_api_log_dir or not os.path.isdir(self._world_api_log_dir):
            return "[read_world_api_call_metadata] No WorldAPI calls recorded. Call run_simulation first."

        call_folders = sorted(
            (
                d
                for d in os.listdir(self._world_api_log_dir)
                if os.path.isdir(os.path.join(self._world_api_log_dir, d))
            ),
            key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0,
        )
        if not call_folders:
            return "[read_world_api_call_metadata] No WorldAPI calls recorded. Call run_simulation first."

        if call_index is not None:
            try:
                call_index = int(call_index)
            except ValueError:
                return f"[read_world_api_call_metadata] Invalid call_index: {call_index}. Must be an integer."

        if call_index is None:
            lines = []
            for folder in call_folders:
                idx_str, _, call_name = folder.partition("_")
                warning = _world_api_call_warning(
                    call_name, os.path.join(self._world_api_log_dir, folder)
                )
                lines.append(f'{idx_str}: "{call_name}"{warning}')
            result = "\n".join(lines)
        else:
            if call_index < 0 or call_index >= len(call_folders):
                return (
                    f"[read_world_api_call_metadata] Index {call_index} out of range. "
                    f"{len(call_folders)} call(s) available (indices 0–{len(call_folders) - 1})."
                )
            folder = call_folders[call_index]
            metadata_path = os.path.join(
                self._world_api_log_dir, folder, "metadata.json"
            )
            if not os.path.exists(metadata_path):
                return f"[read_world_api_call_metadata] No metadata.json found for call {call_index} ({folder})."
            with open(metadata_path, encoding="utf-8") as f:
                result = f.read()

        self._log_critic_tool_call(
            tool_name="read_world_api_call_metadata",
            args={"call_index": str(call_index)},
            result_summary=result[:300],
        )
        return result

    def get_api_documentation(self, method_names: list[str]) -> str:
        """Retrieve the full signature and docstring for one or more WorldAPI methods.

        Call this before writing any simulator code that uses the API, so you
        know the exact argument types and return values.

        Args:
            method_names: List of WorldAPI method names to look up, e.g.
                ``["estimate_3d_points", "fit_2d_primitive"]``.

        Returns:
            A string containing the signature and docstring for each requested
            method, or an error message for any name not found.
        """
        if isinstance(method_names, str):
            method_names = [method_names]
        docs: list[str] = []
        allowed: list[str] = []
        for name in method_names:
            if str(name) == "segment":
                docs.append(
                    "Method 'segment' is disabled for action-conditioned generated "
                    "simulators. Use segment_image_with_gemini / "
                    "segment_trajectory_with_gemini during code generation, then "
                    "bake the resulting parsing logic into fit()."
                )
            else:
                allowed.append(str(name))
        if allowed:
            docs.append(_get_api_documentation(allowed))
        result = "\n\n".join(d for d in docs if d)
        self._log_critic_tool_call(
            "get_api_documentation",
            {"method_names": method_names},
            result[:300],
            extra_files={"documentation.txt": result},
        )
        return result

    def write_code_from_scratch(self, code: str) -> str:
        """Replace the entire sandbox file with the provided source code.

        Use this to write the initial simulator implementation, or to start
        completely fresh when the current working copy is too broken to fix
        with targeted edits.  Unlike edit_code, this replaces the
        full file contents rather than applying a SEARCH/REPLACE patch.

        Args:
            code: Complete Python source code for the new simulator.

        Returns:
            A confirmation message with the number of lines written.
        """
        with open(self._sandbox_path, "w", encoding="utf-8") as f:
            f.write(code)
        n_lines = code.count("\n") + 1
        msg = f"[write_code_from_scratch] Written ({n_lines} lines)."
        self._log_critic_tool_call(
            "write_code_from_scratch",
            {"code": code[:200] + ("…" if len(code) > 200 else "")},
            msg,
        )
        return msg

    def check_compilation(self) -> str:
        """Check the current sandbox code for syntax errors without running it.

        Runs ``python -m py_compile`` on the sandbox file.  This is a fast
        syntax-only check — it does not import or execute the module, so
        ``WorldAPI`` and ``SimulatorBase`` availability do not affect the result.

        Call this after write_code_from_scratch or edit_code to verify
        syntax before attempting a full run_simulation.

        Returns:
            ``"OK"`` if the code compiles without errors, or the syntax error
            message if it does not.
        """
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", self._sandbox_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result = "OK" if proc.returncode == 0 else (proc.stderr or proc.stdout).strip()
        self._log_critic_tool_call("check_compilation", {}, result)
        return result

    def view_debug(self) -> Image.Image | str:
        """Return the last debug image logged by the simulator via log_debug_view().

        Simulators can call ``self.log_debug_view(image)`` at any point during
        ``fit()``, ``render_frame()``, or ``update_simulation()`` to expose an
        auxiliary view — for example a component-only render, an annotated
        overlay highlighting detected objects or fitted geometry, or an
        intermediate state that is hard to infer from the normal output frames.
        Only the **last** image passed to ``log_debug_view`` is retained; call
        this tool immediately after ``run_simulation()`` to retrieve it.

        Use this when something about the simulation looks wrong or needs
        refinement and the normal rendered frames do not give enough detail to
        diagnose why.

        Returns:
            The debug image as a PIL Image, or an error string if no debug
            view has been logged yet.
        """
        if not os.path.exists(self._debug_view_npy):
            result_str = (
                "[view_debug] No debug view available. "
                "The simulator must call self.log_debug_view(image) to populate this."
            )
            self._log_critic_tool_call("view_debug", {}, result_str)
            return result_str

        arr = np.load(self._debug_view_npy)
        img = Image.fromarray(arr)
        result_str = f"[view_debug] Returning debug view ({img.width}x{img.height})."
        self._log_critic_tool_call(
            "view_debug",
            {},
            result_str,
            extra_files={"debug_view.png": img},
        )
        return img

    def view_motion_history_image(self) -> Image.Image | str:
        """Compute and return a Motion History Image (MHI) from the last simulation run.

        The MHI encodes temporal dynamics as a single image:
        - **Colour** indicates *when* motion occurred: blue = early frames,
          purple = mid frames, red = late frames.
        - **Saturation** indicates *how often* a pixel changed: white = frequent
          change, fully saturated = rare change.

        Use this after run_simulation() to verify that the physics dynamics
        are correct — e.g. that objects move in the right direction, at the
        right speed, and for the right duration.

        Returns:
            The MHI as a PIL Image, or an error string if no frames are available.
        """
        if not os.path.exists(self._frames_npy):
            result_str = "[view_motion_history_image] No frames found. Call run_simulation first."
            self._log_critic_tool_call("view_motion_history_image", {}, result_str)
            return result_str

        import time as _time

        _t0 = _time.perf_counter()
        frames_arr = np.load(self._frames_npy)
        _t1 = _time.perf_counter()
        print(f"[MHI] np.load frames ({frames_arr.shape}): {_t1 - _t0:.3f}s")

        if frames_arr.shape[0] == 0:
            result_str = "[view_motion_history_image] No frames available. Call run_simulation first."
            self._log_critic_tool_call("view_motion_history_image", {}, result_str)
            return result_str

        mhi_arr = time_freq(frames_arr)
        _t2 = _time.perf_counter()
        print(f"[MHI] time_freq: {_t2 - _t1:.3f}s")

        mhi_img = Image.fromarray(mhi_arr)
        _t3 = _time.perf_counter()
        print(f"[MHI] Image.fromarray: {_t3 - _t2:.3f}s")
        print(f"[MHI] total: {_t3 - _t0:.3f}s  (frames shape: {frames_arr.shape})")

        result_str = f"[view_motion_history_image] MHI computed from {frames_arr.shape[0]} frames."
        self._log_critic_tool_call(
            "view_motion_history_image",
            {"n_frames": frames_arr.shape[0]},
            result_str,
            extra_files={"mhi.png": mhi_img},
        )
        return mhi_img

    def view_rendered_frame(self, frame_index: int) -> Image.Image | str:
        """Return a specific rendered frame image from the most recent simulation run.

        Args:
            frame_index: Zero-based index of the frame to retrieve.

        Returns:
            The PIL Image for that frame, or an error string if unavailable.
        """
        try:
            frame_index = int(frame_index)
        except ValueError:
            return f"[view_rendered_frame] Invalid frame_index: {frame_index}. Must be an integer."

        if not os.path.exists(self._frames_npy):
            result_str = (
                "[view_rendered_frame] No frames found. Call run_simulation first."
            )
            self._log_critic_tool_call(
                tool_name="view_rendered_frame",
                args={"frame_index": frame_index},
                result_summary=result_str,
            )
            return result_str

        frames_arr = np.load(self._frames_npy)
        available = frames_arr.shape[0]
        if frame_index < 0 or frame_index >= available:
            result_str = (
                f"[view_rendered_frame] Frame {frame_index} not found. "
                f"{available} frame(s) available (indices 0–{available - 1}). "
                "Call run_simulation first."
            )
            self._log_critic_tool_call(
                tool_name="view_rendered_frame",
                args={"frame_index": frame_index},
                result_summary=result_str,
            )
            return result_str

        img = Image.fromarray(frames_arr[frame_index])
        self._log_critic_tool_call(
            tool_name="view_rendered_frame",
            args={"frame_index": frame_index},
            result_summary=f"Returned frame {frame_index} ({img.width}x{img.height})",
            extra_files={"frame.png": img},
        )
        return img

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the temporary sandbox directory."""
        shutil.rmtree(self._sandbox_dir, ignore_errors=True)
        logger.debug("CriticSandbox cleaned up: %s", self._sandbox_dir)
