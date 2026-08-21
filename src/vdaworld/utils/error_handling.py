"""
Error handling utilities for the vdaworld pipeline.

The key distinction is between errors that belong to the VLM-generated code
(correctable) and errors that originate in the pipeline or API backend
(not correctable, should abort and be examined by the developer).
"""
import json
import linecache
import os


class APIInternalError(Exception):
    """Raised by WorldAPI when a backend or infrastructure failure occurs.

    These errors are not the VLM's fault and should never be passed back to the
    VLM for correction — they indicate a problem the developer needs to fix
    (e.g. a microservice is down, a network request timed out).
    """


class AbortLoopError(BaseException):
    """Raised to immediately abort the agentic loop without passing the error to the VLM.

    Inherits from BaseException (not Exception) so it propagates through the
    ``except Exception`` tool-dispatch handlers in vlm.py unimpeded.
    Use this for infrastructure failures (e.g. APIInternalError from a subprocess)
    that are not fixable by the model.
    """


class StaticCheckError(Exception):
    """Raised when pre-execution static checks (syntax/lint) fail on generated code.

    Always correctable — the error is in the generated simulator code, not in
    the pipeline infrastructure.
    """


def trace_errors(exc: Exception, correctable_filename: str) -> dict:
    """Walk the exception traceback and extract the most relevant frame.

    Scans every frame in the traceback and keeps the last one whose source file
    matches *correctable_filename* (the generated simulator module).  If no such
    frame is found, falls back to the final frame and marks the trace as
    non-correctable.

    Args:
        exc: The caught exception.
        correctable_filename: Absolute path to the generated simulator file.

    Returns:
        A dict with keys:

        * ``"lineno"`` — line number of the relevant frame.
        * ``"cls"`` — name of the class whose method raised the error
          (``"Unknown"`` if not in an instance method).
        * ``"exception"`` — ``str(exc)``.
        * ``"line"`` — source code at that line (stripped).
        * ``"type"`` — ``type(exc).__name__``.
        * ``"correctable"`` — ``True`` iff the frame is inside the generated
          simulator module.
    """
    if isinstance(exc, StaticCheckError):
        return {
            "lineno": 0,
            "cls": "Unknown",
            "exception": str(exc),
            "line": "",
            "type": type(exc).__name__,
            "correctable": True,
        }

    tb = exc.__traceback__
    trace = None
    final_tb = tb

    # Walk the full traceback; keep the last frame from the generated simulator
    # (or simulator base class).  Also track the final frame for the fallback.
    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        lineno = tb.tb_lineno
        line = linecache.getline(filename, lineno).strip()

        is_simulator_gen = os.path.abspath(filename) == os.path.abspath(correctable_filename)
        is_core_simulator = os.path.abspath(filename).endswith(os.path.join("core", "simulator.py"))

        if is_simulator_gen or is_core_simulator:
            self_obj = frame.f_locals.get("self")
            trace = {
                "lineno": lineno,
                "cls": self_obj.__class__.__name__ if self_obj is not None else "Unknown",
                "exception": str(exc),
                "line": line,
                "type": type(exc).__name__,
                "correctable": True,
            }

        final_tb = tb
        tb = tb.tb_next

    if trace is None:
        # No frame matched the simulator file — fall back to the final frame.
        frame = final_tb.tb_frame
        trace = {
            "lineno": final_tb.tb_lineno,
            "cls": "Unknown",
            "exception": str(exc),
            "line": linecache.getline(frame.f_code.co_filename, final_tb.tb_lineno).strip(),
            "type": type(exc).__name__,
            "correctable": False,
        }

    return trace


def format_error_trace(trace: dict) -> str:
    """Format a trace dict as a human-readable string for the VLM error critic."""
    return (
        f"{trace['type']} at line {trace['lineno']} in {trace['cls']}:\n"
        f"  {trace['line']}\n"
        f"{trace['exception']}"
    )


def save_error_trace(trace: dict, output_file: str) -> None:
    """Persist a trace dict to *output_file* as JSON (``correctable`` key omitted)."""
    trace_copy = {k: v for k, v in trace.items() if k != "correctable"}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trace_copy, f, indent=4)
