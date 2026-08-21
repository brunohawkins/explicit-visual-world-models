"""
Core abstract interfaces and shared data structures for the vdaworld pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of a single :class:`~vdaworld.core.agentic_generator.AgenticGenerator` run.

    Attributes:
        code: The final simulator source code produced by the agentic loop.
        stage_dir: Filesystem path to the agentic generation stage directory.
        tool_call_count: Number of sandbox tool calls made during generation.
        input_tokens: Total input tokens consumed by the VLM.
        output_tokens: Total output tokens produced by the VLM.
        cached_tokens: Total cached tokens used by the VLM.
        gate_passed: Verdict of the final fidelity gate on the shipped code.
            ``None`` if no gate ran (gate is opt-in); ``True`` if the shipped
            code passed all configured gate criteria; ``False`` if it failed any
            criterion (e.g. crash, poor worst-case reduction_ratio, or failed
            action-realizability check) when the run terminated.
        gate_summary: Human-readable per-trajectory gate table / complaint.
        keep_best_enabled: Whether harness-side checkpoint selection was enabled.
        shipped_checkpoint_tool_call: Tool-call index of the shipped checkpoint.
        shipped_worst_ratio: Worst training reduction_ratio for the shipped checkpoint.
        final_sandbox_worst_ratio: Worst ratio for the final sandbox before restore.
        keep_best_restored: Whether the sandbox was overwritten with a better checkpoint.
    """

    code: str
    stage_dir: str
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    api_retry_count: int = 0
    api_error_count: int = 0
    gate_passed: bool | None = None
    gate_summary: str = ""
    keep_best_enabled: bool = False
    shipped_checkpoint_tool_call: int | None = None
    shipped_worst_ratio: float | None = None
    final_sandbox_worst_ratio: float | None = None
    keep_best_restored: bool = False
