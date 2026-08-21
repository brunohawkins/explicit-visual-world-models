"""
Unified agentic generation loop.

:class:`AgenticGenerator` drives a single tool-calling session that writes,
compiles, and validates simulator code without external critic stages.  It
replaces the previous ``SceneCritic``, ``DynamicsCritic``, and ``ErrorCritic``
classes.
"""

from __future__ import annotations

import logging
import os

from vdaworld.api.llm.methods.prompt import get_prompt
from vdaworld.core.critic_toolbox import CriticSandbox
from vdaworld.core.interfaces import GenerationResult

logger = logging.getLogger(__name__)

# Minimal placeholder written to the sandbox at construction time so the file
# is never empty.  The agent's first action will be write_code_from_scratch.
_STUB_CODE = "# Placeholder — agent will call write_code_from_scratch\n"


class AgenticGenerator:
    """Runs a single agentic tool-calling loop that generates, tests, and
    validates simulator code in one continuous session.

    Args:
        vlm: A :class:`~vdaworld.api.vlm.VLMClient` instance.
        prompts_path: Directory containing the prompt ``.md`` files.
        max_turns: Hard cap on the number of model turns (each turn may contain
            unlimited tool calls).
        n_frames: Number of frames rendered per ``run_simulation()`` call.
        input_image_path: Path to the target/reference image.
        fps: Frames per second passed to the simulator.
        frame_size: ``(width, height)`` tuple for the simulator.
        simulator_class_name: Name of the class to instantiate from the
            generated module.
        output_dir: Root output directory (used for stage-level logging).
        cache_dir: Optional WorldAPI result cache directory.
        available_tools: Bullet-point string of available WorldAPI method
            names, injected into the prompt.
        caption: Natural-language task description, injected into the prompt.
    """

    def __init__(
        self,
        vlm,
        prompts_path: str,
        max_turns: int,
        n_frames: int,
        input_image_path: str,
        fps: int,
        frame_size: tuple[int, int],
        simulator_class_name: str,
        output_dir: str,
        cache_dir: str | None,
        available_tools: str,
        caption: str,
        provide_image: bool = True,
        restricted_tools: bool = False,
        no_api: bool = False,
        no_mhi: bool = False,
        prompt_suffix: str = "",
    ) -> None:
        self._vlm = vlm
        self._prompts_path = prompts_path
        self._max_turns = max_turns
        self._n_frames = n_frames
        self._input_image_path = input_image_path
        self._fps = fps
        self._frame_size = frame_size
        self._simulator_class_name = simulator_class_name
        self._output_dir = output_dir
        self._cache_dir = cache_dir
        self._available_tools = available_tools
        self._caption = caption
        self._provide_image = provide_image
        self._restricted_tools = restricted_tools
        self._no_api = no_api
        self._no_mhi = no_mhi
        self._prompt_suffix = str(prompt_suffix or "").strip()

    # ------------------------------------------------------------------
    # Hooks for subclasses (e.g. PlanningAgenticGenerator)
    # ------------------------------------------------------------------

    def _build_sandbox(
        self,
        sandbox_dir: str,
        tool_calls_log_dir: str,
        world_api_log_dir: str,
    ) -> CriticSandbox:
        """Construct the sandbox object. Override to use a custom sandbox class."""
        return CriticSandbox(
            code=_STUB_CODE,
            fps=self._fps,
            n_frames=self._n_frames,
            frame_size=self._frame_size,
            input_image_path=self._input_image_path,
            simulator_class_name=self._simulator_class_name,
            sandbox_dir=sandbox_dir,
            tool_calls_log_dir=tool_calls_log_dir,
            cache_dir=self._cache_dir,
            world_api_log_dir=world_api_log_dir,
            no_api=self._no_api,
        )

    def _make_finish_guard(self, sandbox: CriticSandbox):
        """Return a ``Callable[[int], str | None]`` finish-guard, or ``None``.

        The guard is invoked when the model voluntarily finishes (emits a turn
        with no tool calls), and may also be invoked by the VLM loop when a
        configured soft turn limit is reached. Returning a non-empty string
        keeps the conversation alive with that string as feedback. The base
        generator installs no guard (returns ``None``); subclasses override to
        enforce a final gate."""
        return None

    def _finish_guard_turn_budget(self) -> tuple[int, int]:
        """Return ``(turns_per_complaint, max_extra_turns)`` for guard feedback.

        The default preserves the historical hard turn cap. Planning subclasses
        can grant bounded headroom so a gate complaint injected at the turn
        limit leaves room for corrective tool calls.
        """
        return (0, 0)

    def _final_gate_record(self, sandbox: CriticSandbox):
        """Return ``(gate_passed: bool | None, gate_summary: str)`` recorded on
        the :class:`GenerationResult` after the loop ends (catches exhausted
        guard retry budgets and any terminal failure). Base: no gate ⇒
        ``(None, "")``."""
        return (None, "")

    def _build_tool_callables(self, sandbox: CriticSandbox) -> list:
        """Assemble the list of VLM-callable tools. Override to add tools."""
        if self._restricted_tools:
            tool_callables = [
                sandbox.get_api_documentation,
                sandbox.write_code_from_scratch,
            ]
        else:
            tool_callables = [
                sandbox.get_api_documentation,
                sandbox.write_code_from_scratch,
                sandbox.read_code,
                sandbox.check_compilation,
                sandbox.run_simulation,
                sandbox.read_terminal,
                sandbox.edit_code,
                sandbox.view_rendered_frame,
                sandbox.view_debug,
                sandbox.view_motion_history_image,
                sandbox.read_tool_call_metadata,
                sandbox.read_world_api_call_metadata,
            ]

        if self._no_api:
            tool_callables = [
                t for t in tool_callables if t.__name__ != "get_api_documentation"
            ]
        if self._no_mhi:
            tool_callables = [
                t for t in tool_callables if t.__name__ != "view_motion_history_image"
            ]
        return tool_callables

    def generate(self, target_image_path: str, stage_dir: str) -> GenerationResult:
        """Run the agentic generation loop and return the final simulator code.

        Creates a :class:`~vdaworld.core.critic_toolbox.CriticSandbox`, builds
        the prompt, and calls
        :meth:`~vdaworld.api.vlm.VLMClient.generate_agentic_reply` with all
        available sandbox tools.  After the loop ends, the contents of the
        sandbox file are returned as the final simulator code.

        Args:
            target_image_path: Path to the ground-truth reference image sent
                to the VLM as the first image in the conversation.
            stage_dir: Pipeline stage directory.  VLM interaction files and
                sandbox tool-call logs are written here.

        Returns:
            A :class:`~vdaworld.core.interfaces.GenerationResult` with the
            final code and metadata.
        """
        tool_calls_log_dir = os.path.join(stage_dir, "tool_calls")
        sandbox_dir = os.path.join(stage_dir, "sandbox")
        world_api_log_dir = os.path.join(sandbox_dir, "api_calls")
        os.makedirs(tool_calls_log_dir, exist_ok=True)

        guard_turn_extension, guard_max_extra_turns = self._finish_guard_turn_budget()
        prompt = get_prompt(
            self._prompts_path,
            "agentic_generation",
            available_tools=self._available_tools,
            caption=self._caption,
            simulator_class_name=self._simulator_class_name,
            max_turns=self._max_turns,
            frame_size=tuple(self._frame_size),
            fps=self._fps,
            finish_guard_turn_extension=guard_turn_extension,
            finish_guard_max_extra_turns=guard_max_extra_turns,
        )
        if self._prompt_suffix:
            prompt = f"{prompt.rstrip()}\n\n{self._prompt_suffix}\n"

        sandbox = self._build_sandbox(
            sandbox_dir=sandbox_dir,
            tool_calls_log_dir=tool_calls_log_dir,
            world_api_log_dir=world_api_log_dir,
        )

        logger.info(
            "AgenticGenerator: starting agentic loop (max_turns=%d)...", self._max_turns
        )
        gate_passed = None
        gate_summary = ""
        try:
            tool_callables = self._build_tool_callables(sandbox)
            finish_guard = self._make_finish_guard(sandbox)

            vlm_result = self._vlm.generate_agentic_reply(
                prompt=prompt,
                image_paths=[target_image_path] if self._provide_image else None,
                tool_callables=tool_callables,
                max_turns=self._max_turns,
                interaction_name="agentic_generation",
                finish_guard=finish_guard,
                finish_guard_turn_extension=guard_turn_extension,
                finish_guard_max_extra_turns=guard_max_extra_turns,
            )
            # Record the final gate verdict on the shipped code BEFORE cleanup.
            # This catches the path where the gate retry budget is exhausted, so
            # a crash or failed criterion is still flagged loudly rather than
            # masquerading as a result.
            gate_passed, gate_summary = self._final_gate_record(sandbox)
            with open(sandbox._sandbox_path, encoding="utf-8") as f:
                final_code = f.read()
        finally:
            sandbox.cleanup()

        tool_call_count = sandbox._tool_call_idx
        logger.info(
            "AgenticGenerator: completed after %d tool call(s) across %d turn(s).",
            tool_call_count,
            vlm_result.turns_used,
        )

        return GenerationResult(
            code=final_code,
            stage_dir=stage_dir,
            tool_call_count=tool_call_count,
            input_tokens=vlm_result.input_tokens,
            output_tokens=vlm_result.output_tokens,
            cached_tokens=vlm_result.cached_tokens,
            api_retry_count=vlm_result.api_retry_count,
            api_error_count=vlm_result.api_error_count,
            gate_passed=gate_passed,
            gate_summary=gate_summary,
        )
