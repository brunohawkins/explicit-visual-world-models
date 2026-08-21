"""
Vision-Language Model client supporting Gemini (Google GenAI) and VLLM
(OpenAI-compatible local server) backends.

Select the backend via the ``backend`` constructor argument:
  - ``"gemini"`` (default): uses the Google GenAI SDK.
  - ``"vllm"``: uses the OpenAI SDK pointed at a local VLLM server.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from vdaworld.api.docs import get_api_documentation

logger = logging.getLogger(__name__)


@dataclass
class AgenticReplyResult:
    """Return value of :meth:`VLMClient.generate_agentic_reply`."""

    text: str
    tool_call_count: int = 0  # total individual tool calls across all turns
    turns_used: int = 0  # number of model turns consumed
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    api_retry_count: int = 0
    api_error_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _image_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return (
        f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"
    )


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_openai_reasoning_text(message) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return ""
    return reasoning if isinstance(reasoning, str) else str(reasoning)


def _callable_to_openai_tool(fn) -> dict:
    """Convert a Python callable to an OpenAI tool schema dict.

    Extracts the function description from the first docstring line, parses
    per-parameter descriptions from the Args/Arguments section, and maps
    Python type annotations to JSON Schema types (including list[T] and X|None).
    """
    import inspect
    import re
    import types as _types

    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n")[0] if doc else fn.__name__

    _primitive = {int: "integer", float: "number", bool: "boolean", str: "string"}

    def _ann_to_schema(ann) -> dict:
        if isinstance(ann, _types.UnionType):
            non_none = [a for a in ann.__args__ if a is not type(None)]
            return _ann_to_schema(non_none[0]) if non_none else {"type": "string"}
        origin = getattr(ann, "__origin__", None)
        if origin is list:
            item_args = getattr(ann, "__args__", (str,))
            return {"type": "array", "items": _ann_to_schema(item_args[0])}
        return {"type": _primitive.get(ann, "string")}

    # Parse the Args/Arguments section for per-parameter descriptions
    param_descs: dict[str, str] = {}
    lines = doc.split("\n")
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(args|arguments)\s*:$", stripped, re.IGNORECASE):
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not line[0].isspace():
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            break
        param_match = re.match(r"^    (\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            rest = param_match.group(2).strip()
            current_desc = [rest] if rest else []
        elif current_param and line.startswith("        "):
            current_desc.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc).strip()

    properties: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        schema = (
            _ann_to_schema(ann)
            if ann is not inspect.Parameter.empty
            else {"type": "string"}
        )
        desc = param_descs.get(name, "")
        if desc:
            schema["description"] = desc
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class VLMClient:
    """Thin wrapper around the Google GenAI or VLLM chat API.

    All pipeline interactions with the VLM go through this class so that
    model selection, authentication, and tool registration are managed in a
    single place.

    Args:
        model_name: Model identifier (Gemini name or HuggingFace ID).
        temperature: Sampling temperature (0.0 = deterministic).
        backend: ``"gemini"`` (default) or ``"vllm"``.
        base_url: Base URL for the VLLM OpenAI-compatible server,
            e.g. ``"http://127.0.0.1:8000/v1"``.  Required when
            ``backend="vllm"``.
        llm_interactions_dir: Directory for logging VLM interactions.
    """

    _SYSTEM_INSTRUCTION = (
        "You are an expert action-conditioned world-model programming agent. Write a "
        "Python simulator that identifies state, action semantics, dynamics, and target "
        "semantics from the offline image-action trajectories exposed by the tools. "
        "Do not assume a caption, attached reference image, deployment environment, or "
        "WorldAPI method is available unless the run explicitly provides it. Dynamics "
        "must depend on actions and current state, never on the desired outcome.\n\n"
        "Work iteratively and incrementally: implement one component at a time, verify "
        "it using the available tools, then proceed to the next.\n\n"
        "Robust API usage: after every WorldAPI call inside fit(), check whether the "
        "result is usable before proceeding. An API call may succeed but return an "
        "empty or degenerate result (e.g. an empty array, zero-magnitude values, fewer "
        "items than expected, or None). When this happens, use a fallback supported by "
        "measurements from the trajectories rather than an unmeasured large effect. Guard "
        "against degenerate results (e.g. `if result is None or len(result) == 0:`), "
        "but do NOT suppress exceptions from incorrect API usage. Exceptions from "
        "wrong argument types or violated API contracts must propagate — never wrap "
        "an entire API call in a bare `except Exception: pass`."
    )

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
        backend: str = "gemini",
        base_url: Optional[str] = None,
        llm_interactions_dir: Optional[str | pathlib.Path] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.backend = backend
        self.base_url = base_url
        self._interaction_count = 0
        self._api_retry_count = 0
        self._api_error_count = 0
        self.llm_interactions_dir = (
            pathlib.Path(llm_interactions_dir) if llm_interactions_dir else None
        )
        self._stage_dir: Optional[pathlib.Path] = None

        if backend == "gemini":
            from google import genai

            # api_key = os.environ.get("GEMINI_API_KEY")
            self._gemini_client = genai.Client(
                vertexai=True, project="inverse-render", location="global"
            )
        elif backend == "gemini_api":
            from google import genai

            self._gemini_client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"), vertexai=False
            )
        elif backend == "vllm":
            if not base_url:
                raise ValueError("base_url is required when backend='vllm'")
            from openai import OpenAI

            self._openai_client = OpenAI(base_url=base_url, api_key="EMPTY")
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose 'gemini' or 'vllm'."
            )

    def set_stage_dir(self, stage_dir: Optional[str | pathlib.Path]) -> None:
        """Set the directory for logging the current pipeline stage's VLM interaction."""
        self._stage_dir = pathlib.Path(stage_dir) if stage_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        use_tools: bool = False,
        interaction_name: Optional[str] = None,
    ) -> str:
        """Send a prompt (and optional images) to the VLM and return the text response."""
        logger.debug(
            "VLMClient.generate_reply: backend=%s model=%s images=%d use_tools=%s",
            self.backend,
            self.model_name,
            len(image_paths) if image_paths else 0,
            use_tools,
        )
        if self.backend in ("gemini", "gemini_api"):
            return self._generate_reply_gemini(
                prompt, image_paths, use_tools, interaction_name
            )
        else:
            return self._generate_reply_vllm(prompt, image_paths, interaction_name)

    def generate_agentic_reply(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        tool_callables: Optional[list] = None,
        max_turns: int = 5,
        interaction_name: Optional[str] = None,
        finish_guard: Optional[callable] = None,
        finish_guard_turn_extension: int = 0,
        finish_guard_max_extra_turns: int = 0,
    ) -> AgenticReplyResult:
        """Run a multi-turn agentic loop with the VLM using manual tool dispatch.

        Args:
            finish_guard: Optional ``Callable[[int turns_used], str | None]``.
                Called whenever the model *voluntarily* finishes (emits a turn
                with no tool calls), and also when the loop reaches the current
                turn limit after a tool-call turn. If it returns a non-empty
                string, that string is injected as a follow-up user message and
                the loop CONTINUES instead of terminating — a "mandatory final
                gate" that keeps the conversation alive so the model can repair
                its final code. Returning ``None``/empty lets the model finish.
                The guard is expected to cap its own interventions. ``None``
                (default) preserves the original terminate-on-no-tool-calls
                behaviour.
            finish_guard_turn_extension: Number of extra turns to grant after a
                guard complaint fires at a turn-limit boundary.
            finish_guard_max_extra_turns: Total guard-triggered extra turns
                allowed beyond ``max_turns``. ``0`` preserves the old hard-cap
                behavior.
        """
        logger.debug(
            "VLMClient.generate_agentic_reply: backend=%s model=%s images=%d tools=%s max_turns=%d",
            self.backend,
            self.model_name,
            len(image_paths) if image_paths else 0,
            [fn.__name__ for fn in (tool_callables or [])],
            max_turns,
        )
        self._api_retry_count = 0
        self._api_error_count = 0
        if self.backend in ("gemini", "gemini_api"):
            return self._generate_agentic_reply_gemini(
                prompt, image_paths, tool_callables, max_turns, interaction_name,
                finish_guard, finish_guard_turn_extension, finish_guard_max_extra_turns,
            )
        else:
            return self._generate_agentic_reply_vllm(
                prompt, image_paths, tool_callables, max_turns, interaction_name,
                finish_guard, finish_guard_turn_extension, finish_guard_max_extra_turns,
            )

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _generate_reply_gemini(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        use_tools: bool,
        interaction_name: Optional[str],
    ) -> str:
        from google.genai import types

        contents: list = [prompt]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    contents.append(Image.open(img_path))
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=self._SYSTEM_INSTRUCTION,
            tools=[get_api_documentation] if use_tools else [],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        chat = self._gemini_client.chats.create(model=self.model_name, config=config)
        response = self._gemini_send(chat, contents)

        thinking_texts = []
        final_texts = []

        if (
            response.candidates
            and response.candidates[0].content
            and hasattr(response.candidates[0].content, "parts")
            and response.candidates[0].content.parts
        ):
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    if getattr(p, "thought", False):
                        thinking_texts.append(p.text)
                    else:
                        final_texts.append(p.text)

        final_text = "\n".join(final_texts)

        self._log_interaction(
            prompt=prompt,
            final_text=final_text,
            thinking_texts=thinking_texts,
            image_paths=image_paths,
            extra_state={"use_tools": use_tools},
            interaction_name=interaction_name,
        )

        if not final_text and not thinking_texts:
            logger.warning(
                "VLM response contained no text parts (parts: %s); returning empty string.",
                (
                    [type(p).__name__ for p in response.candidates[0].content.parts]
                    if response.candidates
                    and hasattr(response.candidates[0].content, "parts")
                    else "no candidates"
                ),
            )

        return final_text

    @staticmethod
    def _call_finish_guard(finish_guard, turns_used: int) -> Optional[str]:
        """Invoke the finish-guard, swallowing any error (a guard bug must never
        block the model from finishing). Returns the complaint string or None."""
        if finish_guard is None:
            return None
        try:
            complaint = finish_guard(turns_used)
        except Exception as exc:  # never let the guard crash the agentic loop
            logger.warning("finish_guard raised %s: %s", type(exc).__name__, exc)
            return None
        return complaint or None

    def _generate_agentic_reply_gemini(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        tool_callables: Optional[list],
        max_turns: int,
        interaction_name: Optional[str],
        finish_guard: Optional[callable] = None,
        finish_guard_turn_extension: int = 0,
        finish_guard_max_extra_turns: int = 0,
    ) -> AgenticReplyResult:
        from google.genai import types

        tool_callables = tool_callables or []
        tool_map: dict[str, callable] = {fn.__name__: fn for fn in tool_callables}

        contents: list = [prompt]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    contents.append(Image.open(img_path))
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=self._SYSTEM_INSTRUCTION,
            tools=tool_callables if tool_callables else [],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        import time as _time

        chat = self._gemini_client.chats.create(model=self.model_name, config=config)
        _t = _time.perf_counter()
        response = self._gemini_send(chat, contents)
        print(f"[vlm] turn 0 (initial prompt): {_time.perf_counter() - _t:.1f}s")

        all_thinking: list[str] = []
        all_turns: list[str] = []
        tool_call_count = 0
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0

        def _accumulate_usage(r) -> None:
            nonlocal total_input_tokens, total_output_tokens, total_cached_tokens
            if r.usage_metadata:
                total_input_tokens += r.usage_metadata.prompt_token_count or 0
                total_output_tokens += r.usage_metadata.candidates_token_count or 0
                total_cached_tokens += r.usage_metadata.cached_content_token_count or 0

        _accumulate_usage(response)

        turn_limit = int(max_turns)
        max_extra = max(0, int(finish_guard_max_extra_turns or 0))
        per_extension = max(0, int(finish_guard_turn_extension or 0))
        extra_used = 0
        hit_turn_limit = False

        def _grant_guard_extension() -> bool:
            nonlocal turn_limit, extra_used
            if per_extension <= 0 or extra_used >= max_extra:
                return False
            grant = min(per_extension, max_extra - extra_used)
            extra_used += grant
            turn_limit += grant
            return True

        while turns_used < turn_limit:
            turns_used += 1
            function_calls = []
            text_parts = []
            thinking_parts = []

            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls.append(part.function_call)
                    elif hasattr(part, "text") and part.text:
                        if getattr(part, "thought", False):
                            thinking_parts.append(part.text)
                        else:
                            text_parts.append(part.text)

            all_thinking.extend(thinking_parts)

            if not function_calls:
                complaint = self._call_finish_guard(finish_guard, turns_used)
                if complaint:
                    all_turns.append(f"=== TURN {turns_used} (finish-guard) ===")
                    if text_parts:
                        all_turns.append("\n".join(text_parts))
                    all_turns.append(f"[finish_guard] {complaint[:300]}")
                    if turns_used >= turn_limit and not _grant_guard_extension():
                        hit_turn_limit = True
                        break
                    print(
                        f"[vlm] finish-guard fired at turn {turns_used}; "
                        f"injecting feedback and continuing."
                    )
                    response = self._gemini_send(
                        chat, [types.Part.from_text(text=complaint)]
                    )
                    _accumulate_usage(response)
                    continue
                all_turns.append(f"=== TURN {turns_used} ===")
                if text_parts:
                    all_turns.append("\n".join(text_parts))
                break

            all_turns.append(f"=== TURN {turns_used} ===")
            if text_parts:
                all_turns.append("\n".join(text_parts))

            response_parts: list = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                tool_call_count += 1
                logger.info(
                    "Agentic tool call (turn %d): %s(%s)",
                    turns_used,
                    fn_name,
                    ", ".join(f"{k}={v!r}" for k, v in fn_args.items()),
                )
                all_turns.append(f"[tool_call] {fn_name}({fn_args})")

                if fn_name not in tool_map:
                    result = f"[error] Unknown tool: {fn_name}"
                    logger.warning("Unknown tool requested: %s", fn_name)
                else:
                    try:
                        _t_tool = _time.perf_counter()
                        result = tool_map[fn_name](**fn_args)
                        print(
                            f"[vlm] tool {fn_name}: {_time.perf_counter() - _t_tool:.1f}s"
                        )
                    except Exception as exc:
                        result = f"[error] {fn_name} raised {type(exc).__name__}: {exc}"
                        logger.warning("Tool %s raised exception: %s", fn_name, exc)

                all_turns.append(f"[tool_result] {fn_name}: {str(result)[:200]}")

                if isinstance(result, Image.Image):
                    response_parts.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"output": "Frame image attached below."},
                        )
                    )
                    response_parts.append(result)
                else:
                    response_parts.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"output": str(result)},
                        )
                    )

            # Single budget message at the end of the turn (after all tool calls)
            remaining_turns = turn_limit - turns_used
            budget_msg = (
                f"[Turn {turns_used}/{turn_limit} complete — "
                f"{remaining_turns} turn(s) remaining.]"
            )
            if remaining_turns <= 8:
                budget_msg += (
                    " BUDGET CHECKPOINT: stop exploratory probes and keep a complete "
                    "runnable simulator in the sandbox. Use remaining turns only for "
                    "validation-guided repairs; do not replace working code with debug "
                    "exceptions or filesystem inspection."
                )
            all_turns.append(f"[budget] {budget_msg}")
            response_parts.append(types.Part.from_text(text=budget_msg))

            _t = _time.perf_counter()
            response = self._gemini_send(chat, response_parts)
            print(
                f"[vlm] turn {turns_used}/{turn_limit}"
                f" ({len(function_calls)} tool call(s)): {_time.perf_counter() - _t:.1f}s"
            )
            _accumulate_usage(response)

            if turns_used >= turn_limit:
                complaint = self._call_finish_guard(finish_guard, turns_used)
                if complaint and _grant_guard_extension():
                    all_turns.append(f"=== TURN {turns_used} (max-turn gate) ===")
                    all_turns.append(f"[finish_guard] {complaint[:300]}")
                    print(
                        f"[vlm] finish-guard fired at turn limit {turns_used}; "
                        f"extended to {turn_limit} turn(s) and injected feedback."
                    )
                    response = self._gemini_send(
                        chat, [types.Part.from_text(text=complaint)]
                    )
                    _accumulate_usage(response)
                    continue
                hit_turn_limit = True

        if hit_turn_limit:
            logger.warning(
                "Agentic generator hit turn_limit=%d without finishing. Extracting partial text.",
                turn_limit,
            )
            all_turns.append(f"=== TURN {turns_used} (incomplete) ===")
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if (
                        hasattr(part, "text")
                        and part.text
                        and not getattr(part, "thought", False)
                    ):
                        all_turns.append(part.text)

        print(
            f"[tokens] input={total_input_tokens:,}  cached={total_cached_tokens:,}"
            f"  output={total_output_tokens:,}"
            f"  total={total_input_tokens + total_output_tokens:,}"
            f"  ({turns_used} turn(s), {tool_call_count} tool call(s))"
        )

        final_text = "\n".join(
            t
            for t in all_turns
            if not t.startswith("[tool_")
            and not t.startswith("[budget]")
            and not t.startswith("[finish_guard]")
            and not t.startswith("===")
        )
        result = AgenticReplyResult(
            text=final_text,
            tool_call_count=tool_call_count,
            turns_used=turns_used,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=total_cached_tokens,
            api_retry_count=self._api_retry_count,
            api_error_count=self._api_error_count,
        )

        self._log_agentic_interaction(
            prompt=prompt,
            final_text=final_text,
            all_thinking=all_thinking,
            all_turns=all_turns,
            image_paths=image_paths,
            tool_map=tool_map,
            tool_call_count=tool_call_count,
            interaction_name=interaction_name,
        )

        return result

    # ------------------------------------------------------------------
    # VLLM backend (OpenAI-compatible)
    # ------------------------------------------------------------------

    def _generate_reply_vllm(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        interaction_name: Optional[str],
    ) -> str:
        user_content: list = [{"type": "text", "text": prompt}]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        messages = [
            {"role": "system", "content": self._SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]

        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        message = response.choices[0].message
        reasoning_text = _extract_openai_reasoning_text(message)
        final_text = message.content or ""

        self._log_interaction(
            prompt=prompt,
            final_text=final_text,
            thinking_texts=[reasoning_text] if reasoning_text else [],
            image_paths=image_paths,
            extra_state={},
            interaction_name=interaction_name,
        )

        return final_text

    def _generate_agentic_reply_vllm(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        tool_callables: Optional[list],
        max_turns: int,
        interaction_name: Optional[str],
        finish_guard: Optional[callable] = None,
        finish_guard_turn_extension: int = 0,
        finish_guard_max_extra_turns: int = 0,
    ) -> AgenticReplyResult:
        import time as _time

        tool_callables = tool_callables or []
        tool_map: dict[str, callable] = {fn.__name__: fn for fn in tool_callables}
        openai_tools = (
            [_callable_to_openai_tool(fn) for fn in tool_callables]
            if tool_callables
            else None
        )

        # Build the initial user message
        user_content: list = [{"type": "text", "text": prompt}]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        messages: list[dict] = [
            {"role": "system", "content": self._SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]

        all_thinking: list[str] = []
        all_turns: list[str] = []
        tool_call_count = 0
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0

        _t = _time.perf_counter()
        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=openai_tools,
            temperature=self.temperature,
        )
        print(f"[vlm] turn 0 (initial prompt): {_time.perf_counter() - _t:.1f}s")

        if response.usage:
            total_input_tokens += response.usage.prompt_tokens or 0
            total_output_tokens += response.usage.completion_tokens or 0

        turn_limit = int(max_turns)
        max_extra = max(0, int(finish_guard_max_extra_turns or 0))
        per_extension = max(0, int(finish_guard_turn_extension or 0))
        extra_used = 0
        hit_turn_limit = False

        def _grant_guard_extension() -> bool:
            nonlocal turn_limit, extra_used
            if per_extension <= 0 or extra_used >= max_extra:
                return False
            grant = min(per_extension, max_extra - extra_used)
            extra_used += grant
            turn_limit += grant
            return True

        while turns_used < turn_limit:
            turns_used += 1
            msg = response.choices[0].message
            reasoning_text = _extract_openai_reasoning_text(msg)
            if reasoning_text:
                all_thinking.append(reasoning_text)

            # Append assistant message to history
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )

            if not msg.tool_calls:
                complaint = self._call_finish_guard(finish_guard, turns_used)
                if complaint:
                    all_turns.append(f"=== TURN {turns_used} (finish-guard) ===")
                    all_turns.append(msg.content or "")
                    all_turns.append(f"[finish_guard] {complaint[:300]}")
                    if turns_used >= turn_limit and not _grant_guard_extension():
                        hit_turn_limit = True
                        break
                    print(
                        f"[vlm] finish-guard fired at turn {turns_used}; "
                        f"injecting feedback and continuing."
                    )
                    messages.append({"role": "user", "content": complaint})
                    response = self._openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=openai_tools,
                        temperature=self.temperature,
                    )
                    if response.usage:
                        total_input_tokens += response.usage.prompt_tokens or 0
                        total_output_tokens += response.usage.completion_tokens or 0
                    continue
                all_turns.append(f"=== TURN {turns_used} ===")
                all_turns.append(msg.content or "")
                break

            all_turns.append(f"=== TURN {turns_used} ===")
            if msg.content:
                all_turns.append(msg.content)

            # Dispatch each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Coerce args to match parameter annotations (e.g. "0" → 0 for int
                # params) so the model doesn't need to retry on type mismatches.
                # get_type_hints() is used instead of param.annotation because
                # `from __future__ import annotations` makes annotations lazy strings.
                if fn_name in tool_map:
                    import typing as _typing

                    _PRIMITIVE = (int, float, bool, str)
                    try:
                        hints = _typing.get_type_hints(tool_map[fn_name])
                    except Exception:
                        hints = {}
                    for pname, ann in hints.items():
                        if pname in fn_args and ann in _PRIMITIVE:
                            try:
                                fn_args[pname] = ann(fn_args[pname])
                            except (ValueError, TypeError):
                                pass

                tool_call_count += 1
                logger.info(
                    "Agentic tool call (turn %d): %s(%s)",
                    turns_used,
                    fn_name,
                    ", ".join(f"{k}={v!r}" for k, v in fn_args.items()),
                )
                all_turns.append(f"[tool_call] {fn_name}({fn_args})")

                if fn_name not in tool_map:
                    tool_result = f"[error] Unknown tool: {fn_name}"
                    logger.warning("Unknown tool requested: %s", fn_name)
                else:
                    try:
                        _t_tool = _time.perf_counter()
                        tool_result = tool_map[fn_name](**fn_args)
                        print(
                            f"[vlm] tool {fn_name}: {_time.perf_counter() - _t_tool:.1f}s"
                        )
                    except Exception as exc:
                        tool_result = (
                            f"[error] {fn_name} raised {type(exc).__name__}: {exc}"
                        )
                        logger.warning("Tool %s raised exception: %s", fn_name, exc)

                all_turns.append(f"[tool_result] {fn_name}: {str(tool_result)[:200]}")

                if isinstance(tool_result, Image.Image):
                    # VLLM's Gemma 4 chat template does not support images
                    # inside tool-role messages.  Send a text acknowledgement
                    # in the tool message, then inject the image as a user
                    # message immediately after — user messages do support
                    # multimodal content.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Frame rendered. See image below.",
                        }
                    )
                    b64 = _pil_to_b64(tool_result)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Rendered frame:"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}"
                                    },
                                },
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result),
                        }
                    )

            # Single budget message at the end of the turn (after all tool calls)
            remaining_turns = turn_limit - turns_used
            budget_msg = (
                f"[Turn {turns_used}/{turn_limit} complete — "
                f"{remaining_turns} turn(s) remaining.]"
            )
            all_turns.append(f"[budget] {budget_msg}")
            messages.append({"role": "user", "content": budget_msg})

            _t = _time.perf_counter()
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=openai_tools,
                temperature=self.temperature,
            )
            print(
                f"[vlm] turn {turns_used}/{turn_limit}"
                f" ({len(msg.tool_calls)} tool call(s)): {_time.perf_counter() - _t:.1f}s"
            )

            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

            if turns_used >= turn_limit:
                complaint = self._call_finish_guard(finish_guard, turns_used)
                if complaint and _grant_guard_extension():
                    all_turns.append(f"=== TURN {turns_used} (max-turn gate) ===")
                    all_turns.append(f"[finish_guard] {complaint[:300]}")
                    print(
                        f"[vlm] finish-guard fired at turn limit {turns_used}; "
                        f"extended to {turn_limit} turn(s) and injected feedback."
                    )
                    messages.append({"role": "user", "content": complaint})
                    response = self._openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=openai_tools,
                        temperature=self.temperature,
                    )
                    if response.usage:
                        total_input_tokens += response.usage.prompt_tokens or 0
                        total_output_tokens += response.usage.completion_tokens or 0
                    continue
                hit_turn_limit = True

        if hit_turn_limit:
            logger.warning(
                "Agentic generator hit turn_limit=%d without finishing. Extracting partial text.",
                turn_limit,
            )
            all_turns.append(f"=== TURN {turns_used} (incomplete) ===")
            final_msg = response.choices[0].message
            if final_msg.content:
                all_turns.append(final_msg.content)

        print(
            f"[tokens] input={total_input_tokens:,}  output={total_output_tokens:,}"
            f"  total={total_input_tokens + total_output_tokens:,}"
            f"  ({turns_used} turn(s), {tool_call_count} tool call(s))"
        )

        final_text = "\n".join(
            t
            for t in all_turns
            if not t.startswith("[tool_")
            and not t.startswith("[budget]")
            and not t.startswith("[finish_guard]")
            and not t.startswith("===")
        )
        result = AgenticReplyResult(
            text=final_text,
            tool_call_count=tool_call_count,
            turns_used=turns_used,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=0,
            api_retry_count=self._api_retry_count,
            api_error_count=self._api_error_count,
        )

        self._log_agentic_interaction(
            prompt=prompt,
            final_text=final_text,
            all_thinking=all_thinking,
            all_turns=all_turns,
            image_paths=image_paths,
            tool_map=tool_map,
            tool_call_count=tool_call_count,
            interaction_name=interaction_name,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gemini_send(self, chat, msg):
        """Wrap chat.send_message with retries on 429 quota and transient 5xx errors."""
        import time as _time

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                return chat.send_message(msg)
            except Exception as exc:
                self._api_error_count += 1
                is_429 = False
                is_transient = False
                try:
                    from google.genai.errors import ClientError, ServerError

                    if isinstance(exc, ClientError):
                        is_429 = getattr(exc, "code", None) == 429 or "429" in str(exc)
                    if isinstance(exc, ServerError):
                        is_transient = True
                except ImportError:
                    pass
                if not is_transient:
                    text = str(exc)
                    is_transient = any(
                        marker in text
                        for marker in (
                            "503",
                            "502",
                            "500",
                            "504",
                            "UNAVAILABLE",
                            "DeadlineExceeded",
                            "Deadline Exceeded",
                            "Connection reset",
                            "Connection aborted",
                        )
                    )
                last_attempt = attempt >= max_attempts - 1
                if is_429 and not last_attempt:
                    self._api_retry_count += 1
                    print(
                        f"[vlm] Google API quota exhausted (429 RESOURCE_EXHAUSTED) "
                        f"— attempt {attempt + 1}/{max_attempts}. Waiting 2 minutes before retrying..."
                    )
                    _time.sleep(120)
                    continue
                if is_transient and not last_attempt:
                    self._api_retry_count += 1
                    backoff = min(15.0 * (2.0 ** attempt), 180.0)
                    print(
                        f"[vlm] Transient Google API error ({type(exc).__name__}: {str(exc)[:120]}) "
                        f"— attempt {attempt + 1}/{max_attempts}. Retrying in {backoff:.0f}s..."
                    )
                    _time.sleep(backoff)
                    continue
                raise

    # ------------------------------------------------------------------
    # Shared logging helpers
    # ------------------------------------------------------------------

    def _resolve_out_dir(
        self, interaction_name: Optional[str]
    ) -> Optional[pathlib.Path]:
        if self._stage_dir:
            out_dir = self._stage_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir
        if self.llm_interactions_dir:
            name = interaction_name or "interaction"
            out_dir = self.llm_interactions_dir / f"{self._interaction_count}_{name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._interaction_count += 1
            return out_dir
        return None

    def _log_interaction(
        self,
        prompt: str,
        final_text: str,
        thinking_texts: list[str],
        image_paths: Optional[list[str]],
        extra_state: dict,
        interaction_name: Optional[str],
    ) -> None:
        out_dir = self._resolve_out_dir(interaction_name)
        if out_dir is None:
            return

        with open(out_dir / "prompt.md", "w") as f:
            f.write(prompt)

        state = {
            "model_name": self.model_name,
            "backend": self.backend,
            "temperature": self.temperature,
            "interaction_name": interaction_name,
            "image_paths": [
                os.path.abspath(p) for p in (image_paths or []) if os.path.exists(p)
            ],
            **extra_state,
        }
        with open(out_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        if thinking_texts:
            with open(out_dir / "thinking.md", "w") as f:
                f.write("\n\n".join(thinking_texts))

        with open(out_dir / "response.md", "w") as f:
            f.write(final_text)

    def _log_agentic_interaction(
        self,
        prompt: str,
        final_text: str,
        all_thinking: list[str],
        all_turns: list[str],
        image_paths: Optional[list[str]],
        tool_map: dict,
        tool_call_count: int,
        interaction_name: Optional[str],
    ) -> None:
        out_dir = self._resolve_out_dir(interaction_name)
        if out_dir is None:
            return

        with open(out_dir / "prompt.md", "w") as f:
            f.write(prompt)

        state = {
            "model_name": self.model_name,
            "backend": self.backend,
            "temperature": self.temperature,
            "interaction_name": interaction_name,
            "image_paths": [
                os.path.abspath(p) for p in (image_paths or []) if os.path.exists(p)
            ],
            "tools": list(tool_map.keys()),
            "tool_call_count": tool_call_count,
        }
        with open(out_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        if all_thinking:
            with open(out_dir / "thinking.md", "w") as f:
                f.write("\n\n".join(all_thinking))

        with open(out_dir / "turns.md", "w") as f:
            f.write("\n\n".join(all_turns))

        with open(out_dir / "response.md", "w") as f:
            f.write(final_text)
