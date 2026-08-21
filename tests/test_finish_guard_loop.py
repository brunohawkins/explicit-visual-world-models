"""Unit tests for the finish-guard continuation logic in VLMClient.

The agentic loop must, on a *voluntary* finish (a turn with no tool calls),
consult the finish-guard: if the guard returns a complaint, the loop injects it
as a follow-up message and CONTINUES; if the guard returns None, the loop
terminates. This is exercised against the vllm (OpenAI-compatible) backend with
a fully mocked client so no network or model is required. The gemini backend
mirrors the same control flow.

It must also consult the same guard at the max-turn boundary after a tool-call
turn, grant bounded extra headroom, inject the complaint, and let the model take
at least one corrective turn after the complaint.
"""

from __future__ import annotations

from types import SimpleNamespace

from vdaworld.api.vlm import VLMClient


class _FakeCompletions:
    """Returns a pre-scripted sequence of responses, one per create() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.kwargs = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        return self._responses.pop(0)


def _no_tool_response():
    """A response that voluntarily finishes (no tool calls)."""
    msg = SimpleNamespace(content="I am done.", tool_calls=None, reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _tool_call_response(name="ping"):
    """A response containing one tool call."""
    fn = SimpleNamespace(name=name, arguments="{}")
    tc = SimpleNamespace(id=f"call_{name}", function=fn)
    msg = SimpleNamespace(content=None, tool_calls=[tc], reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _make_client(responses):
    client = VLMClient(model_name="fake", backend="vllm", base_url="http://localhost:1")
    fake = _FakeCompletions(responses)
    client._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_guard_complaint_continues_then_terminates():
    # 3 responses: initial (finish) + after-complaint (finish) + final (finish).
    client, fake = _make_client(
        [_no_tool_response(), _no_tool_response(), _no_tool_response()]
    )
    guard_calls = []

    def guard(turns_used):
        guard_calls.append(turns_used)
        # Complain once, then allow the finish.
        return "FIX YOUR CODE" if len(guard_calls) == 1 else None

    result = client.generate_agentic_reply(
        prompt="go",
        tool_callables=[],
        max_turns=5,
        finish_guard=guard,
    )
    # Guard consulted twice: blocked the first finish, allowed the second.
    assert guard_calls == [1, 2], guard_calls
    # create() called: initial + one continuation triggered by the complaint.
    assert fake.calls == 2, fake.calls
    assert "FIX YOUR CODE" not in result.text  # complaint is feedback, not output


def test_no_guard_terminates_immediately():
    client, fake = _make_client([_no_tool_response()])
    result = client.generate_agentic_reply(
        prompt="go", tool_callables=[], max_turns=5, finish_guard=None
    )
    assert fake.calls == 1  # initial only; no continuation
    assert isinstance(result.text, str)


def test_guard_exception_does_not_block_finish():
    client, fake = _make_client([_no_tool_response()])

    def bad_guard(turns_used):
        raise ValueError("guard is buggy")

    # A buggy guard must be swallowed and let the model finish (not crash).
    result = client.generate_agentic_reply(
        prompt="go", tool_callables=[], max_turns=5, finish_guard=bad_guard
    )
    assert fake.calls == 1
    assert isinstance(result.text, str)


def test_guard_at_max_turn_extends_and_allows_corrective_tool_turn():
    # Response order:
    # 1) initial model turn calls a tool and consumes the soft cap,
    # 2) model reply to the budget message is discarded when the max-turn gate fires,
    # 3) after the gate complaint, model takes a corrective tool-call turn,
    # 4) model voluntarily finishes and the guard allows it.
    client, fake = _make_client(
        [
            _tool_call_response(),
            _no_tool_response(),
            _tool_call_response(),
            _no_tool_response(),
        ]
    )
    guard_calls = []

    def ping():
        return "pong"

    def guard(turns_used):
        guard_calls.append(turns_used)
        return "FIX YOUR CODE" if len(guard_calls) == 1 else None

    result = client.generate_agentic_reply(
        prompt="go",
        tool_callables=[ping],
        max_turns=1,
        finish_guard=guard,
        finish_guard_turn_extension=2,
        finish_guard_max_extra_turns=2,
    )

    assert guard_calls == [1, 3], guard_calls
    assert result.tool_call_count == 2
    assert result.turns_used == 3
    assert fake.calls == 4

    complaint_messages = [
        msg
        for kwargs in fake.kwargs
        for msg in kwargs["messages"]
        if msg.get("role") == "user" and msg.get("content") == "FIX YOUR CODE"
    ]
    assert complaint_messages, fake.kwargs
