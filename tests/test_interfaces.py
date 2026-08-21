from vdaworld.core.interfaces import GenerationResult


def test_generation_result_defaults():
    gr = GenerationResult(code="print('hi')", stage_dir="/tmp/x")
    assert gr.code == "print('hi')"
    assert gr.stage_dir == "/tmp/x"
    assert gr.tool_call_count == 0
    assert gr.input_tokens == 0
    assert gr.output_tokens == 0
    assert gr.cached_tokens == 0
    assert gr.api_retry_count == 0
    assert gr.api_error_count == 0
    assert gr.keep_best_enabled is False
    assert gr.shipped_checkpoint_tool_call is None
    assert gr.shipped_worst_ratio is None
    assert gr.final_sandbox_worst_ratio is None
    assert gr.keep_best_restored is False


def test_generation_result_fields():
    gr = GenerationResult(
        code="class X: pass",
        stage_dir="/tmp/y",
        tool_call_count=5,
        input_tokens=1234,
        output_tokens=567,
        cached_tokens=89,
        api_retry_count=2,
        api_error_count=3,
        keep_best_enabled=True,
        shipped_checkpoint_tool_call=3,
        shipped_worst_ratio=0.4,
        final_sandbox_worst_ratio=1.2,
        keep_best_restored=True,
    )
    assert gr.tool_call_count == 5
    assert gr.input_tokens == 1234
    assert gr.output_tokens == 567
    assert gr.cached_tokens == 89
    assert gr.api_retry_count == 2
    assert gr.api_error_count == 3
    assert gr.keep_best_enabled is True
    assert gr.shipped_checkpoint_tool_call == 3
    assert gr.shipped_worst_ratio == 0.4
    assert gr.final_sandbox_worst_ratio == 1.2
    assert gr.keep_best_restored is True
