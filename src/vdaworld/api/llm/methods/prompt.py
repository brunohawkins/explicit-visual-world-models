from typing import List
from vdaworld.utils.text_handling import split_markdown_by_headings
import os

PARTS = [
    # Current action-conditioned prompt parts.
    "definitions",
    "deliverable",
    "environment",
    "codegen_inputs",
    "test_time",
    "code_rules",
    "tools",
    # Legacy/generic prompt parts.
    "toolbox",
    "task",
]
SPECS = ["simulator"]


def get_prompt(path, name, **kwargs):
    prompt_path = os.path.join(path, f"prompt_{name}.md")
    prompt: List[str] = split_markdown_by_headings(prompt_path)

    prompt_ = []
    for prompt_part in prompt:
        for part_name in PARTS:
            part_tag = f"[PART_{part_name.upper()}]"
            if part_tag in prompt_part:
                prompt_part = prompt_part.replace(part_tag, _get_part(path, part_name))
        prompt_.append(prompt_part)
    prompt = prompt_

    prompt_ = []
    for prompt_part in prompt:
        for k, v in kwargs.items():
            prompt_part = prompt_part.replace(f"[{k.upper()}]", str(v))
        if "[TURN_BUDGET_SECTION]" in prompt_part:
            prompt_part = prompt_part.replace(
                "[TURN_BUDGET_SECTION]",
                _turn_budget_section(
                    kwargs.get("max_turns"),
                    kwargs.get("finish_guard_max_extra_turns", 0),
                ),
            )
        prompt_.append(prompt_part)
    prompt = prompt_
    return "\n\n".join(prompt)


def _get_part(path, name):
    part_path = os.path.join(path, f"part_{name}.md")
    if not os.path.exists(part_path):
        return f"[PART_{name.upper()}]"
    part: str = split_markdown_by_headings(part_path)[0]

    for spec_name in SPECS:
        spec_tag = f"[SPEC_{spec_name.upper()}]"
        if spec_tag in part:
            part = part.replace(spec_tag, _get_spec(path, spec_name))
        
    return part


def _get_spec(path, name):
    spec_path = os.path.join(path, f"spec_{name}.md")
    spec: str = split_markdown_by_headings(spec_path)[0]
    return spec


def _turn_budget_section(max_turns, finish_guard_max_extra_turns=0):
    if max_turns is None:
        return ""
    max_turns = int(max_turns)
    extra = max(0, int(finish_guard_max_extra_turns or 0))
    if extra:
        budget = (
            f"You have a base budget of **{max_turns} turns**. If you try to finish "
            f"but the final gate reports repairable failures, the harness may grant "
            f"up to **{extra} additional repair turns**."
        )
    else:
        budget = f"You have at most **{max_turns} turns**."
    return (
        "## Turn budget\n\n"
        f"{budget} Use early turns for data inspection, then make targeted code edits "
        "and validate after each change. Do not spend the last turns rewriting from "
        "scratch unless the current simulator is clearly broken."
    )


if __name__ == "__main__":
    prompt = get_prompt("prompts/generic", "generation", caption="test", simulator_class_name="VideoSimulation", available_tools="- `segment`\n- `pts3d`")
    print(prompt)
