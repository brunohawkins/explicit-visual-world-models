"""
Unified SEARCH/REPLACE patcher for LLM-generated diffs.

Expected diff syntax in LLM output::

    <<<<
    [exact code to replace]
    ====
    [new code to insert]
    >>>>

Matching strategy
-----------------
1. **Exact match** — the search block is looked up verbatim in the source.
2. **Indentation-adjusted match** — if the exact match fails, the patcher
   infers that the LLM produced a correctly structured but uniformly
   mis-indented block.  It finds the first non-empty line of the search block
   in the source, computes the indentation delta, shifts every line of both
   the search and replace blocks by that delta, and retries.  This handles the
   common LLM failure mode where all lines are indented by the wrong constant
   amount while their relative indentation is correct.
"""
from __future__ import annotations

import re
from typing import List, Tuple


class PatchError(Exception):
    """Raised when a patch cannot be applied cleanly."""


# ---------------------------------------------------------------------------
# Indentation helpers
# ---------------------------------------------------------------------------

def _leading_spaces(line: str) -> int:
    """Return the number of leading space characters in *line*."""
    return len(line) - len(line.lstrip(" "))


def _adjust_indentation(text: str, delta: int) -> str:
    """Shift every line of *text* by *delta* spaces.

    Positive *delta* adds spaces; negative *delta* removes them (clamped at
    zero so lines cannot have negative indentation).  Blank lines are passed
    through unchanged to avoid introducing spurious trailing whitespace.
    """
    result: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" ")
        if not stripped.strip():
            # Blank / whitespace-only line — preserve as-is.
            result.append(line)
        else:
            current = len(line) - len(stripped)
            new_indent = max(0, current + delta)
            result.append(" " * new_indent + stripped)
    return "".join(result)


# ---------------------------------------------------------------------------
# Core patcher
# ---------------------------------------------------------------------------

class UnifiedPatcher:
    """Applies LLM-generated SEARCH/REPLACE blocks to Python source code."""

    BLOCK_REGEX = re.compile(
        r"<<<<\n(.*?)\n====\n(.*?)\n>>>>",
        re.DOTALL | re.MULTILINE,
    )

    @classmethod
    def extract_blocks(cls, llm_response: str) -> List[Tuple[str, str]]:
        """Parse *llm_response* and return all ``(search, replace)`` pairs."""
        matches = cls.BLOCK_REGEX.findall(llm_response)
        return [(s.strip(), r.strip()) for s, r in matches]

    @classmethod
    def apply_patch(cls, source_code: str, search_text: str, replace_text: str) -> str:
        """Locate *search_text* in *source_code* and replace it with *replace_text*.

        Matching is attempted in two passes:

        1. **Exact** — the search block is looked up verbatim.
        2. **Indentation-adjusted** — if the exact match fails, the patcher
           identifies the first non-empty line of the search block in the
           source, computes the indentation delta between the two, shifts all
           lines of the search and replace blocks by that delta, and retries.

        Raises:
            PatchError: if neither pass finds the block.
        """
        search = search_text.strip("\n")
        replace = replace_text.strip("\n")

        # Pass 1: exact match
        if search in source_code:
            return source_code.replace(search, replace, 1)

        # Pass 2: indentation-adjusted match
        adjusted_source, adjusted_replace = cls._indentation_adjusted_patch(
            source_code, search, replace
        )
        if adjusted_source is not None:
            return source_code.replace(adjusted_source, adjusted_replace, 1)

        raise PatchError(
            f"Search block not found in source (exact and indentation-adjusted):\n{search}"
        )

    @classmethod
    def _indentation_adjusted_patch(
        cls, source_code: str, search: str, replace: str
    ) -> tuple[str, str] | tuple[None, None]:
        """Try to match *search* in *source_code* after correcting indentation.

        Returns the adjusted ``(search_in_source, replace_in_source)`` strings
        on success, or ``(None, None)`` if no consistent adjustment works.
        """
        search_lines = search.splitlines()
        first_nonempty_search = next(
            (l for l in search_lines if l.strip()), None
        )
        if first_nonempty_search is None:
            return None, None

        search_indent = _leading_spaces(first_nonempty_search)
        first_stripped = first_nonempty_search.strip()

        # Collect all indentation levels at which the first line appears in source.
        seen_deltas: set[int] = set()
        for source_line in source_code.splitlines():
            if source_line.strip() != first_stripped:
                continue
            source_indent = _leading_spaces(source_line)
            delta = source_indent - search_indent
            if delta == 0 or delta in seen_deltas:
                continue  # delta=0 was already tried in exact pass
            seen_deltas.add(delta)

            adjusted_search = _adjust_indentation(search, delta)
            if adjusted_search in source_code:
                adjusted_replace = _adjust_indentation(replace, delta)
                return adjusted_search, adjusted_replace

        return None, None

    @classmethod
    def apply_all_patches(cls, source_code: str, llm_response: str) -> str:
        """Parse *llm_response* and apply all extracted patches in order."""
        blocks = cls.extract_blocks(llm_response)
        patched = source_code
        for search_text, replace_text in blocks:
            patched = cls.apply_patch(patched, search_text, replace_text)
        return patched
