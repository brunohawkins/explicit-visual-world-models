"""Filesystem text utilities."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def split_markdown_by_headings(file_path: str) -> list[str]:
    """Read a markdown file and return its content as a single-element list.

    The list wrapper keeps callers compatible with a future version that
    splits on top-level headings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]


def load_text(file_path: str) -> str:
    """Read and return the entire contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
