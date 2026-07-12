"""Shared text-processing helpers used by retrieval, eval, and quiz scripts."""

from __future__ import annotations

import re
from typing import Iterable, List


def clean_snippet(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return " ".join(stripped.split())


def format_metadata(metadata: dict | None) -> str:
    if not metadata:
        return "metadata: none"
    parts = [f"{key}={value}" for key, value in metadata.items()]
    return "metadata: " + ", ".join(parts)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def concat_context(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def compute_overlap_ratio(answer: str, context_parts: Iterable[str]) -> float:
    """Return the fraction of answer tokens that also appear in the contexts."""
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(concat_context(context_parts)))
    if not context_tokens:
        return 0.0
    matches = sum(1 for token in answer_tokens if token in context_tokens)
    return matches / max(1, len(answer_tokens))
