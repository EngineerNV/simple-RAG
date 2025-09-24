"""03_eval.py — judge groundedness and abstention for the agent's answers.

Teacher briefing
-----------------
This milestone closes the loop on your RAG agent. Use simple heuristics (or LLM-based
verification if you prefer) to measure whether responses align with retrieved context
and when the system should decline to answer.

Implementation checklist
------------------------
1. Load evaluation data produced from milestone 2 (question, answer, context, optional metadata).
2. Implement ``is_faithful`` to flag hallucinated answers—consider keyword overlap,
   cited source markers, or a secondary LLM judge.
3. Implement ``should_abstain`` to catch weak retrieval (short context, low scores, etc.).
4. Summarize the evaluation run with aggregate metrics and optionally persist per-example
   reports for reflection.

Stretch goals
-------------
- Track precision/recall style metrics if you have human labels.
- Persist JSON/CSV reports for inclusion in your homework submission.
- Surface a few representative failure cases for discussion.
"""

import csv  # Handle CSV evaluation files for small curated sets
import json  # Handle JSON evaluation exports from the agent pipeline
from pathlib import Path  # Manage input/output paths for evaluation artifacts
from typing import Iterable, List, Mapping  # Provide type hints for QA examples and results


def load_eval_data(filepath: Path) -> List[Mapping[str, str]]:
    """Load evaluation data (questions, answers, contexts) from JSON or CSV."""
    # TODO: Inspect filepath suffix and parse accordingly.
    raise NotImplementedError


def is_faithful(answer: str, context: Iterable[str]) -> bool:
    """Return True when the answer is grounded in the provided context snippets."""
    # TODO: Implement lexical overlap, citation checks, or an LLM-based verifier.
    raise NotImplementedError


def should_abstain(context: Iterable[str], min_length: int = 30, min_score: float | None = None) -> bool:
    """Return True when the context is insufficient or low quality for answering."""
    # TODO: Evaluate chunk length, retrieval scores, or other metadata to decide.
    raise NotImplementedError


def evaluate_qa_pair(qa: Mapping[str, object]) -> Mapping[str, object]:
    """Evaluate a single QA triple and return a structured result dict."""
    result = {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "context": qa.get("context"),
    }
    result["faithful"] = is_faithful(result["answer"], result["context"])
    result["abstain"] = should_abstain(result["context"])
    return result


def save_eval_report(results: List[Mapping[str, object]], out_path: Path) -> None:
    """Persist evaluation results to JSON or CSV for reflection/documentation."""
    # TODO: Support at least one serialization format that matches your coursework needs.
    raise NotImplementedError


def print_eval_summary(results: List[Mapping[str, object]]) -> None:
    """Print aggregate metrics (counts/percentages) for quick feedback."""
    # TODO: Derive pass/fail counts and abstention rates; print them clearly.
    raise NotImplementedError


def main() -> None:
    """CLI entry point: load evaluation data, score it, and report metrics."""
    # TODO: Wire argument parsing, call helpers above, and emit the summary/report.
    raise NotImplementedError


if __name__ == "__main__":
    main()

