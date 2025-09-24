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

import csv
import json
from pathlib import Path
from typing import Iterable, List, Mapping
import argparse
import re


def load_eval_data(filepath: Path) -> List[Mapping[str, str]]:
    """Load evaluation data from JSON (list of dicts) or CSV (headers: question,answer,context).

    Returns a list of mappings with keys: 'question', 'answer', 'context' (context may be a
    single string or a JSON-encoded list of snippets).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Eval file not found: {filepath}")
    if filepath.suffix.lower() == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data)
    else:
        out = []
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({"question": row.get("question"), "answer": row.get("answer"), "context": row.get("context")})
        return out


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def is_faithful(answer: str, context: Iterable[str]) -> bool:
    """Return True when the answer shows sufficient lexical overlap with context.

    Heuristic: at least 30% of answer tokens must appear in the concatenated context.
    """
    if not answer:
        return False
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return False
    ctx_text = " ".join(context) if isinstance(context, (list, tuple)) else (context or "")
    ctx_tokens = set(_tokenize(ctx_text))
    if not ctx_tokens:
        return False
    match = sum(1 for t in answer_tokens if t in ctx_tokens)
    ratio = match / max(1, len(answer_tokens))
    return ratio >= 0.3


def should_abstain(context: Iterable[str], min_length: int = 30, min_ratio: float = 0.05) -> bool:
    """Decide to abstain when context is too short or mostly irrelevant.

    - Abstain if combined context length (chars) < min_length.
    - Abstain if lexical overlap ratio (as defined below) is below min_ratio.
    """
    ctx_text = " ".join(context) if isinstance(context, (list, tuple)) else (context or "")
    if len(ctx_text.strip()) < min_length:
        return True
    # weak overlap check: if less than min_ratio of context tokens are shared with themselves
    ctx_tokens = _tokenize(ctx_text)
    if not ctx_tokens:
        return True
    # simple heuristic: if average token length is tiny, abstain
    avg_token_len = sum(len(t) for t in ctx_tokens) / len(ctx_tokens)
    if avg_token_len < 2:
        return True
    return False


def evaluate_qa_pair(qa: Mapping[str, object]) -> Mapping[str, object]:
    result = {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "context": qa.get("context"),
    }
    ctx = result["context"]
    # if context is JSON list encoded as string, try to parse it
    if isinstance(ctx, str):
        try:
            parsed = json.loads(ctx)
            if isinstance(parsed, list):
                ctx = parsed
        except Exception:
            ctx = [ctx]
    result["faithful"] = is_faithful(result["answer"] or "", ctx or [])
    result["abstain"] = should_abstain(ctx or [])
    return result


def save_eval_report(results: List[Mapping[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_eval_summary(results: List[Mapping[str, object]]) -> None:
    total = len(results)
    faithful = sum(1 for r in results if r.get("faithful"))
    abstain = sum(1 for r in results if r.get("abstain"))
    print(f"Evaluated {total} examples")
    print(f" - Faithful: {faithful} ({faithful/total:.1%})")
    print(f" - Abstain: {abstain} ({abstain/total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA pairs for faithfulness and abstention")
    parser.add_argument("--in", dest="infile", required=True, help="JSON or CSV eval file")
    parser.add_argument("--out", dest="outfile", default="data/eval_report.json", help="Output JSON report path")
    args = parser.parse_args()

    infile = Path(args.infile)
    data = load_eval_data(infile)
    results = [evaluate_qa_pair(q) for q in data]
    save_eval_report(results, Path(args.outfile))
    print_eval_summary(results)


if __name__ == "__main__":
    main()

