"""03_quiz.py — interactive human metrics checklist for retrieval answers.

This script complements the automatic heuristics in ``03_eval.py`` by letting a
reviewer inspect the retrieved evidence, record qualitative tags, and decide
whether an answer is faithful or should have abstained. Each reviewed example is
persisted to JSONL/CSV so downstream tooling (``04_report.py``) can summarise
the findings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from langchain_community.embeddings import HuggingFaceEmbeddings


# -- Lightweight lexical overlap helper -----------------------------------------------------


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _concat_context(parts: Iterable[str]) -> str:
    return " ".join(segment for segment in parts if segment)


def compute_overlap_ratio(answer: str, context_parts: Iterable[str]) -> float:
    """Return lexical overlap ratio between the answer and joined contexts."""

    if not answer:
        return 0.0
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(_tokenize(_concat_context(context_parts)))
    if not context_tokens:
        return 0.0
    matches = sum(1 for token in answer_tokens if token in context_tokens)
    return matches / max(1, len(answer_tokens))


# -- Question loading and persistence -------------------------------------------------------


def load_questions(path: Path) -> List[Mapping[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Questions file must contain a list of objects.")
    questions: List[Mapping[str, str]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("Each question entry must be an object with 'id' and 'question'.")
        if "id" not in item or "question" not in item:
            raise ValueError("Each question entry must define 'id' and 'question' keys.")
        questions.append({"id": str(item["id"]), "question": str(item["question"])})
    return questions


def load_existing_reviews(path: Path) -> List[MutableMapping[str, object]]:
    if not path.exists():
        return []
    reviews: List[MutableMapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return reviews


def save_reviews_jsonl(path: Path, reviews: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in reviews:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


CSV_HEADERS = [
    "id",
    "question",
    "answer",
    "agent_mode",
    "k",
    "max_score",
    "mean_score",
    "overlap_ratio",
    "faithful",
    "should_abstain",
    "tags",
    "notes",
    "timestamp",
]


def save_reviews_csv(path: Path, reviews: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for entry in reviews:
            row = {key: entry.get(key) for key in CSV_HEADERS}
            tags = entry.get("tags")
            if isinstance(tags, list):
                row["tags"] = ",".join(tags)
            writer.writerow(row)


# -- Retrieval plumbing ---------------------------------------------------------------------


def load_query_module():
    return import_module("scripts.02_query")


@dataclass
class RetrievedContext:
    rank: int
    score: float
    metadata: Mapping[str, object]
    snippet: str


@dataclass
class ReviewRecord:
    id: str
    question: str
    answer: str
    agent_mode: str
    k: int
    contexts: List[RetrievedContext]
    max_score: float
    mean_score: float
    overlap_ratio: float
    faithful: bool | None = None
    should_abstain: bool | None = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "agent_mode": self.agent_mode,
            "k": self.k,
            "contexts": [
                {
                    "rank": ctx.rank,
                    "score": ctx.score,
                    "metadata": ctx.metadata,
                    "snippet": ctx.snippet,
                }
                for ctx in self.contexts
            ],
            "max_score": self.max_score,
            "mean_score": self.mean_score,
            "overlap_ratio": self.overlap_ratio,
            "faithful": self.faithful,
            "should_abstain": self.should_abstain,
            "tags": list(self.tags),
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


def describe_contexts(contexts: Sequence[RetrievedContext], width: int) -> str:
    lines: List[str] = []
    if not contexts:
        return "No contexts retrieved."
    for ctx in contexts:
        meta_items = []
        for key, value in (ctx.metadata or {}).items():
            meta_items.append(f"{key}={value}")
        meta_line = "metadata: " + ", ".join(meta_items) if meta_items else "metadata: none"
        lines.append(f"[{ctx.rank}] score: {ctx.score:.3f} | {meta_line}")
        snippet = ctx.snippet or "(empty snippet)"
        lines.append(textwrap.fill(snippet, width=width))
        lines.append("-")
    if lines and lines[-1] == "-":
        lines.pop()
    return "\n".join(lines)


def generate_answer(
    query_module,
    agent_mode: str,
    question: str,
    results,
    k: int,
    provider: str,
    llm_model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
) -> str:
    if agent_mode == "none":
        return query_module.synthesize_from_results(results, k)
    if agent_mode == "pretend":
        cited = [str(idx) for idx in range(min(k, len(results)))]
        synth = " ".join(query_module.clean_snippet(doc.page_content) for doc, _ in results[:k])
        return (
            "Answer (synthesized from sources ["
            + ",".join(cited)
            + "]):\n\n"
            + (synth or "No relevant context found.")
        )
    if not api_key:
        print("[quiz] API key missing; falling back to synthesized answer.", file=sys.stderr)
        return query_module.synthesize_from_results(results, k)
    try:
        llm = query_module.load_chat_model(
            provider=provider,
            model_name=llm_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
    except Exception as exc:  # pragma: no cover - optional dependency/network
        print(f"[quiz] Failed to load chat model ({exc}); using synthesized answer.", file=sys.stderr)
        return query_module.synthesize_from_results(results, k)

    messages = [
        query_module.SystemMessage(content=query_module.SYSTEM_PROMPT),
        query_module.HumanMessage(content=query_module.compose_user_prompt(question, results)),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # pragma: no cover - network failure fallback
        print(f"[quiz] LLM call failed ({exc}); using synthesized answer.", file=sys.stderr)
        return query_module.synthesize_from_results(results, k)

    content = response.content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text", "")))
            else:
                parts.append(str(chunk))
        return " ".join(part for part in parts if part).strip()
    return str(content).strip()


def build_review_record(
    question_entry: Mapping[str, str],
    results,
    query_module,
    agent_mode: str,
    k: int,
    provider: str,
    llm_model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
) -> ReviewRecord:
    contexts: List[RetrievedContext] = []
    snippets: List[str] = []
    scores: List[float] = []
    for idx, (doc, score) in enumerate(results):
        metadata = getattr(doc, "metadata", {}) or {}
        serialisable_meta: Dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serialisable_meta[key] = value
            else:
                serialisable_meta[key] = str(value)
        snippet = query_module.clean_snippet(doc.page_content)
        contexts.append(
            RetrievedContext(rank=idx, score=float(score), metadata=serialisable_meta, snippet=snippet)
        )
        snippets.append(snippet)
        scores.append(float(score))

    answer = generate_answer(
        query_module=query_module,
        agent_mode=agent_mode,
        question=question_entry["question"],
        results=results,
        k=k,
        provider=provider,
        llm_model=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )

    max_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    overlap = compute_overlap_ratio(answer, snippets)

    return ReviewRecord(
        id=question_entry["id"],
        question=question_entry["question"],
        answer=answer,
        agent_mode=agent_mode,
        k=k,
        contexts=contexts,
        max_score=max_score,
        mean_score=mean_score,
        overlap_ratio=overlap,
    )


TAG_OPTIONS = [
    "retrieval-miss",
    "retrieval-partial",
    "too-low-k",
    "chunking-issue",
    "prompt-overreach",
    "ambiguous-question",
    "source-noise",
    "other",
]


def prompt_for_tags(current: List[str]) -> List[str]:
    print("Available tags (toggle by number, comma-separated):")
    for idx, tag in enumerate(TAG_OPTIONS, start=1):
        mark = "[x]" if tag in current else "[ ]"
        print(f"  {idx}. {mark} {tag}")
    raw = input("Select tags: ").strip()
    if not raw:
        return current
    selections = {s.strip() for s in raw.split(",") if s.strip()}
    updated = set(current)
    for sel in selections:
        try:
            pos = int(sel)
        except ValueError:
            continue
        if 1 <= pos <= len(TAG_OPTIONS):
            tag = TAG_OPTIONS[pos - 1]
            if tag in updated:
                updated.remove(tag)
            else:
                updated.add(tag)
    return sorted(updated)


def render_review_card(record: ReviewRecord, width: int) -> None:
    print("\n" + "=" * width)
    print(f"ID: {record.id}")
    print("Question:")
    print(textwrap.fill(record.question, width=width))
    print("\nAnswer:\n")
    print(textwrap.fill(record.answer or "(empty)", width=width))
    print("\nContexts:")
    print(describe_contexts(record.contexts, width=width))
    print("\nMetrics:")
    print(f"  max_score   : {record.max_score:.3f}")
    print(f"  mean_score  : {record.mean_score:.3f}")
    print(f"  overlap_ratio: {record.overlap_ratio:.3f}")
    print("Current labels:")
    print(f"  Faithful?       {record.faithful}")
    print(f"  Should abstain? {record.should_abstain}")
    print(f"  Tags:           {', '.join(record.tags) if record.tags else '(none)'}")
    if record.notes:
        print(f"  Notes:          {record.notes}")
    print("=" * width)


def interactive_loop(
    records: List[ReviewRecord],
    existing: Dict[str, MutableMapping[str, object]],
    jsonl_path: Path,
    csv_path: Path,
    page_width: int,
) -> None:
    saved_entries: List[MutableMapping[str, object]] = []
    id_to_entry: Dict[str, MutableMapping[str, object]] = {}
    if existing:
        for entry in existing.values():
            saved_entries.append(dict(entry))
            id_to_entry[entry["id"]] = entry

    for record in records:
        if record.id in id_to_entry:
            prior = id_to_entry[record.id]
            record.faithful = prior.get("faithful")
            record.should_abstain = prior.get("should_abstain")
            record.tags = list(prior.get("tags") or [])
            record.notes = str(prior.get("notes") or "")
            if prior.get("timestamp"):
                record.timestamp = str(prior["timestamp"])

        while True:
            render_review_card(record, width=page_width)
            print("Commands: [f] toggle faithful, [a] toggle abstain, [m] tags, [n] notes, [s] skip, [q] quit, [enter] save")
            cmd = input("Action: ").strip().lower()
            if not cmd:
                record.timestamp = datetime.now(timezone.utc).isoformat()
                entry = record.to_json()
                for idx, existing_entry in enumerate(saved_entries):
                    if existing_entry.get("id") == record.id:
                        saved_entries[idx] = entry
                        break
                else:
                    saved_entries.append(entry)
                id_to_entry[record.id] = entry
                save_reviews_jsonl(jsonl_path, saved_entries)
                save_reviews_csv(csv_path, saved_entries)
                print(f"Saved review for {record.id}.")
                break
            if cmd == "f":
                record.faithful = not record.faithful if record.faithful is not None else True
            elif cmd == "a":
                record.should_abstain = (
                    not record.should_abstain if record.should_abstain is not None else True
                )
            elif cmd == "m":
                record.tags = prompt_for_tags(record.tags)
            elif cmd == "n":
                record.notes = input("Notes: ").strip()
            elif cmd == "s":
                print(f"Skipping {record.id} without saving.")
                break
            elif cmd == "q":
                print("Exiting quiz. Progress saved.")
                save_reviews_jsonl(jsonl_path, saved_entries)
                save_reviews_csv(csv_path, saved_entries)
                return
            else:
                print("Unrecognised command. Please try again.")

    save_reviews_jsonl(jsonl_path, saved_entries)
    save_reviews_csv(csv_path, saved_entries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive human evaluation checklist")
    parser.add_argument("--questions", required=True, help="JSON file of question objects")
    parser.add_argument("--k", type=int, default=3, help="Number of contexts to retrieve")
    parser.add_argument(
        "--agent-mode",
        choices=["none", "pretend", "llm"],
        default="none",
        help="Answer generation strategy",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="Chat model name for llm mode")
    parser.add_argument("--provider", default="openai", help="LLM provider identifier")
    parser.add_argument("--api-key", dest="api_key", help="API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--base-url", dest="base_url", help="Optional OpenAI-compatible base URL")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=700, help="LLM max tokens")
    parser.add_argument("--out", default="data/human_review.jsonl", help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle question order")
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Only review items missing faithful/abstain labels",
    )
    parser.add_argument("--page-width", type=int, default=100, help="Wrap width for display")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    questions = load_questions(Path(args.questions))
    if args.shuffle:
        random.shuffle(questions)

    jsonl_path = Path(args.out)
    csv_path = jsonl_path.with_suffix(".csv")

    existing_entries_list = load_existing_reviews(jsonl_path) if args.resume else []
    if jsonl_path.exists() and not args.resume and existing_entries_list:
        raise RuntimeError(
            f"Output file {jsonl_path} already exists. Pass --resume to append or remove the file first."
        )
    existing_by_id: Dict[str, MutableMapping[str, object]] = {}
    for entry in existing_entries_list:
        entry_id = str(entry.get("id"))
        if entry_id:
            existing_by_id[entry_id] = entry

    if args.only_unlabeled:
        questions = [
            q
            for q in questions
            if q["id"] not in existing_by_id
            or (
                existing_by_id[q["id"]].get("faithful") is None
                or existing_by_id[q["id"]].get("should_abstain") is None
            )
        ]

    if not questions:
        print("No questions to review.")
        return

    query_module = load_query_module()
    embed = HuggingFaceEmbeddings(model_name=args.model)
    try:
        store = query_module.load_vector_store(query_module.CHROMA_DIR, embed)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    records: List[ReviewRecord] = []
    for question_entry in questions:
        results = query_module.retrieve_contexts(store, question_entry["question"], args.k)
        record = build_review_record(
            question_entry=question_entry,
            results=results,
            query_module=query_module,
            agent_mode=args.agent_mode,
            k=args.k,
            provider=args.provider,
            llm_model=args.llm_model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
        records.append(record)

    existing_map = {entry_id: entry for entry_id, entry in existing_by_id.items()}
    interactive_loop(records, existing_map, jsonl_path, csv_path, page_width=args.page_width)


if __name__ == "__main__":
    main()

