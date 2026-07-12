"""Summarise human quiz annotations into console and Markdown snapshots.

This optional helper ingests JSONL/CSV files produced by ``06_quiz.py`` and
prints an easy-to-skim digest (coverage, faithfulness/abstention trends, tag
frequencies) before emitting a lightweight Markdown report. The intent is to
keep post-quiz analysis approachable while leaving room for teams to build
their own bespoke tooling on top of the saved annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence


def load_jsonl(path: Path) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_csv(path: Path) -> List[MutableMapping[str, object]]:
    rows: List[MutableMapping[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tags = row.get("tags")
            if isinstance(tags, str):
                row["tags"] = [t for t in tags.split(",") if t]
            rows.append(row)
    return rows


def load_datasets(paths: Sequence[Path]) -> List[MutableMapping[str, object]]:
    records: List[MutableMapping[str, object]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            records.extend(load_jsonl(path))
        elif path.suffix.lower() == ".csv":
            records.extend(load_csv(path))
        else:
            raise ValueError(f"Unsupported file extension for {path}. Use JSONL or CSV from 06_quiz.py")
    return records


def _parse_optional_bool(value) -> bool | None:
    """Return ``True``/``False`` when quiz exports store booleans as strings."""

    if value in (True, False):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return None


def compute_summary(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Aggregate quiz rows into metric arrays plus tag counts."""

    total = len(records)
    faithful_labels: List[bool] = []
    abstain_labels: List[bool] = []
    overreach_labels: List[bool] = []
    max_scores: List[float] = []
    mean_scores: List[float] = []
    overlap_ratios: List[float] = []
    tags_counter: Counter[str] = Counter()

    for row in records:
        faithful = _parse_optional_bool(row.get("faithful"))
        abstain = _parse_optional_bool(row.get("should_abstain"))
        try:
            max_score = float(row.get("max_score")) if row.get("max_score") is not None else None
        except (TypeError, ValueError):
            max_score = None
        try:
            mean_score = float(row.get("mean_score")) if row.get("mean_score") is not None else None
        except (TypeError, ValueError):
            mean_score = None
        try:
            overlap = float(row.get("overlap_ratio")) if row.get("overlap_ratio") is not None else None
        except (TypeError, ValueError):
            overlap = None

        if faithful is not None:
            faithful_labels.append(faithful)
        if abstain is not None:
            abstain_labels.append(abstain)
        if faithful is not None and abstain is not None:
            overreach_labels.append(not faithful and not abstain)
        if max_score is not None:
            max_scores.append(max_score)
        if mean_score is not None:
            mean_scores.append(mean_score)
        if overlap is not None:
            overlap_ratios.append(overlap)

        tags = row.get("tags")
        if isinstance(tags, str):
            tags = [t for t in tags.split(",") if t]
        if isinstance(tags, Iterable):
            for tag in tags:
                tags_counter[str(tag)] += 1

    def pct(values: Sequence[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "total": total,
        "faithful_pct": pct(faithful_labels),
        "abstain_pct": pct(abstain_labels),
        "overreach_pct": pct(overreach_labels),
        "max_scores": max_scores,
        "mean_scores": mean_scores,
        "overlap_ratios": overlap_ratios,
        "tags": tags_counter,
    }


def render_distribution(values: Sequence[float]) -> str:
    """Bucket metric arrays into ASCII histograms for quick scanning."""

    if not values:
        return "(no data)"
    bucket_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    counts = [0] * len(bucket_labels)
    for value in values:
        idx = min(len(counts) - 1, max(0, int(value * 5)))
        counts[idx] += 1
    lines = []
    for label, count in zip(bucket_labels, counts):
        bar = "#" * count
        lines.append(f"[{label}] {count:>3}: {bar}")
    return "\n".join(lines)


def print_console_summary(summary: Mapping[str, object]) -> None:
    """Emit the headline statistics developers tend to ask for first."""

    print("Human review summary")
    print(f"Total reviewed: {summary['total']}")
    print(f"Faithful %: {summary['faithful_pct']:.1%}")
    print(f"Proper abstain %: {summary['abstain_pct']:.1%}")
    print(f"Overreach rate: {summary['overreach_pct']:.1%}")

    max_scores = summary.get("max_scores", [])
    mean_scores = summary.get("mean_scores", [])
    if max_scores:
        print(f"Max score mean/median: {statistics.mean(max_scores):.3f} / {statistics.median(max_scores):.3f}")
        print("Max score distribution:")
        print(render_distribution(max_scores))
    if mean_scores:
        print(f"Mean score mean/median: {statistics.mean(mean_scores):.3f} / {statistics.median(mean_scores):.3f}")
        print("Mean score distribution:")
        print(render_distribution(mean_scores))

    tags: Counter[str] = summary.get("tags", Counter())
    if tags:
        print("Top tags:")
        most_common = tags.most_common(5)
        for tag, count in most_common:
            bar = "#" * count
            print(f"  {tag:20s} {count:>3} {bar}")


# Map quiz tags to suggested remediation steps surfaced in the Markdown output.
ACTIONABLE_INTERVENTIONS = {
    "retrieval-miss": "Increase k, add metadata filters, or expand corpus coverage.",
    "retrieval-partial": "Review chunk segmentation or add follow-up retrieval passes.",
    "too-low-k": "Bump k or implement score thresholds to widen the net.",
    "chunking-issue": "Rebuild index with larger chunks and overlapping windows.",
    "prompt-overreach": "Add refusal exemplars and tighten the system prompt.",
    "ambiguous-question": "Introduce clarifier prompts or request follow-up questions.",
    "source-noise": "Clean the corpus or adjust filters to drop noisy documents.",
    "other": "Review notes column for bespoke actions.",
}


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    """Persist a concise Markdown overview for asynchronous sharing."""

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Human Evaluation Snapshot")
    lines.append("")
    lines.append(f"Total reviewed examples: **{summary['total']}**")
    lines.append(
        f"Faithful: **{summary['faithful_pct']:.1%}**, Proper abstain: **{summary['abstain_pct']:.1%}**, Overreach: **{summary['overreach_pct']:.1%}**"
    )
    lines.append("")

    if summary.get("max_scores"):
        lines.append("## Retrieval score trends")
        lines.append("")
        lines.append("**Max score buckets**")
        lines.append("```")
        lines.append(render_distribution(summary["max_scores"]))
        lines.append("```")
        if summary.get("mean_scores"):
            lines.append("**Mean score buckets**")
            lines.append("```")
            lines.append(render_distribution(summary["mean_scores"]))
            lines.append("```")
        lines.append("")

    tags: Counter[str] = summary.get("tags", Counter())
    if tags:
        lines.append("## What reviewers flagged")
        lines.append("")
        for tag, count in tags.most_common():
            guidance = ACTIONABLE_INTERVENTIONS.get(tag, "Review associated notes.")
            lines.append(f"- **{tag}** ({count}) — {guidance}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define CLI arguments so teams can point at their exported quiz files."""

    parser = argparse.ArgumentParser(description="Summarise human quiz annotations")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="JSONL/CSV files from 06_quiz.py")
    parser.add_argument("--out", default="reports/human_eval_report.md", help="Markdown report destination")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = [Path(p) for p in args.inputs]
    records = load_datasets(paths)
    if not records:
        print("No records loaded; nothing to summarise.")
        return

    summary = compute_summary(records)
    print_console_summary(summary)
    write_markdown(Path(args.out), summary)


if __name__ == "__main__":
    main()

