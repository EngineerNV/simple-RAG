"""04_report.py — aggregate human quiz labels into actionable summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


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
            raise ValueError(f"Unsupported file extension for {path}. Use JSONL or CSV from 03_quiz.py")
    return records


def coerce_bool(value) -> bool | None:
    if value in (True, False):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return None


def extract_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(records)
    faithful_labels: List[bool] = []
    abstain_labels: List[bool] = []
    overreach_labels: List[bool] = []
    max_scores: List[float] = []
    mean_scores: List[float] = []
    overlap_ratios: List[float] = []
    tags_counter: Counter[str] = Counter()

    for row in records:
        faithful = coerce_bool(row.get("faithful"))
        abstain = coerce_bool(row.get("should_abstain"))
        max_score = extract_float(row.get("max_score"))
        mean_score = extract_float(row.get("mean_score"))
        overlap = extract_float(row.get("overlap_ratio"))

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
    if not values:
        return "(no data)"
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    counts: List[int] = [0 for _ in buckets]
    for value in values:
        for idx, (low, high) in enumerate(buckets):
            upper = 1.0 if math.isclose(high, 1.0) else high
            if low <= value < upper or (math.isclose(value, upper) and upper == 1.0):
                counts[idx] += 1
                break
    lines = []
    for (low, high), count in zip(buckets, counts):
        bar = "#" * count
        lines.append(f"[{low:.1f}, {high:.1f}) {count:>3}: {bar}")
    return "\n".join(lines)


def print_console_summary(summary: Mapping[str, object]) -> None:
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
        most_common = tags.most_common(8)
        for tag, count in most_common:
            bar = "#" * count
            print(f"  {tag:20s} {count:>3} {bar}")


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


def format_table(rows: List[Sequence[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
    lines = []
    for row in rows:
        padded = [str(cell).ljust(width) for cell, width in zip(row, widths)]
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def group_by(records: Sequence[Mapping[str, object]], key: str) -> Dict[str, List[Mapping[str, object]]]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(key, "unknown"))].append(row)
    return grouped


def compute_threshold(values: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not values or not labels or len(values) != len(labels):
        return None
    paired = sorted(zip(values, labels), key=lambda pair: pair[0])
    unique_values = sorted({value for value, _ in paired})
    best_threshold = None
    best_accuracy = -1.0
    for threshold in unique_values:
        predictions = [value >= threshold for value, _ in paired]
        accuracy = sum(pred == label for pred, (_, label) in zip(predictions, paired)) / len(paired)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def recommend_thresholds(records: Sequence[Mapping[str, object]]) -> Dict[str, float | None]:
    faithful_mask: List[bool] = []
    max_scores: List[float] = []
    overlaps: List[float] = []
    for row in records:
        faithful = coerce_bool(row.get("faithful"))
        if faithful is None:
            continue
        max_score = extract_float(row.get("max_score"))
        overlap = extract_float(row.get("overlap_ratio"))
        if max_score is not None and overlap is not None:
            faithful_mask.append(faithful)
            max_scores.append(max_score)
            overlaps.append(overlap)
    tau_s = compute_threshold(max_scores, faithful_mask) if faithful_mask else None
    tau_f = compute_threshold(overlaps, faithful_mask) if faithful_mask else None
    return {"tau_s": tau_s, "tau_f": tau_f}


def write_markdown(
    path: Path,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    thresholds: Mapping[str, float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Human Evaluation Report")
    lines.append("")
    lines.append(f"Total reviewed examples: **{summary['total']}**")
    lines.append(
        f"Faithful: **{summary['faithful_pct']:.1%}**, Proper abstain: **{summary['abstain_pct']:.1%}**, Overreach: **{summary['overreach_pct']:.1%}**"
    )
    lines.append("")

    def stats_table(group_key: str) -> List[str]:
        group_rows = [[group_key, "N", "Faithful %", "Abstain %", "Overreach %"]]
        grouped = group_by(records, group_key)
        for key, group in sorted(grouped.items()):
            group_summary = summarise(group)
            group_rows.append(
                [
                    key,
                    str(len(group)),
                    f"{group_summary['faithful_pct']:.1%}",
                    f"{group_summary['abstain_pct']:.1%}",
                    f"{group_summary['overreach_pct']:.1%}",
                ]
            )
        return ["## Metrics by " + group_key, "", format_table(group_rows), ""]

    lines.extend(stats_table("agent_mode"))
    lines.extend(stats_table("k"))

    lines.append("## Score Distributions")
    lines.append("")
    if summary.get("max_scores"):
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
        lines.append("## Tag Spotlight")
        lines.append("")
        lines.append("| Tag | Count | Recommendation |")
        lines.append("| --- | ---: | --- |")
        for tag, count in tags.most_common():
            recommendation = ACTIONABLE_INTERVENTIONS.get(tag, "Review associated notes.")
            lines.append(f"| {tag} | {count} | {recommendation} |")
        lines.append("")

    lines.append("## Actionable Playbook")
    lines.append("")
    for tag, recommendation in ACTIONABLE_INTERVENTIONS.items():
        lines.append(f"- **{tag}** — {recommendation}")
    lines.append("")

    tau_s = thresholds.get("tau_s")
    tau_f = thresholds.get("tau_f")
    lines.append("## Threshold Suggestions")
    lines.append("")
    if tau_s is None or tau_f is None:
        lines.append("Not enough labeled data to suggest thresholds yet.")
    else:
        lines.append(
            f"Consider enforcing max_score ≥ **{tau_s:.3f}** (τ_s) and overlap_ratio ≥ **{tau_f:.3f}** (τ_f)"
            " to favour faithful answers."
        )
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise human quiz annotations")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="JSONL/CSV files from 03_quiz.py")
    parser.add_argument("--out", default="reports/human_eval_report.md", help="Markdown report destination")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = [Path(p) for p in args.inputs]
    records = load_datasets(paths)
    if not records:
        print("No records loaded; nothing to summarise.")
        return

    summary = summarise(records)
    print_console_summary(summary)
    thresholds = recommend_thresholds(records)
    write_markdown(Path(args.out), records, summary, thresholds)


if __name__ == "__main__":
    main()

