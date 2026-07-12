# Human Evaluation Guide

This guide explains how to capture human feedback with `scripts/06_quiz.py`,
understand the saved JSONL/CSV fields, and turn the lightweight report from
`scripts/report.py` into concrete retrieval or prompting changes. The commands
below reflect the current CLI options exposed by the repository.

## 1. Prepare Your Corpus and Prompts

1. Add your markdown sources under `data/corpus/` (e.g., `data/corpus/manual.md`).
2. Build the index once the corpus looks right (the script always refreshes the
   collection):
   ```bash
   python scripts/01_build_index.py
   ```
3. Create a questions file such as `data/questions/dev.json` containing objects
   with `id` and `question` keys:
   ```json
   [
     {"id": "ingest-001", "question": "How do we rebuild the vector store?"},
     {"id": "prompt-002", "question": "When should the assistant abstain?"}
   ]
   ```

## 2. Run the Interactive Quiz

Launch the reviewer loop with the same retrieval/LLM settings you ship in the
application:

```bash
python scripts/06_quiz.py \
  --questions data/questions/dev.json \
  --agent-mode pretend \
  --k 3 \
  --out data/human_review.jsonl \
  --resume
```

Key tips:

* `--agent-mode` mirrors `scripts/02_query.py` (`none`, `pretend`, `llm`). Use
  the mode you want to audit.
* `--resume` appends to an existing JSONL so you can label in multiple sessions.
* `--only-unlabeled` filters out entries that already have both faithfulness and
  abstain labels.
* `--page-width` controls terminal wrapping for questions, answers, and
  snippets.

While reviewing, press:

* `f` to toggle **Faithful**.
* `a` to toggle **Should Abstain**.
* `m` to open the tag checklist (retrieval-miss, prompt-overreach, etc.).
* `n` to add free-form notes.
* `Enter` to save the current card, `s` to skip without saving, `q` to quit.

Every time you save, the quiz prints a **progress snapshot** summarising the
running faithful %, abstain %, overreach rate, and the most common tags. This
feedback loop highlights trends immediately—no external tooling required.

### Saved Row Anatomy

Each save appends a JSON object similar to:

```json
{
  "id": "ingest-001",
  "question": "How do we rebuild the vector store?",
  "answer": "Answer (synthesised from sources [0,1]): ...",
  "agent_mode": "pretend",
  "k": 3,
  "contexts": [
    {"rank": 0, "score": 0.81, "metadata": {"source": "handbook.md"}, "snippet": "..."}
  ],
  "max_score": 0.81,
  "mean_score": 0.63,
  "overlap_ratio": 0.42,
  "faithful": true,
  "should_abstain": false,
  "tags": ["prompt-overreach"],
  "notes": "Mention missing rebuild CLI.",
  "timestamp": "2024-05-11T18:22:04.123456+00:00"
}
```

A mirrored CSV keeps the headline columns for spreadsheet workflows. Reviewers
can also edit the JSONL manually if a typo slips through.

## 3. Generate a Snapshot Report

After a labeling session, run:

```bash
python scripts/report.py --in data/human_review.jsonl --out reports/human_eval_report.md
```

The script prints totals, faithful %, abstain %, overreach %, score
means/medians, score distributions, and the top tags with counts. The optional
Markdown file captures the same summary plus the actionable guidance for async
sharing (paste it into docs, issues, or Slack).

## 4. Act on the Feedback

Use the built-in tag recommendations as a checklist:

* **retrieval-miss** — raise `k`, introduce metadata filters, or broaden the
  corpus.
* **retrieval-partial** — adjust chunk sizes/overlap so facts do not split across
  documents.
* **too-low-k** — increase the CLI `--k` or implement a similarity threshold
  before truncating.
* **chunking-issue** — rebuild via `scripts/01_build_index.py` with larger chunks
  or higher overlap.
* **prompt-overreach** — tighten `SYSTEM_PROMPT` and include refusal exemplars so
  the LLM abstains without evidence.
* **ambiguous-question** — add clarifying prompts or template the answer format.
* **source-noise** — clean noisy corpus files and rebuild the index.
* **other** — read the free-form notes for bespoke fixes.

## 5. Iterate

1. Adjust the pipeline (retrieval settings, prompt, corpus) based on the top
   tags.
2. Rebuild the index if corpus content changes:
   ```bash
   python scripts/01_build_index.py
   ```
3. Re-run the quiz on the same question set (use `--resume` to append or drop
   the previous JSONL for a fresh run).
4. Compare report snapshots over time by keeping date-stamped copies in
   `reports/` and using `git diff` or a Markdown-aware viewer.

Repeat until overreach is rare and dominant tags map to deliberate, fixable
causes.
