# Human Evaluation Guide

This guide explains how to run the interactive quiz (`03_quiz.py`), interpret the
saved JSONL/CSV rows, and turn the aggregated report (`04_report.py`) into
practical next steps for the retrieval pipeline.

## 1. Running the Quiz

```bash
python scripts/03_quiz.py \
  --questions data/dev_prompts.json \
  --agent-mode pretend \
  --k 3 \
  --out data/human_review.jsonl \
  --resume
```

Key flags:

* `--questions`: JSON array of `{"id": ..., "question": ...}` objects.
* `--agent-mode`: matches `scripts/02_query.py` (`none`, `pretend`, `llm`).
* `--resume`: append to an existing JSONL instead of overwriting it.
* `--only-unlabeled`: skip items that already have both `faithful` and
  `should_abstain` decisions.
* `--page-width`: wrap interactive output to match your terminal width.

During the quiz press:

* `f` to toggle the **Faithful** flag.
* `a` to toggle **Should Abstain**.
* `m` to open the tag checklist (retrieval-miss, prompt-overreach, etc.).
* `n` to add free-form notes.
* `Enter` to save the current card and move on, `s` to skip, `q` to quit.

Each save writes a JSON line resembling:

```json
{
  "id": "q002",
  "question": "Where is the Chroma index persisted and how is it rebuilt?",
  "answer": "Answer (synthesized from sources [0,1]): ...",
  "agent_mode": "pretend",
  "k": 3,
  "contexts": [
    {
      "rank": 0,
      "score": 0.812,
      "metadata": {"source": "data/corpus/pipeline.md"},
      "snippet": "The Chroma index persists under data/chroma ..."
    }
  ],
  "max_score": 0.812,
  "mean_score": 0.654,
  "overlap_ratio": 0.41,
  "faithful": true,
  "should_abstain": false,
  "tags": ["prompt-overreach"],
  "notes": "Call out missing rebuild instructions.",
  "timestamp": "2024-05-10T18:22:04.123456+00:00"
}
```

The accompanying CSV mirrors the headline columns so that spreadsheets or BI
tools can ingest the dataset quickly.

## 2. Reading the Report

Aggregate existing annotations with:

```bash
python scripts/04_report.py --in data/human_review.jsonl --out reports/human_eval_report.md
```

The console output surfaces headline KPIs:

* **Faithful %** – proportion of answers marked `faithful=true`.
* **Proper abstain %** – proportion of answers tagged `should_abstain=true`.
* **Overreach rate** – answers that were **not** faithful and **not** abstaining.
* Mean/median **max_score** and **mean_score** – quick proxies for retrieval
  strength.

The Markdown report adds:

* Tables sliced by `agent_mode` and `k` so you can compare configurations.
* Histogram-style distributions (ASCII buckets) for retrieval scores.
* Tag spotlight table with tailored interventions.
* Threshold suggestions (`τ_s` for `max_score`, `τ_f` for overlap) showing where a
  simple rule might separate faithful and unfaithful answers.

To compare progress over time, commit each `reports/human_eval_report.md` to a
dated branch or folder (e.g., `reports/2024-05-10_human_eval.md`) and diff the
tables/metrics between runs.

## 3. Turning Metrics into Action

Use the checklist below to translate the most common tags into pipeline tweaks:

* **retrieval-miss** – raise `k`, improve metadata filters, or add new documents.
* **retrieval-partial** – inspect split boundaries; chunk overlap may be too
  narrow.
* **too-low-k** – increase the CLI `--k` or implement dynamic cut-offs based on
  similarity score.
* **chunking-issue** – rerun the index build (`python scripts/01_build_index.py`)
  with larger `--chunk-size` / `--chunk-overlap` arguments (see CLI in the
  script) and ensure the build summary lists the expected document count and
  metadata keys.
* **prompt-overreach** – add refusal exemplars to `SYSTEM_PROMPT` and emphasise
  abstention when no evidence is cited.
* **ambiguous-question** – create a clarifier prompt template or follow-up flow
  before answering.
* **source-noise** – audit the corpus (e.g., `data/corpus/`) for outdated or
  irrelevant files and rebuild once cleaned.
* **other** – inspect the reviewer notes in the JSONL for bespoke actions.

After adjusting chunk sizes, overlaps, or the prompt, rebuild and re-run:

```bash
python scripts/01_build_index.py --chunk-size 1200 --chunk-overlap 200
python scripts/03_quiz.py --questions data/dev_prompts.json --resume
python scripts/04_report.py --in data/human_review.jsonl
```

Confirm that the build step prints the intended collection name, document count,
and metadata keys—these cues verify that the configuration changes took effect.

## 4. Comparing Reports Over Time

* Store previous reports in `reports/` with date-stamped filenames.
* Use `git diff reports/old.md reports/new.md` (or a Markdown-aware diff viewer)
  to spot shifts in faithful %, abstain %, and top tags.
* Track the recommended thresholds; converging `τ_s` and `τ_f` values indicate
  your retrieval quality is stabilising, while big swings suggest the pipeline
  still needs tuning.

Iterate until the overreach rate is comfortably low and the dominant tags map to
known, addressable causes.

