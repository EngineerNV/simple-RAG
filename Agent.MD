# Agent Guide for `simple-RAG`

This guide orients automation agents and contributors so you can ship focused, safe changes without spelunking the entire tree. It complements the top-level `README.md` with contributor-facing details, coding conventions, and integration points.

## Quick setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-min.txt    # or requirements.txt for full stack

# Optional: test tooling
pip install -r requirements-dev.txt    # provides pytest

# Build and smoke-test
python scripts/00_ingest.py
python scripts/01_build_index.py
python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend
```

Environment variables:

- **Auto-detected multi-provider support**: Set any one of these keys and the scripts will automatically detect the provider:
  - `OPENAI_API_KEY` — for OpenAI models (checked first)
  - `GOOGLE_API_KEY` — for Google Gemini models (checked second)
  - `ANTHROPIC_API_KEY` — for Anthropic Claude models (checked third)
- Override auto-detection with `--provider openai|gemini|claude` flag
- Install provider dependencies as needed:
  - `pip install langchain-openai` (default, usually already present)
  - `pip install langchain-google-genai` (for Gemini)
  - `pip install langchain-anthropic` (for Claude)
- **Note:** Only ONE API key is needed. The first key found determines the provider unless you explicitly override with `--provider`.

Quick .env setup (Windows PowerShell):

```powershell
Copy-Item .env.example .env
# Edit .env and set exactly one of these:
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional ingestion paths/config
# CORPUS_DIR=data/corpus
```

## Codebase map (what lives where)

Core pipeline (numbered in run order):

1) `scripts/00_ingest.py` — header-aware, two-pass chunking of Markdown files from `data/corpus/`: pass 1 splits at heading boundaries (`#`/`##`/`###`/`####`) via `MarkdownHeaderTextSplitter`, so every chunk carries its enclosing heading(s) as metadata under those same keys (e.g. `{"#": "...", "##": "...", "###": "Pikachu (...)"}`); pass 2 re-splits any section still over `INGEST_CHUNK_SIZE` tokens with the existing token-aware splitter, preserving overlap and heading metadata on the sub-chunks. Emits LangChain `Document` chunks with `chunk_index`/`chunk_count` and `source`/`source_path` metadata alongside the heading keys. Only `.md` content is supported for now.

2) `scripts/01_build_index.py` — embeds chunks with MiniLM (`HuggingFaceEmbeddings`) and (re)creates `data/chroma/`. Summary prints collection, persist dir, Documents embedded, and a preview of metadata keys; also echoes ingest chunk size/overlap.

3) `scripts/02_query.py` — connects to the persisted Chroma store; retrieval modes: `none`, `pretend`, `llm`. Includes a lightweight reranker that blends retriever score with lexical overlap and stores `combined_score` in `Document.metadata` for display.

4) `scripts/03_eval.py` — lexical heuristics over saved rows (faithfulness/abstain).

5) `scripts/03_quiz.py` + `scripts/report.py` — collect human judgements and summarize.

6) `scripts/04_llm_api.py` — prompt assembly and LLM smoke test with usage display.

7) `scripts/05_chat_cli.py` — interactive chat agent: topic gate → RAG decider → optional rewrite → retrieve → rerank → compose → answer. Shows contexts when `--show-context`.

Agent orchestration and utils:

- `agent_orchestration_helper.py` — loads topic inventory from `rag_content.json`; builds structured decider/rewriter; assembles user payloads (with persona) for the LLM.
- `utils/inventory_view.py` — formats specialization topics.
- `utils/persona.py` — persona preamble builder.
- `utils/rejections.py` — structured off-topic refusals.
- `utils/text_sanitize.py` — suggestion filtering.
- `utils/topic_gate.py` — on-topic classifier with a confidence floor.

Data/config:

- `rag_content.json` — JSON controlling the agent’s scope (topics/inventory).
- `data/corpus/` — place `.md` files here (hidden files ignored).
- `data/chroma/` — auto-created vector index storage.

## Topic scope and content policy

Scope is defined in `rag_content.json`:

- `rag_topic_inventory` (multi-line description) guides the decider/rewriter.
- `specialization_topics` (list) appear in the persona and refusals.

Content format: keep it simple — only Markdown (`.md`) files are supported for ingestion today. The ingester splits at heading boundaries first (every chunk carries its `#`/`##`/`###`/`####` heading path as metadata), then re-splits any section still over `INGEST_CHUNK_SIZE` tokens with a tiktoken-aware splitter, preserving overlap. Headings therefore double as the retrieval-time section-title metadata — keep entities (a species, a gym leader, a case study) each under their own heading rather than bundled as bullets under a shared one, or their chunks won't get a distinct title.

## Reranker (where and how)

- Location: `scripts/02_query.py` → `rerank_results(results, question, alpha=0.5)`.
- Method: min–max normalize the retriever score; compute lexical-overlap with regex tokenization; blend by `combined = alpha * retriever_norm + (1-alpha) * lexical`. The reranker writes `combined_score`, `lexical_overlap`, `retriever_norm` to `doc.metadata` and reorders results; it does not overwrite the original retriever score.
- Displays: `emit_contexts`, `compose_user_prompt`, and the chat CLI’s tables show `score` and `rerank` when present.
- Robustness: failure is non-fatal; the script falls back to the original ordering.

## CLI agent loop (chat)

Flow: topic gate → RAG decider → (optional) rewrite → retrieve → rerank → compose → LLM → answer. The structured helpers are in `agent_orchestration_helper.py`; the topic gate lives in `utils/topic_gate.py`.

Low-confidence fail-safes are intentionally asymmetric: `topic_gate` (`utils/topic_gate.py`) fails **open** (`on_topic=True`) below its confidence floor, so an uncertain classification doesn't wrongly refuse a legitimate question; `decide_use_rag` (`agent_orchestration_helper.py`) fails **closed** (`use_rag=False`), so an uncertain classification doesn't inject irrelevant retrieved context. Both defaults favor answering over refusing/injecting noise — keep this in mind before "fixing" one to match the other.

Useful flags:

- `scripts/05_chat_cli.py --show-context --debug`
- `scripts/02_query.py -q "..." --agent-mode pretend|llm --k 5`

## Logging and warnings

- Do not change packages for deprecation churn. We suppress common LangChain deprecation warnings across scripts using `warnings.filterwarnings` while keeping imports stable.
- Keep outputs user-friendly; when adding logs, prefer concise one-liners and preserve current phrasing that tests assert on (e.g., “Documents embedded”).

## Tests and dev tooling

- Dev-only requirements live in `requirements-dev.txt` (adds `pytest`).
- Run tests:

  ```powershell
  pip install -r requirements-dev.txt
  python -m pytest -q
  ```

- Tests expect:
  - `01_build_index.py` to print “Documents embedded”.
  - `00_ingest.py` to return ≥2 docs for simple H2-structured input and preserve the H1 title in metadata under `"#"` (the general path now, not a small-file fallback).
  - `02_query.py` helpers to coerce/format scores and compose messages.

## Contribution checklist (for agents)

- Respect content policy: ingest `.md` only; don’t commit corpus or vector stores.
- Prefer minimal, localized patches; keep public signatures stable.
- Update `README.md` and this `Agent.MD` when you alter workflow, flags, or outputs.
- Add tests for new behavior and ensure all tests pass with `python -m pytest -q`.
- Avoid changing dependency sets; use fallbacks and warning filters for compatibility.

Happy building — and keep it simple, grounded, and well-documented.
