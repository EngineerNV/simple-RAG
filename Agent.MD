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

1) `scripts/00_ingest.py` — token-aware chunking of Markdown files from `data/corpus/` (+ small H2 fallback for tiny files). Emits LangChain `Document` chunks with `chunk_index/chunk_count` and `source` metadata. Only `.md` content is supported for now.

2) `scripts/01_build_index.py` — embeds chunks with MiniLM (`HuggingFaceEmbeddings`) and (re)creates `data/chroma/`. Summary prints collection, persist dir, Documents embedded, and a preview of metadata keys; also echoes ingest chunk size/overlap. Also writes a small `_index_meta.json` sidecar next to the store recording the embedding model name (filename in `utils.settings.INDEX_METADATA_FILENAME`).

3) `scripts/02_query.py` — connects to the persisted Chroma store; retrieval modes: `none`, `pretend`, `llm`. Includes a lightweight reranker that blends retriever score with lexical overlap and stores `combined_score` in `Document.metadata` for display. `load_vector_store` reads the `_index_meta.json` sidecar and emits a `RuntimeWarning` if the querying embedding model doesn't match the one that built the index (silent if the sidecar is missing/unreadable or either model name is unavailable).

4) `scripts/03_eval.py` — lexical heuristics over saved rows (faithfulness/abstain).

5) `scripts/06_quiz.py` + `scripts/report.py` — collect human judgements and summarize.

6) `scripts/04_llm_api.py` — prompt assembly and LLM smoke test with usage display.

7) `scripts/05_chat_cli.py` — interactive chat agent: one router call (scope + retrieval decision + query rewrite) → retrieve → grounded answer. Shows contexts when `--show-context`.

8) `scripts/07_debug_chat.py` — debug chat TUI; builds its session through `05_chat_cli.py`'s shared `create_chat_runtime` factory, then renders a two-pane Rich layout after each turn: left pane is the rolling conversation (error turns shown in red), right pane shows router decisions (on_topic, use_rag, confidence, search query, reason), per-chunk retrieval scores (score / rerank / lexical), per-stage timing (router ms, retrieval ms, LLM ms, total ms), and memory stats (turn count, raw message count, summary length). Use `--chat-lines` to control the chat history viewport depth. Panel/timing tests: `tests/test_debug_chat.py`.

Agent orchestration and utils:

- `agent_orchestration_helper.py` — loads topic inventory from `rag_content.json`; the structured router (`route_turn`), the prompt contract (`build_system_prompt`, `format_context_documents`), and the testable `ChatSession` turn handler.
- `utils/llm.py` — provider auto-detection, per-provider default models (`DEFAULT_MODELS`), and the single chat-model factory shared by all scripts.
- `utils/settings.py` — `CHROMA_DB_PATH`-aware Chroma directory and embedding defaults.
- `utils/textproc.py` — snippet cleaning / metadata formatting / lexical-overlap helpers.
- `utils/chat_history.py` — `SummaryBufferHistory` rolling-summary conversation memory.
- `utils/inventory_view.py` — formats specialization topics.
- `utils/persona.py` — persona text used in the chat system prompt.
- `utils/rejections.py` — structured off-topic refusals.
- `utils/warnings_filter.py` — shared LangChain deprecation-warning filters.

Data/config:

- `rag_content.json` — JSON controlling the agent’s scope (topics/inventory).
- `data/corpus/` — place `.md` files here (hidden files ignored).
- `data/chroma/` — auto-created vector index storage.

## Topic scope and content policy

Scope is defined in `rag_content.json`:

- `rag_topic_inventory` (multi-line description) guides the router's retrieval decision.
- `specialization_topics` (list) appear in the persona and refusals.

Content format: keep it simple — only Markdown (`.md`) files are supported for ingestion today. The ingester uses a tiktoken-aware splitter with overlap. For very small files, it may use H2 headings to create multiple chunks (so tests and UX see more than one chunk when structure exists).

## Reranker (where and how)

- Location: `scripts/02_query.py` → `rerank_results(results, question, alpha=0.5)`.
- Method: min–max normalize the retriever score; compute lexical-overlap with regex tokenization; blend by `combined = alpha * retriever_norm + (1-alpha) * lexical`. The reranker writes `combined_score`, `lexical_overlap`, `retriever_norm` to `doc.metadata` and reorders results; it does not overwrite the original retriever score.
- Displays: `emit_contexts`, `compose_user_prompt`, and the chat CLI’s tables show `score` and `rerank` when present.
- Robustness: failure is non-fatal; the script falls back to the original ordering.

## CLI agent loop (chat)

Flow: router (one structured call: scope + use-RAG + rewritten search query) → retrieve → grounded LLM answer. Off-topic requests get a structured refusal; low router confidence or a router error falls back to retrieve-and-abstain. All of it lives in `agent_orchestration_helper.py` (`route_turn`, `ChatSession`); the CLI is a thin rendering shell. Offline behaviour tests: `tests/test_chat_session.py`.

Useful flags:

- `scripts/05_chat_cli.py --show-context --debug`
- `scripts/07_debug_chat.py --chat-lines 30 --debug`  (debug TUI: router + retrieval + timing + memory panel)
- `scripts/02_query.py -q "..." --agent-mode pretend|llm --k 5`

## Logging and warnings

- Do not change packages for deprecation churn. Common LangChain deprecation warnings are suppressed via `utils.warnings_filter.suppress_langchain_warnings()` — call it instead of copy-pasting filter blocks.
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
  - `00_ingest.py` to return ≥2 docs for simple H2-structured input and preserve the H1 title in metadata under `"#"`; non-`.md` files are skipped and a file that fails UTF-8 decoding is skipped (with a warning) rather than crashing the whole ingest run.
  - `02_query.py` helpers to coerce/format scores and compose messages; `load_vector_store` warns (doesn't raise) on an embedding-model mismatch against the index's `_index_meta.json` sidecar and stays silent when the sidecar is absent or the models match.
  - `01_build_index.py`'s `persist_chroma` writes that `_index_meta.json` sidecar when the embeddings object exposes `model_name`, and skips it silently otherwise.
  - `03_eval.py`'s `load_eval_data` rejects a JSON eval file whose top level isn't a list instead of silently treating dict keys as records.
  - `06_quiz.py`'s `main` refuses to overwrite an existing non-empty review output file unless `--resume` is passed (`tests/test_quiz.py`).
  - `tests/test_router.py` — router pass-through, low-confidence and error fallbacks to retrieve-and-abstain, prompt contents.
  - `tests/test_chat_session.py` — the chat contract: `<documents>` evidence (escaped, most-relevant first, never sanitized), history stores only the clean user message, off-topic rejection with static fallback, an LLM failure leaves no history residue and the session survives.
  - `tests/test_llm_utils.py` — provider auto-detect precedence, per-provider default models, temperature omission, Gemini `max_output_tokens` mapping.
  - `tests/test_utils_misc.py` — byte-stable system prompt, `SummaryBufferHistory` pruning keeps a human-first history and counts the summary preamble itself against the token budget.
  - `tests/test_debug_chat.py` — debug TUI panels (metrics panel distinguishes empty retrieval from RAG-skipped, survives non-numeric scores, error turns stay visible) and `handle_turn` per-stage timings on success/error/rejection paths.
  - All of these run offline with fakes (`tests/fakes.py`) — no API keys.

## Contribution checklist (for agents)

- Respect content policy: ingest `.md` only; don’t commit corpus or vector stores.
- Prefer minimal, localized patches; keep public signatures stable.
- Update `README.md` and this `Agent.MD` when you alter workflow, flags, or outputs.
- Add tests for new behavior and ensure all tests pass with `python -m pytest -q`.
- Avoid changing dependency sets; use fallbacks and warning filters for compatibility.

Happy building — and keep it simple, grounded, and well-documented.
