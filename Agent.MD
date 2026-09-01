# Agent Guide for `simple-RAG`

This guide orients automation agents and contributors so you can ship focused, safe changes without spelunking the entire tree. It complements the top-level `README.md` with contributor-facing details, coding conventions, and integration points.

## Quick setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-min.txt    # or requirements.txt for full stack

# Optional: test tooling
pip install -r requirements-dev.txt    # provides pytest

# Optional: RAGAS evaluation (scripts/07_ragas_eval.py)
pip install -r requirements-eval.txt

# Build and smoke-test
python scripts/00_ingest.py
python scripts/01_build_index.py
python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend
```

**The `langchain` family in `requirements.txt` is pinned, not loose.** `chat_engine.py` uses the legacy `langchain.memory.ConversationSummaryBufferMemory`, which was removed in `langchain>=1.0` — an unpinned install silently resolves to a version that breaks the chat CLI/TUI at import time. Bump the pinned set together (see the comment above it in `requirements.txt`), never one package at a time, and re-run `python -m pytest -q` plus a real chat session after.

Environment variables & multi-model flexibility:

- **Auto-detected multi-provider support**: Set any one of these keys and the scripts will automatically detect the provider:
  - `OPENAI_API_KEY` — for OpenAI models (checked first; e.g. `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`)
  - `GOOGLE_API_KEY` — for Google Gemini models (checked second; e.g. `gemini-1.5-flash`, `gemini-1.5-pro`)
  - `ANTHROPIC_API_KEY` — for Anthropic Claude models (checked third; e.g. `claude-3-5-sonnet-20241022`, `claude-haiku-4-5`)
- **Flexible Model Selection**: Any script running LLM calls (`02_query.py`, `04_llm_api.py`, `05_chat_cli.py`, `06_chat_tui.py`, `07_ragas_eval.py`) supports selecting a target model via `--llm-model <model-name>` and overriding the provider via `--provider openai|gemini|claude`.
- Install provider dependencies as needed:
  - `pip install langchain-openai` (default, usually already present)
  - `pip install langchain-google-genai` (for Gemini)
  - `pip install langchain-anthropic` (for Claude)
- **Note:** Only ONE API key is needed for basic operation, but you can configure multiple and switch dynamically using `--provider` and `--llm-model`.

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

7) `scripts/05_chat_cli.py` — Rich-based terminal chat REPL. Thin rendering layer over `chat_engine.ChatEngine`; shows contexts when `--show-context`.

8) `scripts/06_chat_tui.py` — full-screen Textual TUI for the same chat pipeline: scrollable chat log, input box, and a live context sidebar (also built on `chat_engine.ChatEngine`, no pipeline logic duplicated). `ctrl+r` resets, `ctrl+s` saves the transcript, `/exit` quits.

9) `scripts/07_ragas_eval.py` — scores the pipeline against `data/eval/golden_qa.json` with [RAGAS](https://github.com/explodinggremlins/ragas) (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall). Requires `requirements-eval.txt` and an API key for whichever provider is set as the judge (`--judge-provider`/`--judge-model`, defaults to the same provider/model as the pipeline under test). See "RAGAS evaluation" below.

10) `scripts/08_eval_tui.py` — full-screen Textual TUI evaluation dashboard allowing users to configure arguments and trigger Human Quiz, Lexical Heuristic Eval, and RAGAS LLM-Judge Eval workflows with real-time log output streaming.

Agent orchestration and utils:

- `chat_engine.py` — UI-agnostic core of the chat pipeline (`ChatEngine`, `ChatEngineConfig`, `TurnResult`, `TranscriptLog`): topic gate → RAG decider → optional rewrite → retrieve → rerank → compose → LLM → history/transcript commit. Shared by `05_chat_cli.py`, `06_chat_tui.py`, and `07_ragas_eval.py` — implement pipeline changes here once, not per-UI.
- `agent_orchestration_helper.py` — loads topic inventory from `rag_content.json`; builds structured decider/rewriter; assembles user payloads (with persona) for the LLM.
- `utils/llm_provider.py` — shared provider auto-detection (`resolve_provider_and_key`) and chat-client construction (`build_chat_model`) for openai/gemini/claude, used by every script that talks to an LLM.
- `utils/inventory_view.py` — formats specialization topics.
- `utils/persona.py` — persona preamble builder.
- `utils/rejections.py` — structured off-topic refusals.
- `utils/text_sanitize.py` — suggestion filtering.
- `utils/topic_gate.py` — on-topic classifier with a confidence floor.

Data/config:

- `rag_content.json` — JSON controlling the agent’s scope (topics/inventory).
- `data/corpus/` — place `.md` files here (hidden files ignored). See the worked heading/metadata example just below before adding content.
- `data/chroma/` — auto-created vector index storage.
- `data/eval/` — `golden_qa.json` (golden question/ground-truth-answer/reference-context set) plus a `README.md` walking through the schema and what each RAGAS metric measures. See "RAGAS evaluation" below.

## Topic scope and content policy

Scope is defined in `rag_content.json`:

- `rag_topic_inventory` (multi-line description) guides the decider/rewriter.
- `specialization_topics` (list) appear in the persona and refusals.

Content format: keep it simple — only Markdown (`.md`) files are supported for ingestion today. The ingester splits at heading boundaries first (every chunk carries its `#`/`##`/`###`/`####` heading path as metadata), then re-splits any section still over `INGEST_CHUNK_SIZE` tokens with a tiktoken-aware splitter, preserving overlap. Headings therefore double as the retrieval-time section-title metadata — keep entities (a species, a gym leader, a case study) each under their own heading rather than bundled as bullets under a shared one, or their chunks won't get a distinct title.

**Worked example, so this isn't abstract.** Given this markdown:

```markdown
## 4. Expanded Species Research

### Poison & Toxin Specialists

#### Grimer → Muk

* **Typing**: Poison
* **Base Stats** (Muk): HP 105 / Atk 105 / ...
```

`00_ingest.py` (via `MarkdownHeaderTextSplitter`, `strip_headers=True`) turns the `Grimer → Muk` section into one chunk whose `page_content` is the bullet list *with the heading re-prepended* (see the "why re-prepend" note in `00_ingest.py::split_markdown`'s docstring) and whose metadata is:

```python
{"##": "4. Expanded Species Research", "###": "Poison & Toxin Specialists", "####": "Grimer → Muk", ...}
```

Now contrast a *bad* structure for the same content — bundling several species as bullets under one heading with no per-entity heading of their own:

```markdown
### Poison Types

* **Grimer → Muk**: Typing: Poison. Base Stats...
* **Koffing → Weezing**: Typing: Poison. Base Stats...
```

This produces **one chunk covering both species**, tagged only `{"###": "Poison Types"}` — a retrieval hit for "Muk" surfaces a chunk that's half about Weezing too, and there's no metadata distinguishing which species is which. This is exactly the bug a Codex review caught in this repo (PR #25): converting the bullet-per-entity sections to heading-per-entity fixed both the missing metadata *and*, once headings are stripped from `page_content`, a follow-up bug where the entity's own name disappeared from the embeddable/searchable text entirely (fixed by re-prepending the heading — see the docstring). **The rule for new corpus content: one heading per thing you want to be independently retrievable and citable.**

## Reranker & Embedding Flexibility

- **Embedding & Reranker Choice**: `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2` with lexical blend fallback) are reference choices chosen for local, zero-cost, CPU-compatible execution. They serve as concrete examples of dense retrieval and cross-encoder reranking.
- **Pluggability**: The pipeline architecture is intentionally decoupled:
  - Embeddings in `01_build_index.py` and `chat_engine.py` can be swapped for OpenAI, Google Gemini, Cohere, or custom embedding classes.
  - Reranking in `scripts/02_query.py` (`rerank_results`) can be replaced with commercial reranking APIs (Cohere, Jina, Voyage AI) or alternative local models (FlashRank, BGE-Reranker) without modifying `chat_engine.py` or the UI frontends (`05_chat_cli.py`, `06_chat_tui.py`).
- Location: `scripts/02_query.py` → `rerank_results(results, question, alpha=0.5, use_cross_encoder=True)`.
- Default method: a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`, lazily loaded once at module level) scores each `(question, doc)` pair jointly and writes a sigmoid-normalized `combined_score`/`cross_encoder_score` to `doc.metadata`. Catches semantic relevance (synonyms, paraphrase, word order) the lexical blend can't.
- Fallback method: if the cross-encoder can't load (missing dependency, offline environment) or `use_cross_encoder=False` is passed, falls back to the original lightweight lexical blend — min–max normalize the retriever score, compute lexical overlap via regex tokenization, blend by `combined = alpha * retriever_norm + (1-alpha) * lexical`. Tests force this path (`use_cross_encoder=False`) to avoid depending on a downloaded ML model.
- Displays: `emit_contexts`, `compose_user_prompt`, and the chat CLI/TUI's context views show `score` and `rerank` when present.
- Robustness: failure at any stage is non-fatal; the script falls back to the original retriever ordering.

## Chat pipeline (topic gate → decider → retrieve → rerank → compose → LLM)

Flow: topic gate → RAG decider → (optional) rewrite → retrieve → rerank → compose → LLM → answer, implemented once in `chat_engine.ChatEngine.process_turn` and shared by every chat surface (`05_chat_cli.py`, `06_chat_tui.py`, `07_ragas_eval.py`). The structured helpers are in `agent_orchestration_helper.py`; the topic gate lives in `utils/topic_gate.py`. Implement pipeline changes in `chat_engine.py`, not in a UI file — a UI file should only ever construct a `ChatEngine`, call `process_turn`/`reset`/`save_transcript`, and render the returned `TurnResult`.

Low-confidence fail-safes are intentionally asymmetric: `topic_gate` (`utils/topic_gate.py`) fails **open** (`on_topic=True`) below its confidence floor, so an uncertain classification doesn't wrongly refuse a legitimate question; `decide_use_rag` (`agent_orchestration_helper.py`) fails **closed** (`use_rag=False`), so an uncertain classification doesn't inject irrelevant retrieved context. Both defaults favor answering over refusing/injecting noise — keep this in mind before "fixing" one to match the other.

Useful flags:

- `scripts/05_chat_cli.py --show-context --debug`
- `scripts/06_chat_tui.py --save-transcript out.md`
- `scripts/02_query.py -q "..." --agent-mode pretend|llm --k 5`

## RAGAS evaluation

- `scripts/07_ragas_eval.py` runs every question in `data/eval/golden_qa.json` (schema: `question`, `ground_truth`, `reference_contexts`) through the *real* pipeline via a single reused `ChatEngine` (calling `engine.reset()` before each question so questions stay independent, not a multi-turn conversation), then scores the batch with RAGAS's Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall.
- The judge model is independently configurable (`--judge-provider`/`--judge-model`/`--judge-api-key`), defaulting to whatever provider/model the pipeline under test uses. A cheap, fast, instruction-following model is recommended for the judge (e.g. `claude-haiku-4-5`) since RAGAS issues several judge calls per metric per question — a 15-question run is on the order of a hundred+ judge calls.
- Requires `requirements-eval.txt` (adds `ragas` and its transitive deps, pinned — see the comment at the top of that file for why unpinned installs break either `ragas` or the base `langchain` pins).
- Output: an aggregate + per-question console summary, plus a full JSON report (`--out`, default `data/eval/ragas_report.json`).
- Extending the golden set: keep `reference_contexts` as verbatim excerpts actually present in the corpus — RAGAS's context metrics score against them directly, so a paraphrased or stale reference silently under/over-states retrieval quality.

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
