# simple-RAG — Retrieval-Augmented Generation Lab

`simple-RAG` is a project that ships a minimal but runnable retrieval pipeline built with LangChain and Chroma. The repository now includes working ingestion, indexing, retrieval, evaluation, and LLM helper scripts so you can focus on experimenting rather than scaffolding.

[Watch Demo Video](https://drive.google.com/file/d/1xM5Y6-JccKDqi6ozH2pny2fycPhVP2VO/view?usp=sharing)

[Watch High Level Summary of Repository](https://drive.google.com/file/d/1InwxNeIiRv4fUCJE-T4WES4uxoB3AuGK/view?usp=sharing)

## Quick start

1. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-min.txt  # or requirements.txt for the full stack
   ```
2. **Configure secrets:** Set the appropriate API key environment variable for your chosen LLM provider (see Environment section below). Alternatively, copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (or `GOOGLE_API_KEY`/`ANTHROPIC_API_KEY`).
3. **Add source material:** drop markdown files into `data/corpus/` (`data/chroma/` is created automatically when you build the index).
   - A sample knowledge base, `Pokémon.MD`, is included so you can immediately test ingestion and retrieval behaviour.
4. **Run the pipeline:**
   ```bash
   python scripts/00_ingest.py            # inspect chunking
   python scripts/01_build_index.py       # embed & persist to data/chroma/
   python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend
   ```

The query script offers three modes:

| Mode      | Description                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------|
| `none`    | Retrieve contexts and print a stitched answer using retrieved text only (no LLM call).              |
| `pretend` | Preview the system prompt, retrieved snippets, and a templated final answer with citations.         |
| `llm`     | Call a live chat model (OpenAI, Gemini, or Claude) using the retrieved contexts as evidence.         |

## Environment and API keys

These scripts load environment variables (from your process and from a `.env` file if present) via `python-dotenv`.

**Multi-provider support with auto-detection:** The project supports OpenAI, Google Gemini, and Anthropic Claude. The provider is **automatically detected** based on which API key is set in your environment. You can override this with the `--provider` flag if needed.

**Environment variables (auto-detection order):**

1. **`OPENAI_API_KEY`** — For OpenAI models
   - Auto-detected provider: `openai` (default model: `gpt-5-mini`)
   - Base URL override: `--base-url` for OpenAI-compatible endpoints

2. **`GOOGLE_API_KEY`** — For Google Gemini models
   - Auto-detected provider: `gemini` (default model: `gemini-2.5-flash`)
   - Requires: `pip install langchain-google-genai`

3. **`ANTHROPIC_API_KEY`** — For Anthropic Claude models
   - Auto-detected provider: `claude` (default model: `claude-sonnet-4-5`)
   - Requires: `pip install langchain-anthropic`

**How auto-detection works:**

The scripts check for API keys in order (OPENAI → GOOGLE → ANTHROPIC). The first key found determines the provider. If you have multiple keys set and want to use a specific provider, use the `--provider` flag to override — the key for that provider's own environment variable is used.

**Default models:** when `--llm-model` is not given, each provider falls back to its own default (`utils/llm.py:DEFAULT_MODELS`): `gpt-5-mini` for OpenAI, `gemini-2.5-flash` for Gemini, and `claude-sonnet-4-5` for Claude. This means a Claude-only or Gemini-only environment works out of the box without passing a model name.

**Example usage:**

```powershell
# OpenAI (auto-detected from OPENAI_API_KEY)
$env:OPENAI_API_KEY="sk-..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm

# Google Gemini (auto-detected from GOOGLE_API_KEY; uses gemini-2.5-flash by default)
$env:GOOGLE_API_KEY="AIza..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm

# Anthropic Claude (auto-detected from ANTHROPIC_API_KEY; uses claude-sonnet-4-5 by default)
$env:ANTHROPIC_API_KEY="sk-ant-..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm

# Force a specific provider (when multiple keys are set)
$env:OPENAI_API_KEY="sk-..."
$env:GOOGLE_API_KEY="AIza..."
python scripts/05_chat_cli.py --provider gemini

# Explicit API key override
python scripts/02_query.py -q "Test" --agent-mode llm --api-key "sk-..." --provider openai
```

**Retriever/ingest configuration via environment:**

- `CORPUS_DIR` — overrides the input folder for `scripts/00_ingest.py` (default: `data/corpus`). Use this in your `.env`.
- `CHROMA_DB_PATH` — overrides where the Chroma index is stored and read (default: `data/chroma`).
- `INGEST_TIKTOKEN_MODEL` — tokenizer model name for token-aware chunking (default: `text-embedding-3-small`).
- `INGEST_CHUNK_SIZE` — approximate chunk size in tokens (default: `400`).
- `INGEST_CHUNK_OVERLAP` — token overlap between adjacent chunks (default: `80`).

The full configuration surface is: these `.env` variables, per-script CLI flags (`--help` on any script), and `rag_content.json` for the chat agent's scope. The YAML files under `configs/` are unused learning templates (see the header comment in each).

Note: The agentic CLI and helper functions reuse the same chat model key; there are no additional secrets required beyond the LLM API key.

## Project structure

| Path | Purpose |
|------|---------|
| `scripts/00_ingest.py` | Loads markdown files from `data/corpus/`, splits them into token-sized chunks with overlap, and previews the resulting `Document` objects. |
| `scripts/01_build_index.py` | Embeds the ingested chunks with `HuggingFaceEmbeddings`, rebuilds `data/chroma/`, and prints a build summary. |
| `scripts/02_query.py` | Connects to the persisted Chroma store and exposes the retrieval CLI described above. |
| `scripts/03_eval.py` | Scores saved question/answer/context rows with lexical heuristics and prints aggregate metrics. |
| `scripts/06_quiz.py` | Interactive reviewer loop for collecting human judgements (faithful/abstain/tags). |
| `scripts/04_llm_api.py` | Standalone helper for formatting prompts and calling a chat model with optional context snippets. |
| `scripts/05_chat_cli.py` | SIMPLE_RAG chat experience: one structured router call per turn, grounded prompting, persona system prompt, and a progress spinner. |
| `scripts/07_debug_chat.py` | Debug chat TUI: same session as the chat CLI, plus a live metrics pane showing router decisions, per-chunk retrieval scores, per-stage timing, and memory stats. |
| `agent_orchestration_helper.py` | Chat orchestration: the router (scope + retrieval + query rewrite in one call), the system prompt/evidence contract, and the testable `ChatSession` turn handler. |
| `scripts/report.py` | Aggregates quiz results into a Markdown summary. |
| `utils/` | Shared modules: `llm.py` (providers/models/factory), `settings.py`, `textproc.py`, `chat_history.py`, persona/rejection helpers. |
| `configs/` | **Unused YAML templates** kept as learning scaffolds — nothing loads them; real config is `.env` + CLI flags + `rag_content.json`. |
| `data/` | Storage root. `corpus/` holds your source files; `chroma/` is created by `01_build_index.py` for the persisted vector index. |

## Running the CLI tools

Most scripts are executable with sensible defaults. Highlights:

```bash
# 1. Build the vector store (rebuilds data/chroma/ each run)
python scripts/01_build_index.py

# 2. Ask a question using retrieval only
python scripts/02_query.py -q "What data directory should I use?"

# 3. Preview how a live LLM call would look without hitting the API
python scripts/02_query.py -q "Summarise the ingestion step" --agent-mode pretend --k 5

# 4. Call the real LLM once OPENAI_API_KEY is set
python scripts/02_query.py -q "How do I rebuild the index?" --agent-mode llm --show-usage

# 5. Chat with the SIMPLE_RAG agent (persona + spinner)
python scripts/05_chat_cli.py --debug --show-context

# 5b. Same chat, but with a live pipeline-metrics panel (router / retrieval / timing / memory)
python scripts/07_debug_chat.py --chat-lines 30

# 6. Score an evaluation dataset produced from the quiz or custom tooling
python scripts/03_eval.py --in data/eval/sample.json --out reports/sample_eval.json

# 7. Smoke-test your API integration with hand-crafted snippets
python scripts/04_llm_api.py --question "How does retrieval work?" --context "The retriever uses Chroma with MiniLM embeddings."
```

Each CLI includes `--help` for a full list of options, including custom embedding names, output paths, and evaluation controls.

### Inside the SIMPLE_RAG chat CLI

`scripts/05_chat_cli.py` behaves like a cheerful teammate:

- **One router call per turn:** a single structured LLM call decides whether the request is in scope, whether retrieval would help, and what standalone query to search with (anaphora like "its evolution" resolved from history). This replaced three separate gate/decider/rewriter calls, roughly halving per-turn latency and cost. If the router fails or is unsure, the turn falls back to retrieve-and-abstain rather than passing content through ungated.
- **Grounded prompting:** the persona and grounding rules live in a byte-stable system prompt (friendly to provider prompt caching). Retrieved chunks are delimited in `<documents>` tags and explicitly marked as evidence-not-instructions, with an abstention rule ("I don't have that in my notes") instead of invented facts.
- **Clean memory:** conversation history stores only your actual messages — retrieval dumps never pollute the rolling summary. An LLM error mid-turn shows an apology and keeps the session alive.
- **Shared orchestration helpers:** the routing logic, prompt contract, and turn handling live in `agent_orchestration_helper.py` (`ChatSession`), keeping the CLI a thin rendering shell that's easy to test offline (see `tests/test_chat_session.py`).

## Data directory layout

```text
data/
├── chroma/   # persisted Chroma collections — created by 01_build_index.py on first run
├── corpus/   # drop your markdown or text sources here (a sample Pokémon.MD ships with the repo)
└── README.md
```

You can safely delete `data/chroma/` to force a rebuild or replace the files under `data/corpus/` between experiments. Only commit anonymised or shareable content.

## Dependency notes

- `requirements-min.txt` keeps installation lean for remote LLM usage.
- `requirements-cpu.txt` adds local embedding support without GPU-specific wheels.
- `requirements.txt` includes optional extras for richer experiments.

Large language model tooling downloads sizeable model weights. Clear caches when needed:

```bash
rm -rf ~/.cache/pip ~/.cache/huggingface
```

## Additional resources

- `docs/eval_guide.md` – step-by-step walkthrough of the quiz + report workflow for human evaluation.
- `Agent.MD` – implementation tips for contributors and automation agents.

Happy building!

## Reranking: how results are improved

To improve the accuracy of the top-k retrieval, the query script applies a lightweight reranker that blends the retriever’s numeric score with a lexical-overlap score. This promotes chunks that actually contain the user’s key terms while still respecting the vector similarity ranking.

- Lexical overlap is the fraction of question tokens found in a candidate chunk:

  $\text{lexical}(q, d) = \frac{|\,\text{tokens}(q) \cap \text{tokens}(d)\,|}{|\,\text{tokens}(q)\,|}$

- Retriever scores are min–max normalized to $[0,1]$ across the retrieved set:

  $\text{retriever\_norm}(s) = \begin{cases}
  0 & \text{if } s_{\max} = s_{\min} = 0 \\
  1 & \text{if } s_{\max} = s_{\min} \neq 0 \\
  \dfrac{s - s_{\min}}{s_{\max} - s_{\min}} & \text{otherwise}
  \end{cases}$

- The final rerank score is a convex combination controlled by $\alpha \in [0,1]$:

  $\text{combined} = \alpha \cdot \text{retriever\_norm} + (1-\alpha) \cdot \text{lexical}$

In `scripts/02_query.py`, reranking stores the following in each chunk’s metadata so they are visible in outputs:

- `combined_score` — the blended score used for ordering
- `lexical_overlap` — fraction of question tokens found in the chunk
- `retriever_norm` — normalized retriever score

You’ll see both the original `score` and `rerank` displayed in:

- `02_query.py` printed contexts and the prompt preview
- `05_chat_cli.py` context tables (when `--show-context`)

Note: reranking is best-effort; if anything goes wrong, the script falls back to the original retriever ordering.

## Prompting: what the LLM receives

For `02_query.py --agent-mode llm`, the message payload is:

- System: a concise instruction that enforces citation and “answer only from context.”
- Human: a composed string containing the user question and a list of retrieved contexts with metadata and scores.

The composed prompt looks roughly like:

```text
Question: <your question>

Contexts:
[source 0] score: 0.842 | rerank: 0.771 | metadata: source=..., chunk_index=..., ...
<cleaned snippet>

[source 1] score: 0.536 | rerank: 0.612 | metadata: ...
<cleaned snippet>
```

This keeps the LLM grounded and makes it easy to attribute facts to specific chunks.

## CLI agent flow: route → retrieve → respond

The chat CLI (`scripts/05_chat_cli.py`) runs a small agentic loop with a single routing decision per turn.

ASCII flow:

```text
User Input
   |
   v
[Router (one structured call)]
   |-- off-topic? --------------> [Polite Rejection]
   |-- on-topic, no RAG needed -> [Direct Answer (no retrieval)]
   |
   v  on-topic + RAG (with rewritten search_query)
[Retriever (Chroma + embeddings)]
   |
   v
[<documents> evidence block + byte-stable system prompt + LLM]
   |
   v
[Grounded answer (abstains when evidence is insufficient)]
```

Key components:

- Router (`agent_orchestration_helper.route_turn`): one structured call that classifies scope, decides on retrieval, and rewrites the search query. Low confidence or a router error falls back to retrieve-and-abstain.
- Retriever: uses the built vector store (`data/chroma`) with `HuggingFaceEmbeddings`.
- Prompt contract (`build_system_prompt` / `format_context_documents`): persona + grounding rules in the system layer; evidence delimited as untrusted `<documents>` data in the current turn only.
- Turn handler (`ChatSession.handle_turn`): retrieval failures degrade gracefully, LLM failures keep the session alive, and history stays clean.

## Evaluation: how this project measures quality

No external eval framework (RAGAS, DeepEval, etc.) is used. Evaluation is deliberately layered from cheapest/most-deterministic to most-expensive, and each layer is home-grown and inspectable:

1. **Deterministic lexical heuristics** — `scripts/03_eval.py` scores saved question/answer/context rows offline: an answer is flagged *faithful* when at least 30% of its tokens appear in the retrieved context (`utils/textproc.compute_overlap_ratio`), and *abstain* when the context is too short or too noisy to support any answer. Run it after every chunking/retrieval/prompt change to spot regressions. No LLM calls, no API key needed for `--in` files or `--agent-mode none|pretend`.
2. **Human-in-the-loop review** — `scripts/06_quiz.py` walks a reviewer through live retrieval results, capturing faithful/abstain judgements, failure tags, and notes into JSONL/CSV; `scripts/report.py` aggregates them into a Markdown digest with per-tag remediation advice (the shared mapping lives in `utils/review_tags.py`). See `docs/eval_guide.md` for the full workflow.
3. **Offline behavior tests** — the pytest suite (`tests/`) pins the chat orchestration *contract* with fakes and zero API calls: router fallback behavior, evidence delimiting, clean history, and session survival after LLM failures (`tests/test_chat_session.py` doubles as the no-key chat smoke test).

Deliberately not built yet (measure the baseline first): a labeled 50–100 query retrieval/generation eval set, and LLM-as-judge metrics on top of it. Those become worthwhile once the lexical baseline above is being tracked and shows where the real bottleneck is.

## Testing: run the test suite with pytest

Install the development requirements (does not change core runtime deps):

```powershell
pip install -r requirements-dev.txt
```

Run all tests:

```powershell
python -m pytest -q
```

Run a single file:

```powershell
python -m pytest -q tests/test_query.py
```

If you’re inside the project’s virtual environment, you can run `pytest` directly.

## One-shot setup: run everything

If you just want to ingest, build, and start chatting in one sitting, run:

```powershell
# 1) Ingest corpus into token-bounded chunks
python scripts/00_ingest.py

# 2) Build the Chroma index from those chunks
python scripts/01_build_index.py

# 3) Chat with the RAG agent (shows contexts if you pass --show-context)
python scripts/05_chat_cli.py --show-context
```

You can also try a single-shot query without chat memory:

```powershell
python scripts/02_query.py -q "Who is Blackpink?" --agent-mode pretend
```

## Utilities overview (utils/)

These shared modules keep the pipeline scripts and chat orchestration tidy:

- `utils/llm.py` — Provider auto-detection, per-provider default models, and the single chat-model factory used by every script.
- `utils/settings.py` — Filesystem defaults (`CHROMA_DB_PATH`-aware Chroma directory, embedding model name).
- `utils/textproc.py` — Snippet cleaning, metadata formatting, and lexical-overlap helpers shared by query/eval/quiz.
- `utils/chat_history.py` — `SummaryBufferHistory`: rolling-summary conversation memory with a token budget.
- `utils/inventory_view.py` — Deduplicates and formats specialization topics into a readable one-liner.
- `utils/persona.py` — The persona text used in the chat system prompt.
- `utils/rejections.py` — Generates brief on-topic refusals (structured output via the LLM) when requests are out of scope.
- `utils/warnings_filter.py` — One home for the LangChain deprecation-warning filters.

The chat CLI wires these together with the router and `ChatSession` in `agent_orchestration_helper.py`.

## Topic scope and updating inventory (rag_content.json)

You control what the agent “specializes” in via the JSON file next to the code: `rag_content.json`.

- `rag_topic_inventory` — Multi-line description of what the archive covers. Used by the router when deciding whether to retrieve.
- `specialization_topics` — A list of plain-text topics that are rendered into the persona and shown in refusals.

Editing this file immediately updates what the CLI considers in-scope (no code changes needed). For example:

```json
{
   "rag_topic_inventory": "RAG covers:\n- Blackpink 2016–2023 discography and milestones\n- Awards, tours, chart rankings\n- Collaborations and brand partnerships",
   "specialization_topics": [
      "Blackpink career milestones",
      "Discography and Billboard chart history",
      "World tours and notable performances"
   ]
}
```

The agent will then:

- Gate off-topic requests and prefer retrieval for questions that overlap the inventory (via the router in `agent_orchestration_helper.py`).
- Phrase the persona and refusals using your `specialization_topics`.

## Content format and ingestion

To keep things simple, ingestion assumes Markdown files for now.

- Place `.md` files in `data/corpus/`. Hidden files (starting with `.`) are ignored.
- Files are split into token-bounded chunks using a tiktoken-aware splitter with overlap.
- For very small files, the ingester may also use H2 headings as a minimal fallback to create multiple chunks.

If you provide non-Markdown files, behavior is undefined; normalize content to `.md` while the project stays simple.
