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
3. **Add source material:** drop markdown files into `data/corpus/` (the `corpus/` and `chroma/` directories are created for you).
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
| `llm`     | Call a live chat model (OpenAI-compatible) using the retrieved contexts as evidence.                |

## Environment and API keys

These scripts load environment variables (from your process and from a `.env` file if present) via `python-dotenv`.

**Multi-provider support with auto-detection:** The project supports OpenAI, Google Gemini, and Anthropic Claude. The provider is **automatically detected** based on which API key is set in your environment. You can override this with the `--provider` flag if needed.

**Environment variables (auto-detection order):**

1. **`OPENAI_API_KEY`** — For OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
   - Auto-detected provider: `openai`
   - Models: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4o`, etc.
   - Base URL override: `--base-url` for OpenAI-compatible endpoints

2. **`GOOGLE_API_KEY`** — For Google Gemini models
   - Auto-detected provider: `gemini`
   - Models: `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`, etc.
   - Requires: `pip install langchain-google-genai`

3. **`ANTHROPIC_API_KEY`** — For Anthropic Claude models
   - Auto-detected provider: `claude`
   - Models: `claude-3-opus-20240229`, `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, etc.
   - Requires: `pip install langchain-anthropic`

**How auto-detection works:**

The scripts check for API keys in order (OPENAI → GOOGLE → ANTHROPIC). The first key found determines the provider. If you have multiple keys set and want to use a specific provider, use the `--provider` flag to override.

**Example usage:**

```powershell
# OpenAI (auto-detected from OPENAI_API_KEY)
$env:OPENAI_API_KEY="sk-..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm

# Google Gemini (auto-detected from GOOGLE_API_KEY)
$env:GOOGLE_API_KEY="AIza..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm --llm-model gemini-pro

# Anthropic Claude (auto-detected from ANTHROPIC_API_KEY)
$env:ANTHROPIC_API_KEY="sk-ant-..."
python scripts/02_query.py -q "What is RAG?" --agent-mode llm --llm-model claude-3-5-sonnet-20241022

# Force a specific provider (when multiple keys are set)
$env:OPENAI_API_KEY="sk-..."
$env:GOOGLE_API_KEY="AIza..."
python scripts/05_chat_cli.py --provider gemini --llm-model gemini-1.5-flash

# Explicit API key override
python scripts/02_query.py -q "Test" --agent-mode llm --api-key "sk-..." --provider openai
```

**Retriever/ingest configuration via environment:**

- `CORPUS_DIR` — overrides the input folder for `scripts/00_ingest.py` (default: `data/corpus`). Use this in your `.env`.
- `INGEST_TIKTOKEN_MODEL` — tokenizer model name for token-aware chunking (default: `text-embedding-3-small`).
- `INGEST_CHUNK_SIZE` — approximate chunk size in tokens (default: `400`).
- `INGEST_CHUNK_OVERLAP` — token overlap between adjacent chunks (default: `80`).

Note: The agentic CLI and helper functions reuse the same chat model key; there are no additional secrets required beyond the LLM API key.## Project structure

| Path | Purpose |
|------|---------|
| `scripts/00_ingest.py` | Loads markdown files from `data/corpus/`, splits them into token-sized chunks with overlap, and previews the resulting `Document` objects. |
| `scripts/01_build_index.py` | Embeds the ingested chunks with `HuggingFaceEmbeddings`, rebuilds `data/chroma/`, and prints a build summary. |
| `scripts/02_query.py` | Connects to the persisted Chroma store and exposes the retrieval CLI described above. |
| `scripts/03_eval.py` | Scores saved question/answer/context rows with lexical heuristics and prints aggregate metrics. |
| `scripts/03_quiz.py` | Interactive reviewer loop for collecting human judgements (faithful/abstain/tags). |
| `scripts/04_llm_api.py` | Standalone helper for formatting prompts and calling a chat model with optional context snippets. |
| `scripts/05_chat_cli.py` | SIMPLE_RAG chat experience with RAG decider, query rewriting, persona prompts, and a progress spinner. |
| `agent_orchestration_helper.py` | Shared helpers for the SIMPLE_RAG CLI (topic inventory, structured decider/rewriter, fallback payload builder). |
| `scripts/report.py` | Aggregates quiz results into a Markdown summary. |
| `configs/` | Starter YAML files for prompts and retrieval parameters—update as you extend the project. |
| `data/` | Storage root. `corpus/` holds your source files; `chroma/` stores the persisted vector index. |

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

# 6. Score an evaluation dataset produced from the quiz or custom tooling
python scripts/03_eval.py --in data/eval/sample.json --out reports/sample_eval.json

# 7. Smoke-test your API integration with hand-crafted snippets
python scripts/04_llm_api.py --question "How does retrieval work?" --context "The retriever uses Chroma with MiniLM embeddings."
```

Each CLI includes `--help` for a full list of options, including custom embedding names, output paths, and evaluation controls.

### Inside the SIMPLE_RAG chat CLI

`scripts/05_chat_cli.py` behaves like a cheerful teammate:

- **Decide → Rewrite → Retrieve:** every turn runs a structured decider to see whether the question falls within the archive topics, optionally rewrites the query for cosine search, and fetches supporting snippets. A Rich status spinner keeps the user informed while the agent is “thinking.”
- **Friendly persona:** when contexts exist, SIMPLE_RAG talks about what it just looked up and cites snippets as `[source #]`. If nothing relevant is found but the question is on theme, it gives a short background answer from its own knowledge; truly off-topic prompts are deflected with gentle suggestions that align with the archive.
- **Shared orchestration helpers:** all of the routing logic, inventory text, and fallbacks live in `agent_orchestration_helper.py`, keeping the CLI tidy and making it easy to reuse the same behaviour elsewhere.

## Data directory layout

The repository ships empty placeholders for the directories referenced in the docs:

```text
data/
├── chroma/   # persisted Chroma collections created by 01_build_index.py
├── corpus/   # drop your markdown or text sources here
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

## CLI agent flow: decide → (rewrite) → retrieve → respond

The chat CLI (`scripts/05_chat_cli.py`) adds a small agentic loop that decides when to use RAG, optionally rewrites the query, retrieves contexts, and answers with citations.

ASCII flow:

```text
User Input
   |
   v
[Topic Gate] -- off-topic? --> [Polite Rejection]
   |
   v
[RAG Decider] -- no --> [Direct Answer (no RAG)]
   |
   v
[Query Rewriter] -> rewritten query
   |
   v
[Retriever (Chroma + embeddings)]
   |
   v
[Reranker (blended score)]
   |
   v
[Compose Prompt + LLM]
   |
   v
[Answer + cited contexts]
```

Key components:

- Topic gate (see `utils/topic_gate.py`): classifies whether the request is within scope.
- RAG decider and query rewriter (see `agent_orchestration_helper.py`): structured helpers that determine whether to use retrieval and produce a cleaner search query when helpful.
- Retriever: uses the built vector store (`data/chroma`) with `HuggingFaceEmbeddings`.
- Reranker: combines retriever score and lexical overlap to refine ordering.

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

These small helpers keep prompting, persona, and gating logic tidy and reusable:

- `utils/inventory_view.py` — Deduplicates and formats specialization topics into a readable one-liner.
- `utils/persona.py` — Builds the persona preamble that’s prepended to the user prompt.
- `utils/rejections.py` — Generates brief on-topic refusals (structured output via the LLM) when requests are out of scope.
- `utils/text_sanitize.py` — Optionally removes suggestion-style text from outputs (to keep answers concise).
- `utils/topic_gate.py` — A small classifier that decides if a request is on-topic; can enforce a minimum confidence.

The chat CLI wires these together with the structured decider/rewriter in `agent_orchestration_helper.py`.

## Topic scope and updating inventory (rag_content.json)

You control what the agent “specializes” in via the JSON file next to the code: `rag_content.json`.

- `rag_topic_inventory` — Multi-line description of what the archive covers. Used by the decider/rewriter for context.
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

- Gate off-topic requests (via `utils/topic_gate.py`).
- Prefer retrieval for questions that overlap the inventory (via the structured decider in `agent_orchestration_helper.py`).
- Phrase the persona and refusals using your `specialization_topics`.

## Content format and ingestion

To keep things simple, ingestion assumes Markdown files for now.

- Place `.md` files in `data/corpus/`. Hidden files (starting with `.`) are ignored.
- Files are split into token-bounded chunks using a tiktoken-aware splitter with overlap.
- For very small files, the ingester may also use H2 headings as a minimal fallback to create multiple chunks.

If you provide non-Markdown files, behavior is undefined; normalize content to `.md` while the project stays simple.
