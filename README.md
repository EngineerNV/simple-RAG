# simple-RAG — Retrieval-Augmented Generation Lab

`simple-RAG` is a teaching project that ships a minimal but runnable retrieval pipeline built with LangChain and Chroma. The repository now includes working ingestion, indexing, retrieval, evaluation, and LLM helper scripts so you can focus on experimenting rather than scaffolding.

## Quick start

1. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-min.txt  # or requirements.txt for the full stack
   ```
2. **Configure secrets:** copy `.env.example` to `.env` and populate API keys (`OPENAI_API_KEY` for `02_query.py`, `RAG_LLM_API_KEY` for `04_llm_api.py`).
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

## Project structure

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
python scripts/04_llm_api.py --question "How does retrieval work?" \
  --context "The retriever uses Chroma with MiniLM embeddings."
```

Each CLI includes `--help` for a full list of options, including custom embedding names, output paths, and evaluation controls.

### Inside the SIMPLE_RAG chat CLI

`scripts/05_chat_cli.py` now behaves like a cheerful teammate rather than a bare pipeline demo:

- **Decide → Rewrite → Retrieve:** every turn runs a structured decider to see whether the question falls within the archive topics, optionally rewrites the query for cosine search, and fetches supporting snippets. A Rich status spinner keeps the user informed while the agent is “thinking.”
- **Friendly persona:** when contexts exist, SIMPLE_RAG talks about what it just looked up and cites snippets as `[source #]`. If nothing relevant is found but the question is on theme, it gives a short background answer from its own knowledge; truly off-topic prompts are deflected with gentle suggestions that align with the archive.
- **Shared orchestration helpers:** all of the routing logic, inventory text, and fallbacks live in `agent_orchestration_helper.py`, keeping the CLI tidy and making it easy to reuse the same behaviour elsewhere.

## Data directory layout

The repository ships empty placeholders for the directories referenced in the docs:

```
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
