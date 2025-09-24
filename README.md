# simple-RAG — Retrieval-Augmented Generation Lab

This workspace supplies a focused homework-style assignment on building a LangChain-powered RAG agent. You receive a working ingestion baseline and progressively design the remaining stages: embedding to Chroma, retrieval + agent routing, LLM prompting, and lightweight evaluation. Each script keeps the same numbering convention (`00`–`04`) so you can run them directly while you implement.

## Learning objectives
- Translate raw markdown documents into LangChain `Document` objects that downstream tooling can consume.
- Persist embeddings to a local Chroma collection and reuse the same model for retrieval.
- Orchestrate a LangChain agent that calls into the Chroma retriever and a chat model to answer questions.
- Capture quick evaluation signals (faithfulness and abstention) to understand when the pipeline is trustworthy.

## Assignment setup
1. Create and activate a Python virtual environment.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and populate provider keys. The LLM hook expects `RAG_LLM_API_KEY` but you may rename it.
4. Place one or more markdown files in `data/corpus/`.
5. Run `python scripts/00_ingest.py` once to verify the corpus is discovered and chunked.

## Baseline provided
- **`scripts/00_ingest.py`** – functional ingestion utility that splits markdown documents into header-aware LangChain `Document` objects.
- **`scripts/01_build_index.py`** – checklist for embedding and persisting chunks with Chroma.
- **`scripts/02_query.py`** – checklist for composing a LangChain agent that talks to the vector store and an LLM.
- **`scripts/03_eval.py`** – checklist for evaluating groundedness and abstention heuristics.
- **`scripts/04_llm_api.py`** – helper for formatting prompts and executing raw LLM calls.
- **`configs/`** – placeholder YAML settings you can extend.
- **`requirements.txt`** – dependency list aligned with the LangChain + Chroma plan.
- **`data/`** – staging area for source corpus and persisted vector stores.

## Milestones

### Milestone 0 – Inspect the ingestion baseline (`scripts/00_ingest.py`)
- Execute `python scripts/00_ingest.py` and note how documents are chunked and which metadata fields are preserved.
- Decide whether additional cleaning is required (stopword removal, metadata enrichment, etc.). Keep the public `ingest()` contract stable so other steps import it.

### Milestone 1 – Build the vector index (`scripts/01_build_index.py`)
Teacher expectations:
- Import `ingest` from `scripts` (re-exported from `00_ingest`) to obtain the `Document` list.
- Instantiate an embedding model from LangChain (for example `HuggingFaceEmbeddings` or `OpenAIEmbeddings`) and use it consistently across retrieval.
- Create or connect to a persisted Chroma collection in `data/chroma/` using `langchain_community.vectorstores.Chroma`.
- Add the documents, confirm they are written to disk, and print a short summary (chunk count, collection name, persistence directory).
- Provide an optional CLI flag (e.g., `--force`) to rebuild the store from scratch.

### Milestone 2 – Wire a retrieval-aware agent (`scripts/02_query.py`)
Teacher expectations:
- Load the same embedding model and wrap the Chroma collection as a retriever.
- Define a LangChain `Tool` that exposes the retriever and register it with an agent (ReAct, ConversationalRetrievalChain, or RetrievalQA chain wrapped as a tool).
- Collect a user question via CLI or function arguments, invoke the agent, and display both the retrieved context snippets and the final answer/abstain message.
- Allow configuration of retrieval depth (`k`) and model choice via CLI arguments or environment variables.

### Milestone 3 – Shape the LLM call (`scripts/04_llm_api.py`)
Teacher expectations:
- Load API credentials from the environment, instantiate your chosen chat/completions client, and keep client creation in a helper.
- Build a prompt template that threads system guidance, the question, and retrieved context chunks.
- Execute the LLM call, capture metadata (model, latency, token usage), and expose a CLI hook for manual smoke tests.

### Milestone 4 – Evaluate groundedness (`scripts/03_eval.py`)
Teacher expectations:
- Curate a small evaluation file (JSON or CSV) with question/answer/context entries produced by your agent.
- Implement `is_faithful` and `should_abstain` heuristics (keyword overlap, citation presence, or LLM-based checks) and summarise pass/fail counts.
- Optionally persist detailed reports for later iteration.

## Deliverables
- Implemented scripts for milestones 1–4 that are runnable via `python scripts/<step>.py`.
- A transcript demonstrating the agent retrieving context and answering (or abstaining) for at least one query.
- A smoke test of the standalone LLM helper that proves the API integration works (or gracefully abstains without a key).
- An evaluation summary that highlights faithful vs. unfaithful answers and abstention decisions.
- Reflection notes on next steps you would explore (model swaps, richer tools, UI ideas, etc.).

## Working notes
- Re-run ingestion whenever the corpus changes; rebuild embeddings if you swap models.
- Keep secrets in `.env` and never commit the file.
- Remove `data/chroma/` to force a clean index rebuild during experiments.
- Use `configs/` to capture tunable parameters once you move beyond the defaults.

