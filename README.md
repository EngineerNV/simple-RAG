
# simple-RAG — Learning RAG Project

This repository is a minimal, learning-focused Retrieval-Augmented Generation (RAG) project skeleton. It's designed for experimentation: drop documents into `data/corpus/`, run the scripts in order, and learn how ingestion, embedding, retrieval, and evaluation fit together.

High-level pipeline (scripts run in order):

- `scripts/00_ingest.py` — load source documents and chunk text.
- `scripts/01_build_index.py` — embed chunks and persist vectors into Chroma (or other vector DB).
- `scripts/02_query.py` — embed a user question, retrieve relevant chunks, and (optionally) call a cloud LLM to generate an answer.
- `scripts/03_eval.py` — simple evaluation helpers to measure faithfulness and when to abstain.

Repository layout
-----------------

- `configs/` — YAML configs and prompts. Example: `rag.yaml`, `prompts.yaml`.
- `data/`
  - `corpus/` — drop your documents here (PDFs, text, markdown).
  - `chroma/` — persistent storage used by the vector DB (created by the index step).
- `scripts/` — lightweight, educational scripts for each pipeline step (see above).
- `.env.example` — environment variables used by the project (e.g., `CORPUS_DIR`, API keys).
- `requirements.txt` — recommended Python packages to install for the pipeline.

How the pipeline works (brief)
-----------------------------

1. Ingest (00_ingest.py)
	- Read files from `data/corpus/` (or a directory set by the `CORPUS_DIR` env var).
	- Convert non-text formats (PDF, DOCX) to text if needed.
	- Split long documents into smaller chunks. Chunking reduces the token size and improves retrieval relevance.

2. Build index (01_build_index.py)
	- Take chunks and produce embeddings via an embeddings model (local or cloud).
	- Store embeddings and metadata in a vector DB (Chroma by default) with `data/chroma/` as the persistence directory.

3. Query (02_query.py)
	- Embed user queries and perform a nearest-neighbors search over the vector DB.
	- Collect top-k chunks as context and optionally call a cloud LLM to produce a final answer.
	- This is the RAG step: the model is augmented with retrieved context to ground its outputs.

4. Eval (03_eval.py)
	- Simple heuristics to check if an answer is supported by the retrieved context (faithfulness) and whether the system should abstain.
	- Designed as learning scaffolding — implement and experiment with different checks.

Learning goals and exercises
----------------------------

- Explore chunking strategies (fixed-size, overlap, sentence-aware).
- Try different embedding models and compare retrieval quality.
- Inspect the `data/chroma/` folder after indexing to learn what persistence looks like.
- Implement simple faithfulness checks in `03_eval.py` (keyword overlap, answer verification via LLM, etc.).
- Replace or extend the vector DB (e.g., FAISS, Milvus) for practice.

Quick start
-----------

1. Create a virtual environment and install dependencies: see `requirements.txt`.
2. Drop documents into `data/corpus/`.
3. (Optional) Set `CORPUS_DIR` in your environment to point somewhere else.
4. Run the scripts in order:

	- `python scripts/00_ingest.py`
	- `python scripts/01_build_index.py`
	- `python scripts/02_query.py`
	- `python scripts/03_eval.py`

Notes
-----

- The scripts are intentionally skeletons with learning prompts and TODOs. They don't aim to be production-ready; they're teaching tools.
- When adding secrets or API keys, use `.env` and never commit secrets to the repo.
- Feel free to extend the scripts, add unit tests, or wire up CI for automated checks.

Contributing
------------

If you improve a lesson or add helpful tests/examples, open a PR. Keep edits focused and include short notes about what learners will gain.

License
-------

See `LICENSE`.
