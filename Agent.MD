# Agent Guide for `simple-RAG`

This document orients automation agents and contributors to the layout, intent, and current state of the repository. Use it as a quick reference before making changes.

## 1. Purpose & High-Level Flow
- **Goal:** Provide a minimal Retrieval-Augmented Generation (RAG) learning sandbox.
- **Pipeline order:**
  1. `scripts/00_ingest.py` – loads markdown-like documents, splits them into header-based sections, and returns `Document` objects from LangChain.
  2. `scripts/01_build_index.py` – scaffold only. Intended to embed chunks (e.g., Hugging Face or OpenAI) and persist them to Chroma under `data/chroma/`.
  3. `scripts/02_query.py` – scaffold only. Should embed a user query, retrieve similar chunks from Chroma, and optionally call an LLM.
  4. `scripts/03_eval.py` – scaffold only. Should evaluate answer faithfulness/abstention heuristics.

Only the ingest step is implemented; later steps are intentionally left as exercises with detailed guidance comments.

## 2. Repository Structure Overview
| Path | Role |
| ---- | ---- |
| `configs/` | YAML templates for RAG settings (`rag.yaml`) and prompt engineering (`prompts.yaml`). Designed for experimentation; values are placeholders. |
| `data/` | Storage root. Expect `corpus/` for raw docs and `chroma/` for persisted vector DB. Currently contains only `README.md`. |
| `scripts/` | Sequential pipeline scripts (`00`–`03`). Later stages still contain TODOs scaffolding. |
| `requirements.txt` | Suggested Python dependencies (LangChain, ChromaDB, transformers, OpenAI/Anthropic, etc.). |
| `README.md` | Learner-facing overview mirroring much of this guide. |

There are no tests or CI configured yet.

## 3. Environment & Setup Notes
- Python project; create a virtual environment and install with `pip install -r requirements.txt`.
- `.env` handling is implied (see `.env.example` reference in README); ensure secrets such as API keys remain out of version control.
- LangChain document objects are used in `00_ingest.py`; later steps should reuse the same embeddings model to avoid vector mismatch.

## 4. Data Handling Expectations
- Place source documents in `data/corpus/` before running ingestion. Markdown is the only format handled right now; extend the ingestion script if other formats are needed.
- Chroma persistence path should stay `data/chroma/`. Remove the folder to rebuild from scratch when experimenting.

## 5. Working with the Scripts
- `00_ingest.py` is executable as-is (prints preview of first few sections). It relies on the `langchain_text_splitters` Markdown splitter and reads all non-hidden files within the corpus directory.
- `01_build_index.py` onwards are stubs with explicit implementation checklists embedded as docstrings. Follow those notes when adding functionality.
- None of the scaffolded scripts currently persist intermediate outputs to disk; plan for your own serialization if you need later steps to consume saved data.

## 6. Suggested Development Workflow
1. Populate `data/corpus/` with markdown files.
2. Run `python scripts/00_ingest.py` to validate splitting.
3. Implement embedding/index logic in `01_build_index.py` (recommended: HuggingFace embeddings + LangChain Chroma wrapper) and persist to `data/chroma/`.
4. Implement query and evaluation scripts iteratively, keeping comments that explain learning takeaways.
5. Document any new configuration keys inside the YAML files so other agents understand expected parameters.

## 7. Quality & Testing Guidance
- No automated tests exist. If you add substantive functionality, consider lightweight CLI smoke tests or unit tests.
- When modifying scripts, ensure they remain runnable as standalone entry points (`python scripts/<name>.py`).
- Follow standard Python formatting (PEP 8) and avoid wrapping imports in try/except unless truly necessary.

## 8. Contribution Etiquette for Agents
- Keep commits focused and descriptive; align with the educational tone of existing comments.
- Update this `Agent.MD` if you significantly change project structure or workflows.
- Respect repository secrets hygiene and do not commit generated vector data or large corpora unless required.

Happy building!
