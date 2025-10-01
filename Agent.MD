# Agent Guide for `simple-RAG`

Welcome! This guide summarises the current behaviour of the repository so you
can ship focused changes without digging through every file. Instructions apply
to the whole repo (there are no nested `AGENTS.md` files).

## 1. Current pipeline

The numbered scripts under `scripts/` are fully implemented and runnable:

1. `00_ingest.py` – loads markdown from `data/corpus/`, splits by headers, and
   prints a preview of the resulting LangChain `Document` chunks.
2. `01_build_index.py` – embeds the chunks with MiniLM via
   `HuggingFaceEmbeddings`, replaces `data/chroma/`, and prints a build summary
   (doc count, collection name, metadata keys).
3. `02_query.py` – CLI for retrieval QA. It connects to the persisted Chroma
   store and supports three modes: `none` (retrieval only), `pretend` (prompt
   preview + templated answer), and `llm` (OpenAI-compatible chat call).
4. `03_eval.py` – scores saved question/answer/context rows with lexical
   heuristics (`is_faithful`, `should_abstain`) and writes optional reports.
5. `03_quiz.py` / `report.py` – collect human judgements and generate Markdown
   summaries.
6. `04_llm_api.py` – standalone smoke test for LLM access (question + context
   snippets -> formatted prompt -> response + usage metrics).
7. `05_chat_cli.py` – interactive SIMPLE_RAG chat agent that decides when to call RAG, rewrites queries for cosine search, and presents answers in a friendly persona (spinner included).
8. `agent_orchestration_helper.py` – shared module for the decider, query rewriter, topic inventory, and fallback payload builder used by the CLI.

## 2. Data layout

- `data/corpus/` and `data/chroma/` are committed as empty directories (only
  `.gitkeep` placeholders). Feel free to drop your own corpus files locally; they
  are ignored by git.
- `01_build_index.py` always recreates `data/chroma/`. Delete the folder to force
  a clean rebuild if you change embeddings.

## 3. Environment notes

- Use `pip install -r requirements-min.txt` for a lean setup, or `requirements.txt`
  if you want every optional dependency.
- Copy `.env.example` to `.env` and set `OPENAI_API_KEY` for `02_query.py --agent-mode llm`
  or `RAG_LLM_API_KEY` for `04_llm_api.py`.
- No automated tests exist. Run the CLI scripts directly to smoke test changes.

## 4. Contribution guidelines

- Preserve the educational tone of docstrings and comments when editing code.
- Update documentation (including this file and `README.md`) when workflow steps
  change.
- Do not commit large corpora or vector stores; keep `data/corpus/` and
  `data/chroma/` empty in git.
- Follow standard Python style (PEP 8); avoid wrapping imports in `try/except`
  unless they are optional dependencies.

Happy building!
