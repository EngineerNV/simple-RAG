# simple-RAG — Retrieval-Augmented Generation Lab

`simple-RAG` is a small, fully runnable Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Chroma. It's meant to be read, not just run — every stage (chunking, embedding, retrieval, reranking, prompting, agent routing, evaluation) is a short, separate script you can inspect and modify independently. This README doubles as a tutorial: skip to [Quick Start](#quick-start) if you just want it running, or start at [Key Concepts](#key-concepts) if you're new to RAG.

📺 [Watch Demo Video](https://drive.google.com/file/d/1xM5Y6-JccKDqi6ozH2pny2fycPhVP2VO/view?usp=sharing) · [Watch High-Level Repository Summary](https://drive.google.com/file/d/1InwxNeIiRv4fUCJE-T4WES4uxoB3AuGK/view?usp=sharing)

---

## Table of contents

1. [Key concepts](#key-concepts) — plain-English definitions for every term used below
2. [Quick start](#quick-start) — macOS/Linux and Windows PowerShell, side by side
3. [The pipeline, step by step](#the-pipeline-step-by-step) — what each stage does and why
4. [The chat agent](#the-chat-agent) — topic gate → decider → retrieve → rerank → respond
5. [Evaluation, two ways](#evaluation-two-ways) — cheap heuristics vs. LLM-judged RAGAS
6. [CLI reference](#cli-reference) — every script and every flag
7. [Environment & provider configuration](#environment--provider-configuration)
8. [Customizing the agent's scope](#customizing-the-agents-scope-rag_contentjson)
9. [Authoring corpus content](#authoring-corpus-content)
10. [Project structure](#project-structure)
11. [Dependency notes](#dependency-notes)
12. [Testing](#testing)
13. [Troubleshooting / FAQ](#troubleshooting--faq)
14. [Additional resources](#additional-resources)

---

## Key concepts

New to RAG? Read this section once; everything else in the README assumes these terms.

| Term | Plain-English definition |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Instead of asking an LLM to answer purely from what it memorized during training, you first *retrieve* relevant text from your own documents, then hand that text to the LLM as evidence and ask it to answer using only that evidence. This reduces hallucination and lets the model answer questions about content it never trained on. |
| **Corpus** | Your source documents — here, the Markdown files in `data/corpus/`. |
| **Chunk** | Documents are too long to hand an LLM whole, so they're split into smaller pieces ("chunks") before indexing. Each chunk becomes one independently-retrievable unit. See [`00_ingest.py`](#00_ingestpy--chunking). |
| **Embedding** | A chunk of text converted into a list of numbers (a vector) such that texts with similar *meaning* end up as nearby vectors. This is what makes "search by meaning" possible instead of just keyword matching. |
| **Vector store** | A database specialized for storing embeddings and finding the nearest ones to a query vector quickly. This project uses [Chroma](https://www.trychroma.com/), persisted to `data/chroma/`. |
| **Retrieval** | Given a question, embed it, then ask the vector store for the *k* chunks whose embeddings are closest (most semantically similar) to the question's embedding. `k` is how many chunks come back — see `--k` / `--retrieval-k` throughout. |
| **Reranking** | Retrieval is fast but approximate. Reranking takes the retrieved candidates and re-scores them more carefully (here, with a cross-encoder — see [§3.4](#4-reranking)) before deciding final order. Retrieval narrows millions of chunks to a handful; reranking picks the best handful. |
| **Cross-encoder** | A model that reads the question *and* a candidate chunk together and outputs one relevance score — slower than embedding-based retrieval but more accurate, so it's only run on the small shortlist retrieval already narrowed down. |
| **LLM provider** | Which company's API answers the question — OpenAI, Google Gemini, or Anthropic Claude are all supported. The project **auto-detects** the provider from whichever API key you've set (see [§7](#environment--provider-configuration)). |
| **System prompt / persona** | The instructions given to the LLM *before* the user's question, establishing its role, tone, and rules (e.g. "only answer from the provided context, cite your sources"). |
| **Topic gate** | A classifier that runs before anything else and decides whether a question is even in-scope for this agent, so it can politely decline unrelated questions instead of guessing. |
| **RAG decider** | A second classifier that decides, for an *on-topic* question, whether retrieval is actually needed (a greeting doesn't need a document search; a factual question probably does). |
| **Query rewriting** | Chat questions are often conversational ("what about its evolution?") and retrieve poorly as-is. The rewriter turns them into a cleaner, self-contained search query before hitting the vector store. |
| **Golden set** | A hand-curated list of questions where a human has already written the *correct* answer and pointed at the *exact* supporting passage. Used to measure pipeline quality objectively instead of eyeballing it. See `data/eval/golden_qa.json`. |
| **Judge model** | In automated evaluation, a second LLM (ideally a different one than the pipeline being tested) reads the question, the pipeline's answer, and the golden answer, and scores the pipeline. "Judging your own homework" with the same model is avoided on purpose. |
| **Faithfulness / Answer Relevancy / Context Precision / Context Recall** | The four RAGAS scores this project reports — defined in full in [§5](#evaluation-two-ways). |

---

## Quick start

Every command below is shown for **macOS/Linux (bash/zsh)** and **Windows (PowerShell)** side by side. Pick your column.

### 1. Create a virtual environment and install dependencies

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

</td><td>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
```

</td></tr>
</table>

> **Why `requirements-cpu.txt` and not `requirements-min.txt`?** Embedding and reranking always run **locally** (via `sentence-transformers`), no matter which LLM provider you choose for chat/generation. `requirements-min.txt` omits `sentence-transformers` to stay lightweight, which means `01_build_index.py` (and the reranker) will fail to import it. Use `requirements-min.txt` only if you're installing `sentence-transformers` some other way, or `requirements.txt` for the full stack (adds the Textual TUI and pinned provider SDKs). See [Dependency notes](#dependency-notes) for the full breakdown.

### 2. Add your API key

Pick **one** of OpenAI, Google Gemini, or Anthropic Claude — the provider is auto-detected from whichever key is set (see [§7](#environment--provider-configuration) for the full detection order and override flag).

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
cp .env.example .env
# then edit .env and set ONE of:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...
#   ANTHROPIC_API_KEY=sk-ant-...
```

</td><td>

```powershell
Copy-Item .env.example .env
# then edit .env and set ONE of:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...
#   ANTHROPIC_API_KEY=sk-ant-...
```

</td></tr>
</table>

A `.env` file works identically on every OS (loaded via `python-dotenv`), so it's the recommended path. If you'd rather set the key for just the current shell session instead:

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
export OPENAI_API_KEY="sk-..."
```

</td><td>

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

</td></tr>
</table>

### 3. Add source material

A sample knowledge base, `data/corpus/Pokémon.MD`, ships with the repo so you can try everything immediately without writing your own content. Drop your own `.md` files into `data/corpus/` when you're ready to replace it (see [Authoring corpus content](#authoring-corpus-content) for the rules that keep retrieval accurate).

### 4. Ingest, index, and ask a question

Identical command on both platforms from here on:

```bash
python scripts/00_ingest.py                                            # preview chunking (no side effects)
python scripts/01_build_index.py                                       # embed & persist to data/chroma/
python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend
```

`--agent-mode pretend` shows you the exact prompt an LLM would receive, with **no API call and no key required** — the fastest way to sanity-check retrieval. Once your key is set, try the real thing:

```bash
python scripts/02_query.py -q "How do I rebuild the index?" --agent-mode llm --show-usage
```

### 5. Chat with the agent

```bash
python scripts/05_chat_cli.py --show-context      # Rich terminal chat, prints context alongside each answer
python scripts/06_chat_tui.py                      # full-screen Textual TUI with a live context sidebar
```

### 6. Run an automated, LLM-judged evaluation

```bash
pip install -r requirements-eval.txt
python scripts/07_ragas_eval.py --golden-set data/eval/golden_qa.json --out data/eval/ragas_report.json
```

To keep the judge model cheap and fast, and independent from the pipeline under test:

```bash
python scripts/07_ragas_eval.py --golden-set data/eval/golden_qa.json --out data/eval/ragas_report.json --provider openai --llm-model gpt-5-mini --judge-provider claude --judge-model claude-haiku-4-5
```

<details>
<summary>Same command, wrapped across multiple lines per shell (click to expand)</summary>

macOS/Linux uses a trailing `\` to continue a line; **PowerShell does not honor `\` as a line-continuation character** — it needs a trailing backtick `` ` `` instead. Pasting a bash-style multi-line command straight into PowerShell is a common source of `'--provider' is not recognized...` errors (see [Troubleshooting](#troubleshooting--faq)).

macOS/Linux:

```bash
python scripts/07_ragas_eval.py \
  --golden-set data/eval/golden_qa.json \
  --out data/eval/ragas_report.json \
  --provider openai \
  --llm-model gpt-5-mini \
  --judge-provider claude \
  --judge-model claude-haiku-4-5
```

Windows (PowerShell):

```powershell
python scripts/07_ragas_eval.py `
  --golden-set data/eval/golden_qa.json `
  --out data/eval/ragas_report.json `
  --provider openai `
  --llm-model gpt-5-mini `
  --judge-provider claude `
  --judge-model claude-haiku-4-5
```

</details>

---

## The pipeline, step by step

Each stage below is one script. Run them in order the first time; after that, only re-run the ones you're actually changing (e.g. re-run `01_build_index.py` after editing the corpus, but not `00_ingest.py`, which is a preview-only dry run).

### 1. Ingestion (`00_ingest.py`)

**What it does:** reads every non-hidden file in `data/corpus/` and splits it into token-bounded chunks, two passes:
1. Split at Markdown heading boundaries (`#`/`##`/`###`/`####`) first, so every chunk carries its enclosing heading(s) as metadata and stays semantically self-contained.
2. Any section still over the token budget gets re-split by a tiktoken-aware splitter, preserving overlap.

**Why headings first:** splitting blindly by token count can slice a sentence — or a whole topic — in half. Splitting at headings first keeps "everything about Pikachu" together as long as it fits, and only falls back to mechanical splitting for oversized sections. See [Authoring corpus content](#authoring-corpus-content) for the rule this implies about how you should structure new files.

**Configuration** is via environment variables, not CLI flags:

| Variable | Default | Meaning |
|---|---|---|
| `CORPUS_DIR` | `data/corpus` | Input folder |
| `INGEST_TIKTOKEN_MODEL` | `text-embedding-3-small` | Tokenizer used to *count* chunk size (doesn't have to match your embedding model) |
| `INGEST_CHUNK_SIZE` | `400` | Target chunk size, in tokens |
| `INGEST_CHUNK_OVERLAP` | `80` | Token overlap between adjacent chunks, so context isn't lost at a chunk boundary |

This script only *previews* chunking (prints the first 3 chunks) — it doesn't write anything to disk. Actual embedding happens in the next step.

### 2. Indexing (`01_build_index.py`)

**What it does:** re-runs ingestion, embeds every chunk with `HuggingFaceEmbeddings` (default model: `sentence-transformers/all-MiniLM-L6-v2`), and **rebuilds `data/chroma/` from scratch** every time it runs (deletes the existing store first). Prints a summary: document count, persist directory, and a preview of chunk metadata keys.

Run this any time you change the corpus or the chunking settings. It has no CLI flags — only the ingest env vars above and a module-level `DEFAULT_MODEL_NAME` constant affect it.

> 💡 **Embedding Choice & Flexibility:** `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` is used as a lightweight, zero-cost default example so the pipeline runs locally out of the box without requiring paid embedding API keys. However, this is purely an example implementation choice. You can easily swap it for cloud provider embeddings (e.g. OpenAI's `text-embedding-3-small` / `text-embedding-3-large`, Google Gemini embeddings, Cohere embeddings) or alternative local embedding models (e.g. `bge-large-en-v1.5`) by modifying `create_retrieval_store` or model configuration in `01_build_index.py`, `chat_engine.py`, and `02_query.py`.

```bash
python scripts/01_build_index.py
```

### 3. Retrieval (`02_query.py`)

**What it does:** embeds your question with the same embedding model used to build the index, asks Chroma for the top `--k` nearest chunks, reranks them (next section), and then does one of three things depending on `--agent-mode`:

| Mode | What happens | Needs an API key? |
|---|---|---|
| `none` | Prints retrieved contexts and a naive concatenated answer. No LLM call at all. | No |
| `pretend` | Shows you the exact system prompt, retrieved snippets, and a templated answer with mock citations — useful for previewing exactly what an LLM would see. | No |
| `llm` | Sends the composed prompt to a real chat model and prints its answer. | Yes |

```bash
python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend --k 5
python scripts/02_query.py -q "How do I rebuild the index?" --agent-mode llm --show-usage
```

### 4. Reranking

Retrieval (previous step) ranks chunks by embedding similarity alone, which is fast but sometimes promotes a chunk that's topically *close* without actually answering the question. Reranking re-scores the retrieved shortlist more carefully before the LLM ever sees it. The **same** `rerank_results()` function (in `02_query.py`) is used everywhere — `02_query.py`, the chat agent, and the RAGAS evaluator all share it via `chat_engine.py`.

- **Primary method — cross-encoder** (used by default): a local `cross-encoder/ms-marco-MiniLM-L-6-v2` model (via `sentence-transformers`, loaded once and cached) reads the `(question, chunk)` pair *together* and produces one relevance score, which catches synonyms, paraphrasing, and word-order effects that pure vector similarity can miss. Stored in metadata as `cross_encoder_score` / `combined_score`.
- **Fallback method — lexical blend**: if the cross-encoder can't load (missing dependency, offline environment) or is explicitly disabled, reranking falls back to a lightweight blend of the retriever's own score and lexical (keyword) overlap:

  $\text{lexical}(q, d) = \dfrac{|\,\text{tokens}(q) \cap \text{tokens}(d)\,|}{|\,\text{tokens}(q)\,|} \qquad \text{combined} = \alpha \cdot \text{retriever\_norm} + (1-\alpha) \cdot \text{lexical}$

  where the retriever's raw score is min–max normalized to $[0,1]$ first, and $\alpha$ (default `0.5`, not currently exposed as a CLI flag) controls the blend. Stored in metadata as `combined_score` / `lexical_overlap` / `retriever_norm`.
- **Failure handling:** reranking is wrapped in try/except everywhere it's called — any failure (model download issue, offline, etc.) silently falls back to the original retriever ordering rather than crashing.

You'll see both the original `score` and the post-rerank `rerank` value in `02_query.py`'s printed output and in the chat CLI/TUI's context views.

> 💡 **Reranking Choice & Flexibility:** The cross-encoder model (`ms-marco-MiniLM-L-6-v2`) and lexical blend fallback were chosen as a practical, self-contained reference implementation that runs locally without API subscriptions. Because reranking logic is decoupled inside `rerank_results()` in `02_query.py`, you can easily replace or extend this step with commercial reranking APIs (e.g. Cohere Rerank, Jina Rerank, Voyage AI) or alternative open-weight models (e.g. FlashRank, BGE-Reranker) without needing to alter any downstream agent or UI components.

### 5. Prompting — what the LLM actually receives

For `--agent-mode llm`, the message payload sent to the model is:

- **System message:** a concise instruction enforcing "answer only from the provided context" and citation of sources.
- **Human message:** the user's question plus the retrieved (and reranked) contexts, each annotated with its scores and metadata.

```text
Question: <your question>

Contexts:
[source 0] score: 0.842 | rerank: 0.771 | metadata: source=..., chunk_index=..., ...
<cleaned snippet>

[source 1] score: 0.536 | rerank: 0.612 | metadata: ...
<cleaned snippet>
```

Keeping scores and metadata visible in the prompt (and in the CLI output) makes it easy to trace any claim in the answer back to a specific chunk.

---

## The chat agent

`scripts/05_chat_cli.py` (Rich terminal REPL) and `scripts/06_chat_tui.py` (full-screen Textual TUI with a live context sidebar) are two different front ends over the **same** shared pipeline logic, implemented once in `chat_engine.py` as `ChatEngine.process_turn()`. `scripts/07_ragas_eval.py` also drives this exact same `ChatEngine`, so evaluation scores reflect the real chat pipeline, not a simplified stand-in.

```text
User Input
   |
   v
[Topic Gate] -- off-topic? --> [Polite Rejection]
   |
   v
[RAG Decider] -- no --> [Direct Answer (no retrieval)]
   |
   v
[Query Rewriter] -> rewritten, self-contained search query
   |
   v
[Retriever (Chroma + embeddings)]
   |
   v
[Reranker (cross-encoder, or lexical blend fallback)]
   |
   v
[Compose Prompt + LLM]
   |
   v
[Answer + cited contexts]
```

- **Topic gate** (`utils/topic_gate.py`) — decides if the question is in-scope at all. **Fails open**: if its confidence is below the threshold, it defaults to `on_topic=True` rather than wrongly refusing a legitimate question.
- **RAG decider** (`agent_orchestration_helper.py`) — decides if retrieval is worth doing for this *particular* on-topic message (a greeting doesn't need a document search). **Fails closed**: below its confidence threshold, it defaults to `use_rag=False` rather than injecting irrelevant context.

  These two fail-safes are deliberately asymmetric — both defaults favor *answering* over refusing or injecting noise. That's a design choice, not an oversight; don't "fix" one to match the other without re-reading why.

- **Query rewriter** (`agent_orchestration_helper.py`) — turns a conversational follow-up ("what about its evolution?") into a cleaner, standalone search query before hitting the vector store.
- **Retriever + reranker** — as described in [§3.3–3.4](#3-retrieval-02_querypy) above.
- **Persona** (`utils/persona.py`) and **off-topic refusals** (`utils/rejections.py`) — both draw their wording from `rag_content.json` (see [§8](#customizing-the-agents-scope-rag_contentjson)), so changing your topic scope automatically updates how the agent introduces itself and how it declines out-of-scope questions.

```bash
python scripts/05_chat_cli.py --show-context --debug
python scripts/06_chat_tui.py --save-transcript out.md
```

Both understand `/reset` (clear conversation memory), `/exit` or `/quit`, and `05_chat_cli.py` additionally understands `/help` and `/showctx`. `06_chat_tui.py` uses `ctrl+r` / `ctrl+s` / `ctrl+c` as keyboard shortcuts instead, since its input box is dedicated to chat text.

---

## Evaluation, two ways

This project has **two independent evaluation tracks** — pick based on whether you want a quick, free, repeatable signal or a slower, paid, more discriminating one.

### A. Human-in-the-loop, lexical heuristics (free, fast, no LLM judge)

- **`03_quiz.py`** — an interactive terminal loop: it retrieves (and optionally generates) an answer for each question in a file you provide, then lets *you* mark it faithful/abstain, tag it, and leave notes. Results save incrementally to JSONL (+ a mirrored CSV).
- **`03_eval.py`** — scores existing question/answer/context rows with a cheap lexical heuristic: word-overlap ratio between answer and context, a faithfulness threshold, and an abstain check. No LLM judge involved, so it's instant and free, but it can't catch subtler issues (a fluent, on-topic-sounding hallucination can still overlap lexically with the context).
- **`report.py`** — aggregates `03_quiz.py` output into a Markdown report with score distributions and per-tag remediation guidance.

Full walkthrough: [`docs/eval_guide.md`](docs/eval_guide.md).

### B. Automated, LLM-judged RAGAS evaluation (`07_ragas_eval.py`)

This is the more rigorous track. It runs every question in a hand-curated **golden set** (`data/eval/golden_qa.json` — question, human-written correct answer, and the verbatim corpus passage that supports it) through the real `ChatEngine` pipeline, then hands a second, independently-configurable **judge** LLM the question, the pipeline's actual answer, the chunks it actually retrieved, and the golden answer/passage, and asks it to score four things:

| Metric | Question it answers | Catches |
|---|---|---|
| **Faithfulness** | Did the answer only claim things the retrieved text actually supports? | Hallucination |
| **Answer Relevancy** | Did the answer address the question that was asked? | Accurate-but-off-topic answers |
| **Context Precision** | Was what got retrieved actually useful? | A noisy retriever/reranker |
| **Context Recall** | Did retrieval find everything needed to answer correctly? | A retriever that missed something |

The judge is intentionally allowed to be a **different model** than the pipeline under test (`--judge-provider`/`--judge-model`/`--judge-api-key`, all defaulting to the pipeline's own `--provider`/`--llm-model`/`--api-key` if omitted) — so the model isn't grading its own homework.

```bash
python scripts/01_build_index.py
python scripts/07_ragas_eval.py --golden-set data/eval/golden_qa.json --out data/eval/ragas_report.json
```

**This is real API spend, not a free check**: one pipeline call per question, plus *several* judge calls per metric per question — a 15-question golden set is on the order of a hundred-plus LLM calls. Use a cheap, fast, instruction-following judge model (`claude-haiku-4-5`, `gpt-5-mini`, etc.) rather than your most expensive model.

Deep-dive tutorial (schema, a worked example question end-to-end, how to read a low score in each metric, how to extend the golden set): [`data/eval/README.md`](data/eval/README.md).

---

## CLI reference

Every script accepts `--help` for the authoritative, always-in-sync list of options. The tables below are a quick-reference companion.

| Script | Purpose |
|---|---|
| [`00_ingest.py`](#00_ingestpy-1) | Preview chunking of `data/corpus/` |
| [`01_build_index.py`](#01_build_indexpy-1) | Embed chunks and (re)build `data/chroma/` |
| [`02_query.py`](#02_querypy-1) | Retrieve + rerank + optionally answer one question |
| [`03_eval.py`](#03_evalpy) | Lexical-heuristic scoring of saved Q/A rows |
| [`03_quiz.py`](#03_quizpy) | Interactive human review loop |
| [`04_llm_api.py`](#04_llm_apipy) | Standalone LLM smoke test (no retrieval) |
| [`05_chat_cli.py`](#05_chat_clipy) | Rich terminal chat REPL |
| [`06_chat_tui.py`](#06_chat_tuipy) | Full-screen Textual chat TUI |
| [`07_ragas_eval.py`](#07_ragas_evalpy) | Automated, LLM-judged RAGAS evaluation |
| [`08_eval_tui.py`](#08_eval_tuipy) | Full-screen Textual evaluation dashboard for RAGAS, lexical eval & human quiz |
| [`report.py`](#reportpy) | Summarize `03_quiz.py` output into a Markdown report |
| [`delete_chroma.py`](#delete_chromapy) | Reset `data/chunks/` and/or `data/chroma/` |

#### `00_ingest.py`

No CLI flags — configured via the environment variables in [§3.1](#1-ingestion-00_ingestpy).

```bash
python scripts/00_ingest.py
```

#### `01_build_index.py`

No CLI flags.

```bash
python scripts/01_build_index.py
```

<details>
<summary><code>02_query.py</code> — full flag list</summary>

| Flag | Default | Description |
|---|---|---|
| `-q, --question` | *(required)* | Question to query against the store |
| `--k` | `3` | Number of contexts to retrieve |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (must match the index) |
| `--agent-mode` | `none` | `none` \| `pretend` \| `llm` |
| `--provider` | auto-detected | LLM provider override |
| `--llm-model` | `gpt-5-mini` | Chat model name |
| `--api-key` | auto-detected | API key override |
| `--base-url` | `None` | Base URL override, for OpenAI-compatible endpoints |
| `--temperature` | `0.2` | Sampling temperature |
| `--max-tokens` | `2000` | Max response tokens |
| `--show-usage` | off | Print token usage metadata when the provider returns it |

```bash
python scripts/02_query.py -q "What is the pipeline?" --agent-mode pretend --k 5
```

</details>

<details>
<summary><code>03_eval.py</code> — full flag list</summary>

Requires exactly one of `--in` / `--questions`.

| Flag | Default | Description |
|---|---|---|
| `--in` | — | Existing JSON or CSV eval file |
| `--questions` | — | Questions file to auto-generate answers for, via the retrieval pipeline |
| `--out` | `data/eval_report.json` | Output JSON report path |
| `--agent-mode` | `pretend` | Answering strategy |
| `--k` | `3` | Contexts to retrieve when generating predictions |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--llm-model` | `gpt-5-mini` | Chat model (only used with `--agent-mode llm`) |
| `--provider` / `--api-key` / `--temperature` / `--max-tokens` / `--base-url` | — | Same as `02_query.py` |
| `--rebuild-index` | off | Run `01_build_index.py` before evaluation |

```bash
python scripts/03_eval.py --in data/eval/sample.json --out reports/sample_eval.json
```

</details>

<details>
<summary><code>03_quiz.py</code> — full flag list</summary>

| Flag | Default | Description |
|---|---|---|
| `--questions` | *(required)* | JSON file of question objects |
| `--k` | `3` | Contexts to retrieve |
| `--agent-mode` | `none` | Answer generation strategy |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--llm-model` | `gpt-5-mini` | Chat model for `llm` mode |
| `--provider` / `--api-key` / `--base-url` / `--temperature` / `--max-tokens` | — | Same as `02_query.py` |
| `--out` | `data/human_review.jsonl` | Output JSONL path |
| `--resume` | off | Resume from an existing output file |
| `--shuffle` | off | Shuffle question order |
| `--only-unlabeled` | off | Only review items missing faithful/abstain labels |
| `--page-width` | `100` | Terminal wrap width |

```bash
python scripts/03_quiz.py --questions data/questions/dev.json --agent-mode pretend --k 3 --out data/human_review.jsonl --resume
```

> ⚠️ `03_quiz.py` requires an API key in the environment even in `none`/`pretend` mode — it resolves a provider/key unconditionally at startup and exits if none is found. See [Troubleshooting](#troubleshooting--faq).

</details>

<details>
<summary><code>04_llm_api.py</code> — full flag list</summary>

A standalone smoke test — it calls the LLM directly with hand-supplied context snippets, bypassing retrieval entirely.

| Flag | Default | Description |
|---|---|---|
| `--question` | *(required)* | Question to ask |
| `--context` | `[]` | Context snippet; repeat the flag to add more |
| `--context-file` | `None` | File of context snippets (JSON list, or text separated by blank lines) |
| `--instructions` | `None` | Override the system prompt |
| `--api-key` | auto-detected | API key override |
| `--model` | `gpt-3.5-turbo` | Chat model identifier |
| `--provider` / `--temperature` / `--max-tokens` / `--base-url` | — | Same as `02_query.py` |

```bash
python scripts/04_llm_api.py --question "How does retrieval work?" --context "The retriever uses Chroma with MiniLM embeddings."
```

> Note: this script's default model (`gpt-3.5-turbo`) is older than the `gpt-5-mini` default used everywhere else in the repo — pass `--model` explicitly if you want consistency.

</details>

<details>
<summary><code>05_chat_cli.py</code> — full flag list</summary>

| Flag | Default | Description |
|---|---|---|
| `--retrieval-k` | `3` | Contexts to retrieve per turn |
| `--embedding-model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--persist-dir` | `data/chroma` | Chroma store path |
| `--llm-model` | `gpt-5-mini` | Chat model |
| `--provider` / `--api-key` / `--base-url` / `--temperature` / `--max-tokens` | — | Same as `02_query.py` |
| `--system-prompt` | `"Stay in expert research assistant mode. Follow the persona and context provided in the user message."` | Override the system prompt |
| `--show-context` | off | Display retrieved snippets alongside each answer |
| `--save-transcript` | `None` | File path to write a Markdown transcript on exit |
| `--debug` | off | Verbose debug logging to stderr |

```bash
python scripts/05_chat_cli.py --show-context --debug
```

</details>

<details>
<summary><code>06_chat_tui.py</code> — full flag list</summary>

Same pipeline as `05_chat_cli.py`, rendered as a full-screen Textual app instead of a scrolling REPL — a chat-log pane, a live context sidebar for the latest turn, and an input box. LLM calls run in a background thread so the UI stays responsive.

| Flag | Default | Description |
|---|---|---|
| `--retrieval-k` / `--embedding-model` / `--persist-dir` / `--llm-model` / `--provider` / `--api-key` / `--base-url` / `--temperature` / `--max-tokens` / `--system-prompt` | — | Same as `05_chat_cli.py` |
| `--save-transcript` | `None` | File path to write a Markdown transcript on exit |
| `--debug` | off | Verbose logging written to `chat_tui.log` (not the TUI itself, since stdout is occupied by the UI) |

Keybindings: `ctrl+r` reset chat · `ctrl+s` save transcript · `ctrl+c` quit. Typing `/exit`, `/quit`, or `/reset` also works.

```bash
python scripts/06_chat_tui.py --save-transcript out.md
```

Requires `textual`, which is only in `requirements.txt` (not `-min.txt` or `-cpu.txt`).

</details>

<details>
<summary><code>07_ragas_eval.py</code> — full flag list</summary>

| Flag | Default | Description |
|---|---|---|
| `--golden-set` | `data/eval/golden_qa.json` | Path to the golden QA JSON file |
| `--out` | `data/eval/ragas_report.json` | Output JSON report path |
| `--retrieval-k` | `3` | Contexts to retrieve per question |
| `--embedding-model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--persist-dir` | `data/chroma` | Chroma store path |
| `--llm-model` | `gpt-5-mini` | Chat model for the **pipeline under test** |
| `--provider` | auto-detected | Provider for the pipeline under test |
| `--api-key` | auto-detected | API key for the pipeline under test |
| `--judge-model` | defaults to `--llm-model` | Chat model used as the RAGAS **judge** |
| `--judge-provider` | defaults to `--provider` | Provider for the judge model |
| `--judge-api-key` | defaults to `--api-key` | API key for the judge model |
| `--temperature` | `0.2` | Sampling temperature (pipeline under test) |
| `--max-tokens` | `2000` | Max tokens (pipeline under test) |

Requires `requirements-eval.txt`.

```bash
python scripts/07_ragas_eval.py --golden-set data/eval/golden_qa.json --out data/eval/ragas_report.json --provider openai --llm-model gpt-5-mini --judge-provider claude --judge-model claude-haiku-4-5
```

</details>

<details>
<summary><code>report.py</code> and <code>delete_chroma.py</code></summary>

**`report.py`** — aggregates one or more `03_quiz.py` output files into a Markdown report.

| Flag | Default | Description |
|---|---|---|
| `--in` | *(required)* | One or more JSONL/CSV files from `03_quiz.py` |
| `--out` | `reports/human_eval_report.md` | Markdown report destination |

```bash
python scripts/report.py --in data/human_review.jsonl --out reports/human_eval_report.md
```

**`delete_chroma.py`** — stdlib-only utility to reset `data/chunks/` and/or `data/chroma/` before a fresh ingest (directories are emptied, not removed). Refuses to delete anything outside `data/`.

| Flag | Default | Description |
|---|---|---|
| `--root` | inferred repo root | Project root |
| `--targets` | `chunks chroma` | Which directories to clear |
| `--dry-run` | off | Show what would be deleted, without deleting |
| `--force` / `-y` | off | Skip the confirmation prompt |

```bash
python scripts/delete_chroma.py --dry-run
python scripts/delete_chroma.py --targets chroma --force
```

</details>

---

## Environment & provider configuration

### Model selection & flexible model switching

`simple-RAG` supports selecting any model provided by OpenAI, Google Gemini, or Anthropic Claude via the `--llm-model` flag. You are not limited to the default model (`gpt-5-mini`).

#### Supported provider models (examples)

| Provider | `--provider` value | `--llm-model` examples | Required API Key |
|---|---|---|---|
| **OpenAI** | `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-5-mini` (default) | `OPENAI_API_KEY` |
| **Google Gemini** | `gemini` | `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-pro` | `GOOGLE_API_KEY` |
| **Anthropic Claude** | `claude` | `claude-3-5-sonnet-20241022`, `claude-haiku-4-5`, `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |

#### Provider & Model Switching Examples

If you have keys configured, you can switch providers and models dynamically across query and chat scripts:

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
# Querying with Gemini
python scripts/02_query.py -q "What is retrieval?" --agent-mode llm --provider gemini --llm-model gemini-1.5-flash

# Chatting with Claude
python scripts/05_chat_cli.py --provider claude --llm-model claude-3-5-sonnet-20241022
```

</td><td>

```powershell
# Querying with Gemini
python scripts/02_query.py -q "What is retrieval?" --agent-mode llm --provider gemini --llm-model gemini-1.5-flash

# Chatting with Claude
python scripts/05_chat_cli.py --provider claude --llm-model claude-3-5-sonnet-20241022
```

</td></tr>
</table>

#### Other scripts supporting model selection

Model selection isn't restricted to `02_query.py` or chat scripts. Other scripts in the pipeline accept `--llm-model` and `--provider` as well, such as:
- `scripts/04_llm_api.py`: Standalone LLM prompt test (`python scripts/04_llm_api.py --question "..." --provider openai --llm-model gpt-4o`)
- `scripts/03_eval.py` & `scripts/03_quiz.py`: Evaluation and review scripts.
- `scripts/07_ragas_eval.py`: Automated evaluation using pipeline and judge models.

All scripts load environment variables from the process **and** from a `.env` file (via `python-dotenv`), so `.env` is the recommended, cross-platform way to configure everything below.

**Provider auto-detection** checks these keys in order — the first one found wins:

1. **`OPENAI_API_KEY`** → provider `openai` (models: `gpt-4`, `gpt-4o`, `gpt-5-mini`, etc.)
2. **`GOOGLE_API_KEY`** → provider `gemini` (models: `gemini-pro`, `gemini-1.5-flash`, etc.; requires `pip install langchain-google-genai`)
3. **`ANTHROPIC_API_KEY`** → provider `claude` (models: `claude-3-5-sonnet-20241022`, `claude-haiku-4-5`, etc.; requires `pip install langchain-anthropic`)

If you have multiple keys set, force a specific one with `--provider`:

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
python scripts/05_chat_cli.py --provider gemini --llm-model gemini-1.5-flash
```

</td><td>

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:GOOGLE_API_KEY = "AIza..."
python scripts/05_chat_cli.py --provider gemini --llm-model gemini-1.5-flash
```

</td></tr>
</table>

Every script also accepts an explicit `--api-key` to bypass auto-detection entirely:

```bash
python scripts/02_query.py -q "Test" --agent-mode llm --api-key "sk-..." --provider openai
```

**Ingestion configuration** (see [§3.1](#1-ingestion-00_ingestpy)) is also environment-driven: `CORPUS_DIR`, `INGEST_TIKTOKEN_MODEL`, `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`.

No other secrets are required — the chat agent and all helper scripts reuse the same LLM API key.

---

## Customizing the agent's scope (`rag_content.json`)

`rag_content.json` (repo root) controls what the agent considers "in scope," loaded by `agent_orchestration_helper.py` at import time:

- **`rag_topic_inventory`** — a multi-line description of what the archive covers, fed to the RAG decider and query rewriter for context.
- **`specialization_topics`** — a list of plain-text topics rendered into the persona and shown in off-topic refusals.

This repo's actual `rag_content.json` ships themed around the sample Pokémon corpus:

```json
{
  "rag_topic_inventory": "RAG covers:\n- Kanto region field notes for Generation I Pokémon (focus on Pikachu)\n- Species bios, habitats, abilities, typings, base stats, movesets\n- Trainer tips, battle tactics, evolutionary paths, item interactions\n- No real-time events; canonical up to the Indigo League era (circa 1998)",
  "specialization_topics": [
    "Kanto Pokémon species bios and typings",
    "Battle tactics and movesets",
    "Evolutionary paths and item interactions",
    "Trainer tips for the Indigo League era"
  ]
}
```

Editing this file immediately changes the agent's behavior — no code changes needed:

- The topic gate uses it to decide what's in-scope vs. what gets politely declined.
- The RAG decider uses it to judge whether a question overlaps the archive.
- The persona and refusal text are phrased using `specialization_topics`.

Swap in your own topics and description here when you replace the sample corpus with your own content.

---

## Authoring corpus content

Only `.md` (Markdown) files are supported for ingestion today; hidden files (starting with `.`) are ignored. Behavior on non-Markdown files is undefined — normalize content to `.md`.

**The rule that matters most: one heading per thing you want independently retrievable and citable.** Ingestion splits at heading boundaries first (see [§3.1](#1-ingestion-00_ingestpy)), and every chunk's metadata records its enclosing heading path. If two distinct entities share one heading, they'll end up as *one* chunk with metadata that doesn't distinguish between them — a retrieval hit for one will drag in text about the other, uncited and unlabeled.

**Good** — each entity gets its own heading:

```markdown
## 4. Expanded Species Research

### Poison & Toxin Specialists

#### Grimer → Muk

* **Typing**: Poison
* **Base Stats** (Muk): HP 105 / Atk 105 / ...
```

This becomes one chunk with metadata `{"##": "4. Expanded Species Research", "###": "Poison & Toxin Specialists", "####": "Grimer → Muk", ...}` — retrievable and citable specifically as "Grimer → Muk."

**Bad** — multiple entities bundled as bullets under a shared heading:

```markdown
### Poison Types

* **Grimer → Muk**: Typing: Poison. Base Stats...
* **Koffing → Weezing**: Typing: Poison. Base Stats...
```

This produces **one chunk covering both species**, tagged only `{"###": "Poison Types"}` — a search for "Muk" can surface a chunk that's half about Weezing, with no metadata to tell them apart. (This exact pattern was a real bug caught in this repo's history — converting bullet-per-entity sections to heading-per-entity fixed both the missing metadata and the entity's name disappearing from the searchable text once headings are stripped from `page_content`.)

---

## Project structure

| Path | Purpose |
|---|---|
| `scripts/00_ingest.py` | Chunking preview — see [§3.1](#1-ingestion-00_ingestpy) |
| `scripts/01_build_index.py` | Embed + persist to `data/chroma/` — see [§3.2](#2-indexing-01_build_indexpy) |
| `scripts/02_query.py` | Retrieve + rerank + optional LLM call — see [§3.3](#3-retrieval-02_querypy) |
| `scripts/03_eval.py` | Lexical-heuristic Q/A scoring |
| `scripts/03_quiz.py` | Interactive human review loop |
| `scripts/04_llm_api.py` | Standalone LLM smoke test |
| `scripts/05_chat_cli.py` | Rich terminal chat REPL |
| `scripts/06_chat_tui.py` | Full-screen Textual chat TUI |
| `scripts/07_ragas_eval.py` | Automated RAGAS evaluation |
| `scripts/report.py` | Markdown report from quiz results |
| `scripts/delete_chroma.py` | Reset `data/chunks/` / `data/chroma/` |
| `chat_engine.py` | **Shared, UI-agnostic pipeline core** (`ChatEngine`, `ChatEngineConfig`, `TurnResult`) — implement pipeline changes here once; `05_chat_cli.py`, `06_chat_tui.py`, and `07_ragas_eval.py` all build on this instead of duplicating logic |
| `agent_orchestration_helper.py` | Loads `rag_content.json`; builds the structured RAG decider/query rewriter; assembles the final user prompt payload with persona |
| `utils/llm_provider.py` | Provider auto-detection and chat-client construction |
| `utils/topic_gate.py` | On-topic classifier (fails open) |
| `utils/persona.py` | Persona preamble builder |
| `utils/rejections.py` | Structured off-topic refusals |
| `utils/text_sanitize.py` | Strips suggestion-style filler from LLM output |
| `utils/inventory_view.py` | Formats specialization topics into a readable one-liner |
| `rag_content.json` | Agent's topic scope — see [§8](#customizing-the-agents-scope-rag_contentjson) |
| `configs/prompts.yaml`, `configs/rag.yaml` | **Template/placeholder files, not currently loaded by any script.** Actual runtime config is via CLI flags + env vars (retrieval/generation) and `rag_content.json` (topic scope). Free to wire up if you want file-based config, but nothing reads them today. |
| `data/corpus/` | Your source `.md` files (a sample `Pokémon.MD` ships by default) |
| `data/chroma/` | Persisted Chroma vector index — safe to delete to force a rebuild |
| `data/eval/` | `golden_qa.json` + tutorial `README.md` for RAGAS evaluation |
| `tests/` | pytest suite — see [Testing](#testing) |

```text
data/
├── chroma/   # persisted Chroma collections created by 01_build_index.py
├── corpus/   # drop your markdown sources here
├── eval/     # golden_qa.json + RAGAS tutorial README
└── README.md
```

---

## Dependency notes

| File | Contains | Use when |
|---|---|---|
| `requirements-min.txt` | Core LangChain + OpenAI client + Chroma. **No `sentence-transformers`.** | You've installed embedding support separately, or only need scripts that don't touch retrieval (e.g. `04_llm_api.py`). Ingestion/indexing/reranking will fail to import without `sentence-transformers` present some other way. |
| `requirements-cpu.txt` | `requirements-min.txt` + `sentence-transformers` + `transformers`, CPU-only. | **Recommended default** — everything in this README works, no GPU required. |
| `requirements.txt` | Full pinned stack: exact `langchain`/`langchain-core`/`langchain-community` versions, plus `textual` (for `06_chat_tui.py`), `unstructured`, and all three provider SDKs. | You want the chat TUI, or want the exact versions this repo is tested against. |
| `requirements-dev.txt` | `pytest` | Running the test suite. |
| `requirements-eval.txt` | `ragas` + pinned transitive deps, layered on top of `requirements.txt` | Running `07_ragas_eval.py`. |
| `requirements-full.txt` | Chains `requirements.txt` + `requirements-eval.txt` + `requirements-dev.txt` via `-r` includes | You don't want to think about which file(s) you need — one `pip install -r requirements-full.txt` gets everything: chat TUI, RAGAS eval, and pytest. |

**Why `requirements.txt` pins `langchain` tightly instead of using loose ranges:** `chat_engine.py` uses the legacy `langchain.memory.ConversationSummaryBufferMemory`, which was removed in `langchain>=1.0`. An unpinned install can silently resolve to a version that breaks the chat CLI/TUI at import time. If you ever need to bump these, bump the pinned set together (see the comment above it in `requirements.txt`) and re-run the test suite plus a real chat session afterward — never bump one LangChain package at a time.

Large language model tooling downloads sizeable model weights (embedding + cross-encoder models). Clear caches when needed:

<table>
<tr><th>macOS / Linux</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
rm -rf ~/.cache/pip ~/.cache/huggingface
```

</td><td>

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\pip\cache" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface" -ErrorAction SilentlyContinue
```

</td></tr>
</table>

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Run a single file:

```bash
python -m pytest -q tests/test_query.py
```

(If you're already inside the project's virtual environment, `pytest -q` works directly, on either platform.)

The suite covers ingestion (heading/H1 preservation, hidden-file skipping), index building (metadata sanitization, rebuild-replaces-existing-store behavior), retrieval helpers (snippet cleaning, score coercion, prompt composition, the cross-encoder reranker *and* its lexical fallback), and the lexical eval heuristics. It does **not** currently cover `chat_engine.py`, `agent_orchestration_helper.py`, `utils/`, the chat CLI/TUI, `07_ragas_eval.py`, `report.py`, or `delete_chroma.py` — keep that in mind before treating a green test run as full pipeline coverage.

---

## Troubleshooting / FAQ

**PowerShell says `'--provider' is not recognized as the name of a cmdlet...`**
You likely pasted a bash-style multi-line command (using trailing `\`) into PowerShell. PowerShell doesn't treat `\` as a line continuation — it runs `python script.py \` as one command and then tries to run the next line (`--provider openai \`) as its own command, which fails since `--provider` isn't a program name. Either put the whole command on one line, or use a trailing backtick `` ` `` instead of `\` — see the [Quick Start](#6-run-an-automated-llm-judged-evaluation) example for both forms side by side.

**`01_build_index.py` fails with an import error mentioning `sentence_transformers`.**
You installed `requirements-min.txt`, which deliberately omits `sentence-transformers` to stay lightweight — but embedding (and the reranker) always run locally regardless of which LLM provider you use for chat. Install `requirements-cpu.txt` or `requirements.txt` instead. See [Dependency notes](#dependency-notes).

**Using an OpenAI reasoning-family model (`gpt-5*`, `o1`, `o3`, `o4`) — as the pipeline model, judge model, or both.**
These models reject the classic `max_tokens`/non-default-`temperature` chat-completions parameters that the pinned `langchain-openai==0.1.25` always sends, and the legacy conversation-memory class can't count their tokens either. All three are already worked around: `utils/llm_provider.py::is_openai_reasoning_model()` detects these models and routes around the first two constraints, and `07_ragas_eval.py` additionally wraps the judge model with RAGAS's own `LangchainLLMWrapper(..., bypass_temperature=True)` so RAGAS's per-metric temperature overrides don't hit the same wall. Nothing to configure — this is automatic — but if you see a `400 BadRequestError` mentioning `max_tokens` or `temperature` from OpenAI, or a `NotImplementedError` from `get_num_tokens_from_messages`, you've likely hit a model name these detectors don't yet recognize (e.g. a future model family) — check `is_openai_reasoning_model()`'s prefix list.

**`03_quiz.py` exits immediately even in `--agent-mode pretend`, saying no API key was found.**
This is a real quirk, not a documentation gap — `03_quiz.py` resolves a provider/key at startup unconditionally, even in modes that don't call an LLM. Set any one of `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` before running it, regardless of `--agent-mode`.

- What is the input for RAGAS? The golden set at `data/eval/golden_qa.json`.
- What does it evaluate? Faithfulness, Answer Relevancy, Context Precision, Context Recall — see [§5](#evaluation-two-ways).
- Do I need an API key? Yes — the pipeline under test and/or the judge model needs one of `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`.
- Where is the output? `data/eval/ragas_report.json` (or wherever `--out` points).
- Why is the judge model separate from the pipeline model? So the model isn't grading its own homework.
- Is this expensive? Yes — RAGAS issues multiple judge calls per metric per question. Keep the judge model cheap and fast.
- How do I run it quickly? `python scripts/01_build_index.py` then `python scripts/07_ragas_eval.py --golden-set data/eval/golden_qa.json --out data/eval/ragas_report.json`.

---

## Additional resources

- [`data/eval/README.md`](data/eval/README.md) — tutorial-level deep dive into RAGAS: the golden-set schema, a worked example question traced end-to-end through the pipeline, and how to read a low score in each metric.
- [`docs/eval_guide.md`](docs/eval_guide.md) — step-by-step walkthrough of the human-review quiz + report workflow.
- [`Agent.MD`](Agent.MD) — contributor/automation-agent-facing companion to this README: coding conventions, integration points, and a contribution checklist.

Happy building!
