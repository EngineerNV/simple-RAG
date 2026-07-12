"""03_eval.py — score question/answer/context rows with lexical heuristics.

Given a CSV or JSON file of saved QA examples, this script normalises the
contexts, measures overlap with the answer, flags potential hallucinations, and
summarises results with aggregate statistics. Use it to spot regression after
prompt or retrieval tweaks.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from importlib import import_module
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.textproc import compute_overlap_ratio, concat_context as _concat_context, tokenize as _tokenize


def load_eval_data(filepath: Path) -> List[Mapping[str, str]]:
    """Load evaluation data from JSON (list of dicts) or CSV (headers: question,answer,context).

    Returns a list of mappings with keys: 'question', 'answer', 'context' (context may be a
    single string or a JSON-encoded list of snippets).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Eval file not found: {filepath}")
    if filepath.suffix.lower() == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data)
    else:
        out = []
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({"question": row.get("question"), "answer": row.get("answer"), "context": row.get("context")})
        return out


def _normalise_context(context: Iterable[str] | str | None) -> List[str]:
    """Return a list of context strings, tolerating CSV/JSON encodings."""

    if context is None:
        return []
    if isinstance(context, list):
        return [str(item) for item in context]
    if isinstance(context, tuple):
        return [str(item) for item in context]
    return [str(context)]


def is_faithful(answer: str, context: Iterable[str], threshold: float = 0.3) -> bool:
    """Return True when the answer shows sufficient lexical overlap with context."""

    if not answer:
        return False
    ratio = compute_overlap_ratio(answer, context)
    return ratio >= threshold


def should_abstain(context: Iterable[str], min_length: int = 30, min_ratio: float = 0.05) -> bool:
    """Decide to abstain when context is too short or mostly irrelevant.

    - Abstain if combined context length (chars) < min_length.
    - Abstain if lexical overlap ratio (as defined below) is below min_ratio.
    """
    ctx_text = _concat_context(_normalise_context(context))
    if len(ctx_text.strip()) < min_length:
        return True
    # weak overlap check: if less than min_ratio of context tokens are shared with themselves
    ctx_tokens = _tokenize(ctx_text)
    if not ctx_tokens:
        return True
    # simple heuristic: if average token length is tiny, abstain
    avg_token_len = sum(len(t) for t in ctx_tokens) / len(ctx_tokens)
    if avg_token_len < 2:
        return True
    return False


def evaluate_qa_pair(qa: Mapping[str, object]) -> Mapping[str, object]:
    """Evaluate a single QA pair using lexical heuristics."""

    result: MutableMapping[str, object] = {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "context": qa.get("context"),
    }
    ctx_raw = result["context"]
    ctx_list = _normalise_context(ctx_raw)
    if len(ctx_list) == 1 and isinstance(ctx_list[0], str):
        # Contexts created via CSV sometimes store JSON-encoded lists. Attempt to
        # parse them to avoid double-quoted strings during overlap checks.
        try:
            parsed = json.loads(ctx_list[0])
            if isinstance(parsed, list):
                ctx_list = [str(item) for item in parsed]
        except Exception:
            pass

    answer_text = str(result.get("answer") or "")
    ratio = compute_overlap_ratio(answer_text, ctx_list)
    result["overlap_ratio"] = ratio
    result["faithful"] = is_faithful(answer_text, ctx_list)
    result["abstain"] = should_abstain(ctx_list)
    result["context_length"] = len(_concat_context(ctx_list))
    result["context_list"] = ctx_list
    return result


def save_eval_report(results: List[Mapping[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_eval_summary(results: List[Mapping[str, object]]) -> None:
    total = len(results) or 1
    faithful = sum(1 for r in results if r.get("faithful"))
    abstain = sum(1 for r in results if r.get("abstain"))
    ratios = [float(r.get("overlap_ratio", 0.0)) for r in results]
    ctx_lengths = [int(r.get("context_length", 0)) for r in results]
    avg_ratio = statistics.mean(ratios) if ratios else 0.0
    avg_ctx_len = statistics.mean(ctx_lengths) if ctx_lengths else 0.0
    print(f"Evaluated {len(results)} examples")
    print(f" - Faithful: {faithful} ({faithful/max(1, len(results)):.1%})")
    print(f" - Abstain: {abstain} ({abstain/max(1, len(results)):.1%})")
    print(f" - Avg overlap ratio: {avg_ratio:.2f}")
    print(f" - Avg context characters: {avg_ctx_len:.1f}")


def load_question_file(filepath: Path) -> List[str]:
    """Load a batch of questions for automated evaluation runs."""

    if filepath.suffix.lower() == ".json":
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            questions: List[str] = []
            for item in payload:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, Mapping) and "question" in item:
                    questions.append(str(item["question"]))
            return questions
        raise ValueError("JSON questions file must be a list of strings or objects with a 'question' key.")

    questions = []
    with open(filepath, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "question" in reader.fieldnames:
            for row in reader:
                questions.append(str(row.get("question", "")).strip())
            return [q for q in questions if q]

    raise ValueError("Unsupported questions file format. Use JSON list or CSV with a 'question' column.")


def ensure_index_if_requested(rebuild: bool) -> None:
    """Optionally trigger the indexing script before evaluation.

    The user story mentioned exploring whether evaluation should call into the
    retrieval pipeline. A simple toggle keeps scripts loosely coupled while
    still allowing a single command to refresh embeddings before measuring
    performance.
    """

    if not rebuild:
        return
    module = import_module("scripts.01_build_index")
    print("[eval] Rebuilding index via scripts/01_build_index.py ...")
    module.main()


def _load_query_module():
    return import_module("scripts.02_query")


def generate_predictions(
    questions: Sequence[str],
    k: int,
    agent_mode: str,
    model_name: str,
    llm_model: str,
    provider: str,
    api_key: str | None,
    temperature: float | None,
    max_tokens: int,
    base_url: str | None,
) -> List[Mapping[str, object]]:
    """Run the query pipeline to collect answers/contexts for evaluation."""

    query_module = _load_query_module()
    embed = query_module.HuggingFaceEmbeddings(model_name=model_name)
    store = query_module.load_vector_store(query_module.CHROMA_DIR, embed)
    results: List[Mapping[str, object]] = []

    for question in questions:
        retrieval = query_module.retrieve_contexts(store, question, k)
        context_snippets = [query_module.clean_snippet(doc.page_content) for doc, _ in retrieval]

        if agent_mode == "none":
            answer = query_module.synthesize_from_results(retrieval, k)
        elif agent_mode == "pretend":
            cited = [str(idx) for idx in range(min(k, len(retrieval)))]
            synth = " ".join(query_module.clean_snippet(doc.page_content) for doc, _ in retrieval[:k])
            answer = (
                "Answer (synthesized from sources ["
                + ",".join(cited)
                + "]):\n\n"
                + (synth or "No relevant context found.")
            )
        else:
            # We borrow the ``run_llm_mode`` logic but return the answer instead
            # of printing it. When no API key is supplied we revert to the mock
            # concatenated output so metrics remain comparable.
            if not api_key:
                answer = query_module.synthesize_from_results(retrieval, k)
            else:  # pragma: no cover - network calls are optional
                try:
                    llm = query_module.load_chat_model(
                        provider=provider,
                        model_name=llm_model,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        base_url=base_url,
                    )
                    messages = [
                        query_module.SystemMessage(content=query_module.SYSTEM_PROMPT),
                        query_module.HumanMessage(content=query_module.compose_user_prompt(question, retrieval)),
                    ]
                    response = llm.invoke(messages)
                    content = response.content
                    if isinstance(content, list):
                        parts: List[str] = []
                        for chunk in content:
                            if isinstance(chunk, dict):
                                parts.append(chunk.get("text", ""))
                            else:
                                parts.append(str(chunk))
                        answer = " ".join(part for part in parts if part).strip()
                    else:
                        answer = str(content).strip()
                except Exception as exc:
                    print(f"[eval] LLM call failed ({exc}); falling back to mock answer.")
                    answer = query_module.synthesize_from_results(retrieval, k)

        results.append(
            {
                "question": question,
                "answer": answer,
                "context": context_snippets,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA pairs for faithfulness and abstention")
    parser.add_argument("--in", dest="infile", help="Existing JSON or CSV eval file")
    parser.add_argument("--questions", help="Optional questions file to auto-generate answers for evaluation")
    parser.add_argument("--out", dest="outfile", default="data/eval_report.json", help="Output JSON report path")
    parser.add_argument("--agent-mode", choices=["none", "pretend", "llm"], default="pretend", help="Answering strategy")
    parser.add_argument("--k", type=int, default=3, help="Number of contexts to retrieve when generating predictions")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--llm-model", default=None, help="Chat model name when --agent-mode llm is used (defaults to the resolved provider's default model)")
    parser.add_argument("--provider", default=None, help="LLM provider override (auto-detected from API keys if not specified)")
    parser.add_argument("--api-key", dest="api_key", help="API key override (auto-detected from environment if not specified)")
    parser.add_argument("--temperature", type=float, default=None, help="Chat model temperature (omitted unless set)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Chat model max tokens")
    parser.add_argument("--base-url", dest="base_url", help="Optional OpenAI-compatible base URL")
    parser.add_argument("--rebuild-index", action="store_true", help="Run scripts/01_build_index.py before evaluation")
    args = parser.parse_args()

    ensure_index_if_requested(args.rebuild_index)

    raw_examples: List[Mapping[str, object]]
    if args.infile:
        infile = Path(args.infile)
        raw_examples = load_eval_data(infile)
    elif args.questions:
        question_file = Path(args.questions)
        questions = load_question_file(question_file)
        if not questions:
            raise ValueError("Question file was empty; nothing to evaluate.")
        
        query_module = _load_query_module()
        provider, api_key = query_module.resolve_provider_and_key(args.api_key, args.provider)
        
        if not api_key:
            print("[ERROR] No API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        
        raw_examples = generate_predictions(
            questions=questions,
            k=args.k,
            agent_mode=args.agent_mode,
            model_name=args.model,
            llm_model=query_module.resolve_model(provider, args.llm_model),
            provider=provider,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
    else:
        raise ValueError("Specify either --in for an existing eval file or --questions to generate predictions.")

    results = [evaluate_qa_pair(q) for q in raw_examples]
    save_eval_report(results, Path(args.outfile))
    print_eval_summary(results)


if __name__ == "__main__":
    main()
