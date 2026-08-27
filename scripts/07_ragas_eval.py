"""07_ragas_eval.py — score the RAG pipeline against a golden question set with RAGAS.

For each (question, ground_truth, reference_contexts) triple in the golden
set, this runs the *real* chat pipeline (topic gate -> decider -> retrieve ->
rerank -> compose -> LLM, via ``chat_engine.ChatEngine``) to collect the
actual answer and retrieved contexts, then scores the batch with RAGAS:
Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall.

Running this makes real LLM calls (one per pipeline turn, plus several judge
calls per RAGAS metric per question) and requires an installed `ragas`
(see requirements-eval.txt) plus an API key for whichever provider you use.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, List, Sequence

try:  # Optional dependency for convenient local development.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy deprecation warnings without changing packages.
try:  # Best-effort: some environments provide this warning class
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", message=r".*HuggingFaceEmbeddings.*was deprecated.*")
    warnings.filterwarnings("ignore", message=r".*manual persistence method is no longer supported.*")
# ragas.metrics' top-level Faithfulness/AnswerRelevancy/etc. classes emit a
# forward-looking deprecation notice (ragas.metrics.collections is the
# eventual replacement); harmless today, silenced to keep output readable.
warnings.filterwarnings("ignore", message=r".*is deprecated and will be removed in v1\.0.*")

from chat_engine import CHROMA_DIR, DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, ChatEngine, ChatEngineConfig
from utils.llm_provider import resolve_provider_and_key


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the RAG pipeline against a golden QA set with RAGAS")
    parser.add_argument("--golden-set", default="data/eval/golden_qa.json", help="Path to the golden QA JSON file")
    parser.add_argument("--out", default="data/eval/ragas_report.json", help="Output JSON report path")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Number of context chunks to retrieve per question")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model to load for retrieval")
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR), help="Path to the persisted Chroma directory")

    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model used by the pipeline under test")
    parser.add_argument("--provider", default=None, help="LLM provider for the pipeline under test (auto-detected if omitted)")
    parser.add_argument("--api-key", dest="api_key", help="API key for the pipeline under test (auto-detected if omitted)")

    parser.add_argument(
        "--judge-model",
        default=None,
        help="Chat model used as the RAGAS judge. Defaults to --llm-model. "
        "A cheap, fast, instruction-following model is recommended (e.g. claude-haiku-4-5, "
        "gpt-5-mini) since RAGAS issues several judge calls per metric per question.",
    )
    parser.add_argument("--judge-provider", default=None, help="Provider for the judge model. Defaults to --provider.")
    parser.add_argument("--judge-api-key", dest="judge_api_key", help="API key for the judge model. Defaults to --api-key.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the pipeline under test")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens for the pipeline under test")
    return parser.parse_args(argv)


def load_golden_set(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Golden set at {path} must be a non-empty JSON list.")
    return data


def collect_samples(engine: ChatEngine, golden_items: Sequence[dict]) -> tuple[list, list[str]]:
    """Run each golden question through the real pipeline; return (samples, warnings)."""

    from ragas import SingleTurnSample

    samples = []
    warnings_out: list[str] = []
    for item in golden_items:
        question = item["question"]
        engine.reset()  # each golden question is independent, not a multi-turn conversation
        turn = engine.process_turn(question)
        if not turn.ok:
            warnings_out.append(f"Question skipped (LLM call failed): {question!r} -> {turn.error}")
            continue
        retrieved_contexts = [getattr(doc, "page_content", "") for doc, _ in turn.results]
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=turn.answer,
                retrieved_contexts=retrieved_contexts,
                reference=item.get("ground_truth"),
                reference_contexts=item.get("reference_contexts"),
            )
        )
    return samples, warnings_out


def run_ragas(samples: list, judge_llm: Any, judge_embeddings: Any):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    return evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)


def print_summary(result, out_path: Path) -> None:
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}]

    print("\n=== RAGAS aggregate scores ===")
    for col in metric_cols:
        print(f" - {col}: {df[col].mean():.3f}")

    print("\n=== Per-question scores ===")
    for _, row in df.iterrows():
        question = str(row["user_input"])[:70]
        scores = "  ".join(f"{col}={row[col]:.2f}" for col in metric_cols)
        print(f" - {question}\n     {scores}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    print(f"\nFull per-question report written to {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"[ERROR] Golden set not found: {golden_path}", file=sys.stderr)
        sys.exit(1)
    golden_items = load_golden_set(golden_path)

    config = ChatEngineConfig(
        persist_dir=Path(args.persist_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        provider=args.provider,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retrieval_k=args.retrieval_k,
    )
    try:
        engine = ChatEngine(config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(golden_items)} golden questions through the pipeline (provider={engine.provider})...")
    samples, warnings_out = collect_samples(engine, golden_items)
    for w in warnings_out:
        print(f"[WARN] {w}", file=sys.stderr)
    if not samples:
        print("[ERROR] No samples were collected; nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    judge_provider, judge_api_key = resolve_provider_and_key(
        args.judge_api_key or args.api_key, args.judge_provider or args.provider
    )
    if not judge_api_key:
        print(
            "[ERROR] No API key found for the judge model. Set OPENAI_API_KEY, GOOGLE_API_KEY, "
            "or ANTHROPIC_API_KEY, or pass --judge-api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    judge_model_name = args.judge_model or args.llm_model
    from chat_engine import load_chat_model

    judge_llm = load_chat_model(
        provider=judge_provider,
        model_name=judge_model_name,
        api_key=judge_api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        base_url=None,
    )
    print(f"Scoring {len(samples)} samples with RAGAS (judge={judge_provider}/{judge_model_name})...")

    result = run_ragas(samples, judge_llm, engine.store.embeddings)
    print_summary(result, Path(args.out))


if __name__ == "__main__":
    main()
