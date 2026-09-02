"""07_ragas_eval.py — score the RAG pipeline against a golden question set with RAGAS.

New to RAGAS or to evaluating a RAG pipeline at all? Read data/eval/README.md
first — it explains *why* a golden set exists, walks one question through
this whole file step by step, and defines what each of the four scores below
actually measures. This docstring is the short version.

The idea in one paragraph: a "golden set" (data/eval/golden_qa.json) is a
list of questions where a human has already written down the correct answer
and pointed at the exact corpus passage that supports it. This script runs
each question through the *real* chat pipeline (topic gate -> decider ->
retrieve -> rerank -> compose -> LLM, via chat_engine.ChatEngine -- no
shortcuts, it's the same code path a chat user hits), then hands a second,
independent "judge" LLM the question, the pipeline's actual answer, the
chunks it actually retrieved, and the human-written correct answer/passage --
and asks the judge to score four things:

- Faithfulness       -- did the answer only claim things the retrieved text
                         actually supports? (catches hallucination)
- AnswerRelevancy     -- did the answer address the question that was asked?
                         (catches accurate-but-off-topic answers)
- ContextPrecision    -- was what got retrieved actually useful?
                         (catches a noisy retriever/reranker)
- ContextRecall       -- did retrieval find everything needed to answer
                         correctly? (catches a retriever that missed something)

Running this makes real LLM calls (one per pipeline turn, plus several judge
calls per RAGAS metric per question -- a 15-question golden set is on the
order of a hundred-plus calls) and requires an installed `ragas` (see
requirements-eval.txt) plus an API key for whichever provider you use.
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
# Same deal for ragas.llms.base.LangchainLLMWrapper (llm_factory is the
# eventual replacement) -- used below for the reasoning-model judge path.
warnings.filterwarnings("ignore", message=r".*LangchainLLMWrapper is deprecated.*")

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
    """Turn each golden question into a RAGAS ``SingleTurnSample`` by actually running it.

    This is the step that makes the eval honest: ``retrieved_contexts`` and
    ``response`` below come from *actually calling* ``engine.process_turn``,
    not from anything precomputed. If a corpus edit or a chunking/reranking
    change makes retrieval worse, this function is where that regression
    first shows up (as different chunks flowing into the sample), before
    RAGAS ever scores it.

    ``reference`` (the human-written correct answer) and ``reference_contexts``
    (the verbatim corpus passage that supports it) come straight from the
    golden set file and never change -- see data/eval/README.md for why they
    have to be exact quotes, not paraphrases.
    """

    from ragas import SingleTurnSample

    samples = []
    warnings_out: list[str] = []
    for item in golden_items:
        question = item["question"]
        # Each golden question is graded independently. Without this reset,
        # ChatEngine's conversation memory would carry over between
        # questions -- question 5 would be answered with question 4's
        # context still influencing the RAG decider and query rewriter,
        # which is a multi-turn *conversation* test, not what a golden set
        # of independent questions is meant to measure.
        engine.reset()
        turn = engine.process_turn(question)
        if not turn.ok:
            # turn.error means the pipeline's own LLM call failed (rate
            # limit, bad key, etc.) -- not a quality problem to score, so we
            # skip it and keep going rather than aborting the whole run over
            # one question.
            warnings_out.append(f"Question skipped (LLM call failed): {question!r} -> {turn.error}")
            continue
        # RAGAS's context metrics compare against raw passage text, so we
        # pass the plain page_content here -- not the "[source N] score=..."
        # annotated strings chat_engine.format_contexts() builds for display.
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
    """Score the collected samples with RAGAS's four classic metrics.

    ``judge_llm`` reads each sample and produces the scores; it's
    deliberately allowed to be a different model/provider than the pipeline
    being tested (see ``--judge-model``/``--judge-provider`` in
    ``parse_args``) so the same model isn't grading its own homework.
    ``judge_embeddings`` is only needed by AnswerRelevancy, which measures
    "did the answer address the question?" by embedding a few paraphrased
    guesses at what question the answer *would* suit, and comparing those
    embeddings to the real question.
    """

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    return evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)


def print_summary(result, out_path: Path) -> None:
    """Print an aggregate + per-question breakdown, and save the full report as JSON.

    Read the per-question breakdown, not just the aggregate, before drawing
    conclusions -- a single very-low score dragging down the average tells
    you something different (one question the pipeline handles badly) than
    every score being moderately low (a systemic issue). data/eval/README.md
    has worked examples of what a low score in each metric actually implies.
    """

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
        enable_semantic_cache=False,  # Independent, uncached retrieval per golden question
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

    # The judge defaults to the same provider/model/key as the pipeline under
    # test (simplest thing that works for a first run), but every judge_*
    # flag can override independently -- e.g. test a gpt-5-mini pipeline
    # while judging with claude-haiku-4-5, so the judge isn't the same model
    # (and same potential blind spots) as what it's grading.
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
    from utils.llm_provider import is_openai_reasoning_model

    judge_llm = load_chat_model(
        provider=judge_provider,
        model_name=judge_model_name,
        api_key=judge_api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        base_url=None,
    )
    if judge_provider.lower() == "openai" and is_openai_reasoning_model(judge_model_name):
        # RAGAS overrides temperature per-call for its own metric machinery
        # (self-consistency sampling, NLI decomposition, etc.), but OpenAI's
        # reasoning-family models (o1/o3/o4, gpt-5+) reject any non-default
        # temperature outright. RAGAS anticipates exactly this case --
        # LangchainLLMWrapper(..., bypass_temperature=True) skips those
        # per-call overrides so the judge model's own default is used.
        from ragas.llms.base import LangchainLLMWrapper

        judge_llm = LangchainLLMWrapper(judge_llm, bypass_temperature=True)
    print(f"Scoring {len(samples)} samples with RAGAS (judge={judge_provider}/{judge_model_name})...")

    result = run_ragas(samples, judge_llm, engine.store.embeddings)
    print_summary(result, Path(args.out))


if __name__ == "__main__":
    main()
