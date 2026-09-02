"""02_query.py — retrieve contexts and optionally call a chat model.

The CLI shipped here connects to the persisted Chroma store, retrieves the top-k
contexts for a supplied question, and offers three execution modes:
``none`` (print contexts + stitched answer), ``pretend`` (prompt preview with
mocked citations), and ``llm`` (live OpenAI-compatible call). Use it to validate
your index before integrating richer agents.
"""

from __future__ import annotations

import argparse
# Optional dependency; fall back to a no-op when not installed.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from utils.llm_provider import (
    MissingProviderDependencyError,
    UnsupportedProviderError,
    auto_detect_provider,
    build_chat_model as load_chat_model,
    resolve_provider_and_key,
)

# Suppress noisy deprecation warnings without changing packages.
try:  # Best-effort: some environments provide this warning class
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    # Fallback to message-based filters if the class isn't importable
    warnings.filterwarnings(
        "ignore",
        message=r".*HuggingFaceEmbeddings.*was deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*manual persistence method is no longer supported.*",
    )

"""
This script provides a minimal, runnable retrieval smoke-test so you can see the
end-to-end flow without needing an external LLM. It loads the persisted Chroma
collection, runs a retriever for the supplied question, and prints the top
contexts plus a simple synthesized answer (concatenation of retrieved chunks).

The implementation intentionally keeps the agent light-weight so it runs in
offline environments and is easy to extend to call a real chat model later.
"""

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5-mini"
CHROMA_DIR = Path("data") / "chroma"
SYSTEM_PROMPT = (
    "Answer ONLY using the provided contexts. If unknown, say you don't know. "
    "Cite as [source N]."
)

RetrieverResult = List[Tuple[Document, float | None]]
load_dotenv()


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is requested without an API key."""


class LLMInvocationError(RuntimeError):
    """Raised when the underlying LLM client fails to produce a response."""


@dataclass
class LLMResult:
    """Container for the normalized LLM response and metadata."""

    text: str
    raw_response: object
    usage: dict | None

def load_vector_store(persist_dir: Path, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Connect to the Chroma collection built during the indexing milestone."""

    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist directory not found: {persist_dir}")
    store = Chroma(persist_directory=str(persist_dir), embedding_function=embedding_model)
    return store


def create_retrieval_store(
    model_name: str = DEFAULT_MODEL_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> tuple[HuggingFaceEmbeddings, Chroma]:
    """Instantiate embeddings and connect to the persisted Chroma store."""

    embed = HuggingFaceEmbeddings(model_name=model_name)
    store = load_vector_store(persist_dir, embed)
    return embed, store


def clean_snippet(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return " ".join(stripped.split())


def format_metadata(metadata: dict | None) -> str:
    if not metadata:
        return "metadata: none"
    parts = [f"{key}={value}" for key, value in metadata.items()]
    return "metadata: " + ", ".join(parts)


def retrieve_contexts(store: Chroma, question: str | Sequence[float], k: int) -> RetrieverResult:
    if not isinstance(question, str) and hasattr(store, "similarity_search_by_vector_with_relevance_scores"):
        raw_results = store.similarity_search_by_vector_with_relevance_scores(list(question), k=k)
        relevance_fn_getter = getattr(store, "_select_relevance_score_fn", None)
        relevance_fn = relevance_fn_getter() if callable(relevance_fn_getter) else None
        results = [
            (doc, relevance_fn(score) if relevance_fn is not None else score)
            for doc, score in raw_results
        ]
    else:
        results = store.similarity_search_with_relevance_scores(question, k=k)
    formatted: RetrieverResult = []
    for doc, score in results:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            warnings.warn(
                "Encountered non-numeric relevance score from retriever; treating as unknown.",
                RuntimeWarning,
                stacklevel=2,
            )
            numeric_score = None
        formatted.append((doc, numeric_score))
    return formatted


CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None
_cross_encoder_load_failed = False


def _load_cross_encoder():
    """Lazily load and cache the cross-encoder reranker model.

    Returns ``None`` (without retrying) if it can't be loaded -- missing
    optional dependency, offline environment, etc. -- so callers can fall
    back to the lexical reranker gracefully.
    """
    global _cross_encoder, _cross_encoder_load_failed
    if _cross_encoder is not None or _cross_encoder_load_failed:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    except Exception:
        _cross_encoder_load_failed = True
        return None
    return _cross_encoder


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _rerank_cross_encoder(results: RetrieverResult, question: str) -> RetrieverResult | None:
    """Rerank with a cross-encoder model; returns ``None`` if unavailable.

    Cross-encoders score a (query, document) pair jointly, which captures
    semantic relevance the lexical blend below can't (synonyms, paraphrase,
    word order) at the cost of one forward pass per candidate -- fine at the
    small k this project retrieves with.
    """
    if not results:
        return None
    model = _load_cross_encoder()
    if model is None:
        return None
    pairs = [(question, getattr(doc, "page_content", "") or "") for doc, _ in results]
    try:
        raw_scores = model.predict(pairs)
    except Exception:
        return None

    reranked: list[tuple[tuple[Document, float | None], float]] = []
    for (doc, score), raw_score in zip(results, raw_scores):
        combined = _sigmoid(float(raw_score))
        try:
            md = getattr(doc, "metadata", {}) or {}
            md["combined_score"] = round(combined, 4)
            md["cross_encoder_score"] = round(combined, 4)
            doc.metadata = md
        except Exception:
            pass
        reranked.append(((doc, score), combined))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in reranked]


def rerank_results(
    results: RetrieverResult,
    question: str,
    alpha: float = 0.5,
    use_cross_encoder: bool = True,
) -> RetrieverResult:
    """Rerank retrieved results, preferring a local cross-encoder model.

    Parameters
    ----------
    results:
        List of (Document, score) tuples returned by the retriever. Scores may
        be None if the retriever did not provide numeric values.
    question:
        The original user query.
    alpha:
        Weight for the original retriever score (0..1) in the lexical-blend
        fallback below; unused when the cross-encoder path succeeds.
    use_cross_encoder:
        Try the cross-encoder model first (default). Set to False to force
        the lexical blend -- e.g. in tests, to avoid depending on a
        downloaded ML model -- or in offline environments where the model
        can't be fetched (loading already fails gracefully either way).

    Returns
    -------
    RetrieverResult
        Results sorted by the combined score (descending).
    """
    if use_cross_encoder:
        ce_result = _rerank_cross_encoder(results, question)
        if ce_result is not None:
            return ce_result
        # Cross-encoder unavailable (dependency missing, offline, or a
        # load/predict failure) -- fall through to the lexical blend.

    import re

    # Extract numeric retriever scores and compute normalization bounds.
    raw_scores: list[float] = []
    for _, s in results:
        raw_scores.append(float(s) if s is not None else 0.0)

    if raw_scores:
        min_s = min(raw_scores)
        max_s = max(raw_scores)
    else:
        min_s = max_s = 0.0

    def normalize(s: float) -> float:
        if max_s == min_s:
            return 1.0 if max_s != 0 else 0.0
        return (s - min_s) / (max_s - min_s)

    # Tokenize question once
    q_tokens = set(re.findall(r"\w+", (question or "").lower()))
    if not q_tokens:
        # If question tokenization fails, fallback to retriever-only ordering
        scored = [((doc, score), normalize(float(score) if score is not None else 0.0)) for doc, score in results]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_sorted]

    reranked: list[tuple[tuple[Document, float | None], float]] = []
    for (doc, score) in results:
        doc_text = getattr(doc, "page_content", "") or ""
        d_tokens = set(re.findall(r"\w+", doc_text.lower()))
        # lexical overlap: fraction of question tokens present in the doc
        lexical = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else 0.0
        retriever_norm = normalize(float(score) if score is not None else 0.0)
        combined = alpha * retriever_norm + (1.0 - alpha) * lexical
        # Store combined and components in metadata for downstream display
        try:
            md = getattr(doc, "metadata", {}) or {}
            md["combined_score"] = round(combined, 4)
            md["lexical_overlap"] = round(lexical, 4)
            md["retriever_norm"] = round(retriever_norm, 4)
            doc.metadata = md
        except Exception:
            pass
        reranked.append(((doc, score), combined))

    reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)
    return [item[0] for item in reranked_sorted]


def emit_contexts(
    results: RetrieverResult,
    header: str,
    *,
    label_factory: Callable[[int], str] | None = None,
    limit: int | None = None,
) -> None:
    print(f"\n{header}")
    if not results:
        print("No contexts retrieved.")
        return
    label_factory = label_factory or (lambda idx: f"[{idx}]")
    window = results if limit is None else results[:limit]
    for idx, (doc, score) in enumerate(window):
        meta_line = format_metadata(getattr(doc, "metadata", {}))
        snippet = clean_snippet(doc.page_content)
        score_display = f"{score:.3f}" if score is not None else "n/a"
        # Display combined score if present in metadata
        combined = None
        try:
            combined = getattr(doc, "metadata", {}).get("combined_score")
        except Exception:
            combined = None
        if combined is not None:
            print(f"{label_factory(idx)} score: {score_display} | rerank: {combined:.3f} | {meta_line}")
        else:
            print(f"{label_factory(idx)} score: {score_display} | {meta_line}")
        if snippet:
            print(textwrap.fill(snippet, width=120))
        else:
            print("(empty snippet)")
        print("-")


def synthesize_from_results(results: RetrieverResult, limit: int) -> str:
    snippets: List[str] = []
    for doc, _ in results[:limit]:
        snippet = clean_snippet(doc.page_content)
        if snippet:
            snippets.append(snippet)
    return "\n\n".join(snippets) if snippets else "No relevant context found."


def run_none_mode(results: RetrieverResult, k: int) -> None:
    emit_contexts(results, "Retrieved contexts:")
    print("\nSynthesized answer:\n")
    print(synthesize_from_results(results, k))


def run_pretend_mode(question: str, results: RetrieverResult, k: int) -> None:
    print("\n=== Pretend Agent Prompt Preview ===\n")
    print("System instruction:")
    print(textwrap.fill(SYSTEM_PROMPT, width=120))
    print("\nUser question:")
    print(textwrap.fill(question, width=120))
    emit_contexts(
        results,
        "Retrieved contexts (top k):",
        label_factory=lambda idx: f"[source {idx}]",
        limit=k,
    )
    cited = [str(idx) for idx in range(min(k, len(results)))]
    synth = " ".join(
        clean_snippet(doc.page_content) for doc, _ in results[:k]
    )
    templated_answer = (
        "Answer (synthesized from sources ["
        + ",".join(cited)
        + "]):\n\n"
        + (synth or "No relevant context found.")
    )
    print("\n=== Pretend Agent Final Answer ===\n")
    print(templated_answer)


def compose_user_prompt(question: str, results: RetrieverResult) -> str:
    lines: List[str] = [f"Question: {question}", "", "Contexts:"]
    if not results:
        lines.append("No context retrieved.")
    else:
        for idx, (doc, score) in enumerate(results):
            meta_line = format_metadata(getattr(doc, "metadata", {}))
            snippet = clean_snippet(doc.page_content)
            score_display = f"{score:.3f}" if score is not None else "n/a"
            combined = None
            try:
                combined = getattr(doc, "metadata", {}).get("combined_score")
            except Exception:
                combined = None
            if combined is not None:
                lines.append(f"[source {idx}] score: {score_display} | rerank: {combined:.3f} | {meta_line}")
            else:
                lines.append(f"[source {idx}] score: {score_display} | {meta_line}")
            lines.append(snippet or "(empty snippet)")
            lines.append("")
    return "\n".join(lines).strip()


def compose_messages(
    question: str,
    results: RetrieverResult,
    system_prompt: str | None = None,
) -> List[BaseMessage]:
    """Create the chat messages for the retrieval-informed LLM call."""

    return [
        SystemMessage(content=system_prompt if system_prompt is not None else SYSTEM_PROMPT),
        HumanMessage(content=compose_user_prompt(question, results)),
    ]


def extract_text_from_response(response) -> str:
    """Normalize the message content into a plain string for downstream use."""

    content = response.content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(chunk.get("text", ""))
            else:
                parts.append(str(chunk))
        final_text = " ".join(part for part in parts if part).strip()
    else:
        final_text = str(content).strip()
    return final_text


def extract_usage_metadata(response) -> dict | None:
    usage = None
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage")
    if not usage and hasattr(response, "additional_kwargs"):
        usage = response.additional_kwargs.get("usage")
    return usage


def call_chat_model(
    messages: Sequence[BaseMessage],
    provider: str,
    model_name: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
) -> LLMResult:
    """Call the chat model and return a normalized ``LLMResult``."""

    if not api_key:
        raise MissingAPIKeyError("API key is required for live LLM calls.")

    llm = load_chat_model(provider, model_name, api_key, temperature, max_tokens, base_url)
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # pragma: no cover - network/remote failure
        raise LLMInvocationError(f"LLM call failed: {exc}") from exc

    final_text = extract_text_from_response(response)
    usage = extract_usage_metadata(response)
    return LLMResult(text=final_text, raw_response=response, usage=usage)


def print_usage_metadata(usage: dict | None, show_usage: bool) -> None:
    if not show_usage or not usage:
        return
    print("\nUsage metadata:")
    if isinstance(usage, dict):
        for key, value in usage.items():
            print(f"- {key}: {value}")
    else:
        print(usage)


def print_mock_answer(results: RetrieverResult) -> None:
    print("\n=== Final Answer ===\n")
    indices = [str(idx) for idx in range(len(results))]
    snippets: List[str] = []
    for doc, _ in results:
        snippet = clean_snippet(doc.page_content)
        if snippet:
            snippets.append(snippet)
    combined = " ".join(snippets) if snippets else "No relevant context available."
    print(f"(mock) Answer (synthesized from sources [{', '.join(indices)}]): {combined}")


def run_llm_mode(
    question: str,
    results: RetrieverResult,
    provider: str,
    model_name: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
    show_usage: bool,
) -> None:
    emit_contexts(results, "=== Retrieved Evidence ===")
    messages = compose_messages(question, results)
    try:
        result = call_chat_model(
            messages=messages,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
    except MissingAPIKeyError:
        print("OpenAI API key missing. Falling back to mock answer.", file=sys.stderr)
        print_mock_answer(results)
        return
    except (UnsupportedProviderError, MissingProviderDependencyError, LLMInvocationError) as exc:
        print(str(exc), file=sys.stderr)
        print_mock_answer(results)
        return

    print("\n=== Final Answer ===\n")
    print(result.text or "I don't know.")
    print_usage_metadata(result.usage, show_usage)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval smoke-test: query the Chroma store")
    parser.add_argument("--question", "-q", required=True, help="Question to query against the store")
    parser.add_argument("--k", type=int, default=3, help="Number of contexts to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model name to use (must match index)")
    parser.add_argument(
        "--agent-mode",
        choices=["none", "pretend", "llm"],
        default="none",
        help="Agent mode: 'none' (no LLM), 'pretend' (show prompt + templated answer), or 'llm' (call real model)",
    )
    parser.add_argument("--provider", default=None, help="LLM provider override (auto-detected from API keys if not specified)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model name to request (default: gpt-5-mini)")
    parser.add_argument("--api-key", dest="api_key", help="API key override (auto-detected from environment if not specified)")
    parser.add_argument("--base-url", dest="base_url", help="Optional base URL for OpenAI-compatible endpoints")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for the chat model response")
    parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Print token usage metadata when returned by the provider",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    _, store = create_retrieval_store(model_name=args.model, persist_dir=CHROMA_DIR)
    results = retrieve_contexts(store, args.question, args.k)
    # Apply lightweight reranker to improve lexical relevance blending with
    # retriever scores. This is intentionally lightweight and runs locally.
    try:
        results = rerank_results(results, args.question, alpha=0.5)
    except Exception:
        # Reranker is best-effort; if it fails, fall back to original ordering.
        pass

    if args.agent_mode == "none":
        run_none_mode(results, args.k)
        return

    if args.agent_mode == "pretend":
        run_pretend_mode(args.question, results, args.k)
        return

    provider, api_key = resolve_provider_and_key(args.api_key, args.provider)
    
    if not api_key:
        print("[ERROR] No API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    run_llm_mode(
        question=args.question,
        results=results,
        provider=provider,
        model_name=args.llm_model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        show_usage=args.show_usage,
    )


if __name__ == "__main__":
    main()
