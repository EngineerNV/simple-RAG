"""02_query.py — retrieve contexts and optionally call a chat model.

The CLI shipped here connects to the persisted Chroma store, retrieves the top-k
contexts for a supplied question, and offers three execution modes:
``none`` (print contexts + stitched answer), ``pretend`` (prompt preview with
mocked citations), and ``llm`` (live OpenAI-compatible call). Use it to validate
your index before integrating richer agents.
"""

from __future__ import annotations

import argparse
import os
# Optional dependency; fall back to a no-op when not installed.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False
import sys
import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

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

RetrieverResult = List[Tuple[Document, float]]
load_dotenv()


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is requested without an API key."""


class LLMInvocationError(RuntimeError):
    """Raised when the underlying LLM client fails to produce a response."""

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


def retrieve_contexts(store: Chroma, question: str, k: int) -> RetrieverResult:
    results = store.similarity_search_with_relevance_scores(question, k=k)
    formatted: RetrieverResult = []
    for doc, score in results:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0
        formatted.append((doc, numeric_score))
    return formatted


def emit_contexts(results: RetrieverResult, header: str) -> None:
    print(f"\n{header}")
    if not results:
        print("No contexts retrieved.")
        return
    for idx, (doc, score) in enumerate(results):
        meta_line = format_metadata(getattr(doc, "metadata", {}))
        snippet = clean_snippet(doc.page_content)
        print(f"[{idx}] score: {score:.3f} | {meta_line}")
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
    print("\nRetrieved contexts (top k):")
    if not results:
        print("No contexts retrieved.")
    for idx, (doc, score) in enumerate(results[:k]):
        meta_line = format_metadata(getattr(doc, "metadata", {}))
        snippet = clean_snippet(doc.page_content)
        print(f"[source {idx}] score: {score:.3f} | {meta_line}")
        if snippet:
            print(textwrap.fill(snippet, width=120))
        else:
            print("(empty snippet)")
        print("-")
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
            lines.append(f"[source {idx}] score: {score:.3f} | {meta_line}")
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
        SystemMessage(content=system_prompt or SYSTEM_PROMPT),
        HumanMessage(content=compose_user_prompt(question, results)),
    ]


def load_chat_model(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
):
    if provider != "openai":
        raise RuntimeError(f"Unsupported provider '{provider}'.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Missing optional dependency 'langchain-openai'. Install it with `pip install langchain-openai`."
        ) from exc

    init_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "openai_api_key": api_key,
    }
    if base_url:
        init_kwargs["openai_api_base"] = base_url
    return ChatOpenAI(**init_kwargs)


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
) -> tuple[str, object, dict | None]:
    """Call the chat model and return (text, raw_response, usage_metadata)."""

    if not api_key:
        raise MissingAPIKeyError("API key is required for live LLM calls.")

    llm = load_chat_model(provider, model_name, api_key, temperature, max_tokens, base_url)
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # pragma: no cover - network/remote failure
        raise LLMInvocationError(f"LLM call failed: {exc}") from exc

    final_text = extract_text_from_response(response)
    usage = extract_usage_metadata(response)
    return final_text, response, usage


def print_usage_metadata(response, show_usage: bool) -> None:
    if not show_usage:
        return
    usage = extract_usage_metadata(response)
    if not usage:
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
        final_text, response, _ = call_chat_model(
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
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print_mock_answer(results)
        return

    print("\n=== Final Answer ===\n")
    print(final_text or "I don't know.")
    print_usage_metadata(response, show_usage)


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
    parser.add_argument("--provider", default="openai", help="LLM provider identifier (default: openai)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model name to request (default: gpt-5-mini)")
    parser.add_argument("--api-key", dest="api_key", help="API key for the chosen provider (defaults to environment)")
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

    if args.agent_mode == "none":
        run_none_mode(results, args.k)
        return

    if args.agent_mode == "pretend":
        run_pretend_mode(args.question, results, args.k)
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    run_llm_mode(
        question=args.question,
        results=results,
        provider=args.provider,
        model_name=args.llm_model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        show_usage=args.show_usage,
    )


if __name__ == "__main__":
    main()
