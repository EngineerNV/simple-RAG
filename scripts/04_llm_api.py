"""04_llm_api.py — format prompts and execute LLM calls for quick smoke tests.

This utility loads API credentials, constructs an OpenAI-compatible chat client,
assembles a prompt from a question plus optional context snippets, and prints the
model response along with latency and token usage. It is useful for validating
provider access before wiring the call into other scripts.
"""

from __future__ import annotations

import argparse  # Parse CLI arguments for quick smoke tests
import json  # Allow CLI users to pass context snippets via JSON payloads
import time  # Capture latency metrics for reporting
from pathlib import Path  # Load context snippets from files during CLI tests
import sys
from typing import Iterable, Optional  # Type hints for prompt assembly and inputs

try:  # Optional dependency for local development convenience.
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from langchain_core.language_models.chat_models import (  # Provide a common interface for chat models
    BaseChatModel,
)
from langchain_core.messages import AIMessage  # Represent the structured response returned by chat models
from langchain_core.prompts import ChatPromptTemplate  # Build message templates for question + context prompts

from utils.llm import (
    load_chat_model,
    resolve_model,
    resolve_provider_and_key,
)
from utils.warnings_filter import suppress_langchain_warnings

suppress_langchain_warnings()


def build_llm_client(
    api_key: str,
    model_name: str,
    *,
    temperature: float | None = None,
    max_tokens: int = 2000,
    base_url: Optional[str] = None,
    provider: str = "openai",
) -> BaseChatModel:
    """Instantiate the provider's chat client (thin wrapper over utils.llm)."""

    return load_chat_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def assemble_prompt(
    question: str,
    contexts: Iterable[str],
    instructions: Optional[str] = None,
) -> ChatPromptTemplate:
    """Combine the question, supporting contexts, and optional system message."""

    system_text = instructions or (
        "You are a retrieval-augmented assistant. Answer using the provided context. "
        "If the answer is not contained within the context, reply with 'I don't know'."
    )
    context_lines = [chunk.strip() for chunk in contexts if chunk and chunk.strip()]
    context_block = "\n\n".join(context_lines) if context_lines else "No supporting context provided."

    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )
    return template.partial(context=context_block, question=question)


def call_llm(prompt: ChatPromptTemplate, client: BaseChatModel) -> tuple[AIMessage, float]:
    """Send the prompt to the configured client and return the structured response."""

    messages = prompt.format_messages()
    start = time.perf_counter()
    response = client.invoke(messages)
    latency = time.perf_counter() - start
    if not isinstance(response, AIMessage):
        raise RuntimeError(f"Expected an AIMessage from the chat model, got {type(response)!r} instead.")
    return response, latency


def pretty_print(result: AIMessage, latency_s: float) -> None:
    """Emit the model response with any formatting that helps debugging."""

    content = result.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part)
    else:
        text = str(content)

    print("\nAssistant:\n")
    print(text.strip() or "(empty response)")
    print(f"\nLatency: {latency_s:.2f}s")

    metadata = getattr(result, "response_metadata", {}) or {}
    usage = metadata.get("token_usage") or metadata.get("usage")
    if usage:
        print("Token usage:")
        for key, value in usage.items():
            print(f"- {key}: {value}")


def load_context_from_file(path: Path) -> Iterable[str]:
    """Utility for CLI usage: load context snippets from a JSON or text file."""

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict) and "contexts" in payload:
            payload = payload["contexts"]
        if not isinstance(payload, list):
            raise ValueError("JSON context file must contain a list of strings or a 'contexts' list.")
        return [str(item) for item in payload]

    # Treat everything else as newline-delimited snippets separated by blank lines.
    chunks = [chunk.strip() for chunk in text.split("\n\n")]
    return [chunk for chunk in chunks if chunk]


def main() -> None:
    """Simple CLI hook for manual experimentation."""

    load_dotenv()

    parser = argparse.ArgumentParser(description="Call the configured chat model with optional context snippets")
    parser.add_argument("--question", required=True, help="Question to send to the chat model")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Additional context snippets (can be repeated)",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Path to a file containing context snippets (JSON list or text separated by blank lines)",
    )
    parser.add_argument("--instructions", help="Override the system prompt")
    parser.add_argument("--api-key", dest="api_key", help="Explicit API key override (auto-detected if not specified)")
    parser.add_argument("--model", default=None, help="Chat model identifier (defaults to the resolved provider's default model)")
    parser.add_argument("--provider", default=None, help="LLM provider override (auto-detected from API keys if not specified)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (omitted unless set)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for the response")
    parser.add_argument("--base-url", dest="base_url", help="Optional custom base URL for OpenAI-compatible APIs")
    args = parser.parse_args()

    contexts: list[str] = list(args.context)
    if args.context_file:
        contexts.extend(load_context_from_file(args.context_file))

    provider, api_key = resolve_provider_and_key(args.api_key, args.provider)

    if not api_key:
        print("[ERROR] No API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = build_llm_client(
        api_key=api_key,
        model_name=resolve_model(provider, args.model),
        provider=provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )

    prompt = assemble_prompt(args.question, contexts, instructions=args.instructions)
    response, latency = call_llm(prompt, client)
    pretty_print(response, latency)


if __name__ == "__main__":
    main()
