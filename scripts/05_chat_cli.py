"""05_chat_cli.py — playful terminal chat interface backed by the RAG pipeline.

This script turns the retrieval pipeline into an interactive chat experience. It
keeps a running conversation, maintains a rolling summary via LangChain memory so
the LLM can carry context, and surfaces the retrieved snippets that ground each
answer. Per-turn orchestration (routing, retrieval, prompting) lives in
:mod:`agent_orchestration_helper`; this file only renders the conversation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # Optional dependency for convenient local development.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_orchestration_helper import (
    ChatSession,
    build_router,
    build_system_prompt,
)
from utils import build_rejection_writer, suppress_langchain_warnings
from utils import settings
from utils.chat_history import SummaryBufferHistory
from utils.llm import resolve_model, resolve_provider_and_key

query_module = import_module("scripts.02_query")

clean_snippet = query_module.clean_snippet  # type: ignore[attr-defined]
create_retrieval_store = query_module.create_retrieval_store  # type: ignore[attr-defined]
load_chat_model = query_module.load_chat_model  # type: ignore[attr-defined]
retrieve_contexts = query_module.retrieve_contexts  # type: ignore[attr-defined]

suppress_langchain_warnings()

logger = logging.getLogger(__name__)


DEFAULT_EMBED_MODEL = settings.DEFAULT_EMBED_MODEL
CHROMA_DIR = settings.CHROMA_DIR


@dataclass
class TranscriptLog:
    """Tracks retrieved contexts for each user turn."""

    user_contexts: List[Sequence[str]] = field(default_factory=list)

    def add_user_context(self, contexts: Sequence[str]) -> None:
        self.user_contexts.append(contexts)

    def reset(self) -> None:
        self.user_contexts.clear()

    def iter_user_contexts(self) -> Iterable[Sequence[str]]:
        return iter(self.user_contexts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal RAG chat playground")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Number of context chunks to retrieve per question")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model to load for retrieval")
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR), help="Path to the persisted Chroma directory")
    parser.add_argument("--llm-model", default=None, help="Chat model identifier (defaults to the resolved provider's default model)")
    parser.add_argument("--provider", default=None, help="LLM provider override (auto-detected from API keys if not specified)")
    parser.add_argument("--api-key", dest="api_key", help="Explicit API key override for the LLM provider")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for the chat model (omitted unless set)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens per LLM response")
    parser.add_argument("--base-url", dest="base_url", help="Optional base URL for OpenAI-compatible endpoints")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Replace the entire built-in system prompt (persona + grounding rules) with your own text",
    )
    parser.add_argument("--show-context", action="store_true", help="Display retrieved snippets for each answer")
    parser.add_argument("--save-transcript", dest="transcript_path", help="Optional file path to write the chat transcript on exit")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to stderr")
    return parser.parse_args(argv)


def render_context_table(results, console: Console) -> None:
    if not results:
        console.print("[yellow]No contexts retrieved for this turn.[/yellow]")
        return
    table = Table(box=None)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Snippet", style="white")
    for idx, (doc, score) in enumerate(results):
        snippet = clean_snippet(doc.page_content) or "(empty snippet)"
        meta = doc.metadata or {}
        meta_bits = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "no metadata"
        combined = meta.get("combined_score")
        if combined is not None:
            header = f"[{idx}] score={score:.3f} rerank={combined:.3f}\n{meta_bits}"
        else:
            header = f"[{idx}] score={score:.3f}\n{meta_bits}"
        table.add_row(header, textwrap.fill(snippet, width=80))
    console.print(table)


def save_transcript(
    history: SummaryBufferHistory,
    transcript_log: TranscriptLog,
    path: Path,
    console: Console,
) -> None:
    try:
        lines = []
        summary_text = history.moving_summary_buffer.strip()
        if summary_text:
            lines.append("# Conversation summary\n" + summary_text + "\n")

        user_contexts = list(transcript_log.iter_user_contexts())
        user_index = 0

        for message in history.raw_messages:
            if isinstance(message, HumanMessage):
                lines.append(f"## User\n{message.content}\n")
                if user_index < len(user_contexts) and user_contexts[user_index]:
                    lines.append("Retrieved contexts:\n")
                    for ctx in user_contexts[user_index]:
                        lines.append(f"- {ctx}")
                    lines.append("")
                user_index += 1
            elif isinstance(message, AIMessage):
                lines.append(f"## Assistant\n{message.content}\n")
            else:
                speaker = message.__class__.__name__
                lines.append(f"## {speaker}\n{message.content}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Transcript written to {path}[/green]")
    except Exception as exc:  # pragma: no cover - filesystem failure
        console.print(f"[red]Failed to save transcript: {exc}[/red]")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    console = Console()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.debug:
        logger.debug("Debug logging enabled.")

    try:
        _, store = create_retrieval_store(model_name=args.embedding_model, persist_dir=Path(args.persist_dir))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    logger.debug("Retrieval store initialised with embedding model '%s' at '%s'.", args.embedding_model, args.persist_dir)

    provider, api_key = resolve_provider_and_key(args.api_key, args.provider)

    if not api_key:
        console.print(
            "[red]No API key available. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable.[/red]"
        )
        sys.exit(1)

    llm_model = resolve_model(provider, args.llm_model)
    logger.debug("Using provider '%s' with model '%s'.", provider, llm_model)

    try:
        llm = load_chat_model(
            provider=provider,
            model_name=llm_model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    router_llm = build_router(llm)
    rejection_llm = build_rejection_writer(llm)

    history = SummaryBufferHistory(llm=llm, max_token_limit=max(args.max_tokens * 2, 1200))

    logger.debug("Conversation memory initialised with max_token_limit=%s.", max(args.max_tokens * 2, 1200))

    # The system prompt stays byte-identical across turns (good for provider
    # prompt caching); the retrieved evidence rides in the current human turn
    # only, while history stores just the user's actual message.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{context_block}{user_message}"),
        ]
    )
    response_chain = prompt | llm | StrOutputParser()
    chat_with_history = RunnableWithMessageHistory(
        response_chain,
        lambda _session_id: history,
        input_messages_key="user_message",
        history_messages_key="chat_history",
    )

    session = ChatSession(
        router_llm=router_llm,
        rejection_llm=rejection_llm,
        chat_with_history=chat_with_history,
        history_adapter=history,
        store=store,
        retrieve_contexts_fn=retrieve_contexts,
        system_prompt=args.system_prompt or build_system_prompt(),
        retrieval_k=args.retrieval_k,
    )

    console.print(
        Panel(
            Text(
                "Type your message to converse with the RAG assistant. Commands: /help, /exit, /reset, /showctx",
                style="cyan",
            ),
            title="simple-RAG chat",
        )
    )

    transcript_log = TranscriptLog()
    show_context = args.show_context

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Session ended.[/red]")
            break

        if not user_message.strip():
            continue

        logger.debug("Received user message: %s", user_message)

        if user_message.startswith("/"):
            command = user_message.lstrip("/").lower()
            if command in {"exit", "quit"}:
                break
            if command == "help":
                console.print("Commands: /help, /exit, /reset, /showctx (toggle context display)")
                continue
            if command == "reset":
                history.clear()
                transcript_log.reset()
                console.print("[green]Cleared conversation history.[/green]")
                continue
            if command == "showctx":
                show_context = not show_context
                console.print(f"[green]Context display {'enabled' if show_context else 'disabled'}.[/green]")
                continue
            console.print(f"[yellow]Unknown command '{command}'. Try /help.[/yellow]")
            continue

        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            result = session.handle_turn(user_message)

        if result.error:
            # The failed turn left no residue in history; keep the session alive.
            console.print(f"[red]LLM call failed: {result.error}[/red]")

        transcript_log.add_user_context(result.context_blocks)
        if result.used_rag and show_context:
            render_context_table(result.results, console)

        console.print(Panel(result.answer, title="Assistant", style="green"))

    if args.transcript_path:
        save_transcript(history, transcript_log, Path(args.transcript_path), console)


if __name__ == "__main__":
    main()
