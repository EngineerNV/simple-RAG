"""05_chat_cli.py — playful terminal chat interface backed by the RAG pipeline.

This script turns the retrieval pipeline into an interactive chat experience. It
keeps a running conversation, maintains a rolling summary via LangChain memory so
the LLM can carry context, and surfaces the retrieved snippets that ground each
answer.

The actual pipeline (topic gate -> decider -> retrieve -> rerank -> compose ->
LLM) lives in ``chat_engine.ChatEngine``, shared with ``06_chat_tui.py``. This
file only handles Rich-based terminal rendering and the REPL loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Sequence

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

from chat_engine import (
    CHROMA_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    ChatEngine,
    ChatEngineConfig,
    clean_snippet,
)

logger = logging.getLogger(__name__)

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal RAG chat playground")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Number of context chunks to retrieve per question")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model to load for retrieval")
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR), help="Path to the persisted Chroma directory")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model identifier for responses")
    parser.add_argument("--provider", default=None, help="LLM provider override (auto-detected from API keys if not specified)")
    parser.add_argument("--api-key", dest="api_key", help="Explicit API key override for the LLM provider")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens per LLM response")
    parser.add_argument("--base-url", dest="base_url", help="Optional base URL for OpenAI-compatible endpoints")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt that governs assistant behaviour")
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
        combined = None
        try:
            combined = getattr(doc, "metadata", {}).get("combined_score")
        except Exception:
            combined = None
        score_display = f"{score:.3f}" if score is not None else "n/a"
        if combined is not None:
            header = f"[{idx}] score={score_display} rerank={combined:.3f}\n{meta_bits}"
        else:
            header = f"[{idx}] score={score_display}\n{meta_bits}"
        table.add_row(header, textwrap.fill(snippet, width=80))
    console.print(table)


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

    config = ChatEngineConfig(
        persist_dir=Path(args.persist_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        provider=args.provider,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        retrieval_k=args.retrieval_k,
    )

    try:
        engine = ChatEngine(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    logger.debug(
        "Retrieval store initialised with embedding model '%s' at '%s'.", args.embedding_model, args.persist_dir
    )
    logger.debug("Using provider '%s' with model '%s'.", engine.provider, args.llm_model)

    console.print(
        Panel(
            Text(
                "Type your message to converse with the RAG assistant. Commands: /help, /exit, /reset, /showctx",
                style="cyan",
            ),
            title="simple-RAG chat",
        )
    )

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
                engine.reset()
                console.print("[green]Cleared conversation history.[/green]")
                continue
            if command == "showctx":
                show_context = not show_context
                console.print(f"[green]Context display {'enabled' if show_context else 'disabled'}.[/green]")
                continue
            console.print(f"[yellow]Unknown command '{command}'. Try /help.[/yellow]")
            continue

        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            turn = engine.process_turn(user_message)

        if not turn.ok:
            console.print(f"[red]LLM call failed: {turn.error}[/red]")
            # Don't record a broken turn; let the user retry without losing
            # the session (and without losing --save-transcript on exit).
            continue

        if turn.use_rag and show_context:
            render_context_table(turn.results, console)

        console.print(Panel(turn.answer, title="Assistant", style="green"))

    if args.transcript_path:
        try:
            engine.save_transcript(Path(args.transcript_path))
            console.print(f"[green]Transcript written to {args.transcript_path}[/green]")
        except Exception as exc:  # pragma: no cover - filesystem failure
            console.print(f"[red]Failed to save transcript: {exc}[/red]")


if __name__ == "__main__":
    main()
