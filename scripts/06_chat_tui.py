"""06_chat_tui.py — Textual TUI for the RAG chat pipeline.

Same pipeline as scripts/05_chat_cli.py (topic gate -> decider -> retrieve ->
rerank -> compose -> LLM), shared via chat_engine.ChatEngine -- no pipeline
logic is duplicated here. This file only implements the Textual widgets/app
and the background-thread plumbing needed to keep the UI responsive while a
turn (which makes blocking LLM calls) runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
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

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, RichLog

from chat_engine import (
    CHROMA_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    ChatEngine,
    ChatEngineConfig,
    TurnResult,
    clean_snippet,
)

logger = logging.getLogger(__name__)

# Suppress noisy deprecation warnings without changing packages.
try:  # Best-effort: some environments provide this warning class
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", message=r".*HuggingFaceEmbeddings.*was deprecated.*")
    warnings.filterwarnings("ignore", message=r".*manual persistence method is no longer supported.*")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Textual RAG chat TUI")
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
    parser.add_argument("--save-transcript", dest="transcript_path", help="Optional file path to write the chat transcript on exit")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging (written to a log file, not the TUI)")
    parser.add_argument("--semantic-cache-size", type=int, default=20, help="Max entries in the semantic result cache (LRU)")
    parser.add_argument("--semantic-cache-threshold", type=float, default=0.93, help="Cosine similarity required for a semantic cache hit")
    parser.add_argument("--no-semantic-cache", dest="enable_semantic_cache", action="store_false", help="Disable the semantic result cache")
    return parser.parse_args(argv)


def format_context_panel(turn: TurnResult) -> str:
    """Render a turn's retrieved contexts for the sidebar RichLog."""

    if not turn.use_rag or not turn.results:
        return "[dim]No contexts retrieved for this turn.[/dim]"
    lines: list[str] = []
    for idx, (doc, score) in enumerate(turn.results):
        snippet = clean_snippet(doc.page_content) or "(empty snippet)"
        combined = (getattr(doc, "metadata", {}) or {}).get("combined_score")
        score_display = f"{score:.3f}" if score is not None else "n/a"
        header = f"[cyan][{idx}][/cyan] score={score_display}"
        if combined is not None:
            header += f" rerank={combined:.3f}"
        lines.append(f"{header}\n{snippet}")
    return "\n\n".join(lines)


class ChatTUI(App):
    """A scrollable chat log, an input box, and a context sidebar."""

    CSS = """
    #chat-log {
        width: 2fr;
        border: round $accent;
        padding: 0 1;
    }
    #context-panel {
        width: 1fr;
        border: round $accent;
        padding: 0 1;
    }
    #chat-input {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+r", "reset_chat", "Reset"),
        ("ctrl+s", "save_transcript", "Save transcript"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, engine: ChatEngine, transcript_path: str | None = None):
        super().__init__()
        self.engine = engine
        self.transcript_path = transcript_path

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False, auto_scroll=True)
            with VerticalScroll(id="context-panel"):
                yield RichLog(id="context-log", markup=True, wrap=True, highlight=False)
        yield Input(placeholder="Type a message… (/reset, /exit, ctrl+s to save)", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-log", RichLog).write(
            "[cyan]Type your message to converse with the RAG assistant. "
            "Commands: /exit, /reset. Ctrl+S saves the transcript.[/cyan]"
        )
        self.query_one("#context-log", RichLog).write("[dim]Retrieved context for the latest turn appears here.[/dim]")
        self.query_one("#chat-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        event.input.value = ""
        if not message:
            return

        lowered = message.lower()
        if lowered in {"/exit", "/quit"}:
            self.exit()
            return
        if lowered == "/reset":
            self.action_reset_chat()
            return

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(f"\n[bold cyan]You:[/bold cyan] {message}")
        event.input.disabled = True
        self._process_turn_worker(message)

    @work(thread=True)
    def _process_turn_worker(self, message: str) -> None:
        """Runs ChatEngine.process_turn (blocking LLM calls) off the UI thread."""

        turn = self.engine.process_turn(message)
        self.call_from_thread(self._display_turn, turn)

    def _display_turn(self, turn: TurnResult) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        if not turn.ok:
            chat_log.write(f"[red]LLM call failed: {turn.error}[/red]")
        else:
            chat_log.write(f"[bold green]Assistant:[/bold green] {turn.answer}")
            context_log = self.query_one("#context-log", RichLog)
            context_log.clear()
            context_log.write(format_context_panel(turn))

        chat_input = self.query_one("#chat-input", Input)
        chat_input.disabled = False
        chat_input.focus()

    def action_reset_chat(self) -> None:
        self.engine.reset()
        self.query_one("#chat-log", RichLog).write("[green]Cleared conversation history.[/green]")
        context_log = self.query_one("#context-log", RichLog)
        context_log.clear()
        context_log.write("[dim]Retrieved context for the latest turn appears here.[/dim]")

    def action_save_transcript(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        if not self.transcript_path:
            chat_log.write("[yellow]No --save-transcript path was given at startup.[/yellow]")
            return
        try:
            self.engine.save_transcript(Path(self.transcript_path))
            chat_log.write(f"[green]Transcript written to {self.transcript_path}[/green]")
        except Exception as exc:  # pragma: no cover - filesystem failure
            chat_log.write(f"[red]Failed to save transcript: {exc}[/red]")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename="chat_tui.log" if args.debug else None,
    )

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
        enable_semantic_cache=args.enable_semantic_cache,
        semantic_cache_size=args.semantic_cache_size,
        semantic_cache_similarity_threshold=args.semantic_cache_threshold,
    )

    try:
        engine = ChatEngine(config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    app = ChatTUI(engine, transcript_path=args.transcript_path)
    app.run()

    if args.transcript_path:
        try:
            engine.save_transcript(Path(args.transcript_path))
            print(f"Transcript written to {args.transcript_path}")
        except Exception as exc:  # pragma: no cover - filesystem failure
            print(f"Failed to save transcript: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
