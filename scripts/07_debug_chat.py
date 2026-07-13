"""07_debug_chat.py — debug chat TUI with live RAG pipeline metrics.

Same RAG session as ``05_chat_cli.py`` (built through its shared
``create_chat_runtime`` factory) but renders a two-pane Rich layout after each
turn: the left pane shows the rolling conversation and the right pane shows
router decisions, per-chunk retrieval scores, per-stage timing, and memory
stats.

Usage::

    python scripts/07_debug_chat.py --help
    python scripts/07_debug_chat.py                  # auto-detect provider
    python scripts/07_debug_chat.py --provider gemini --chat-lines 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from agent_orchestration_helper import TurnResult
from utils import settings
from utils.chat_history import SummaryBufferHistory

chat_cli_module = import_module("scripts.05_chat_cli")

TranscriptLog = chat_cli_module.TranscriptLog  # type: ignore[attr-defined]
create_chat_runtime = chat_cli_module.create_chat_runtime  # type: ignore[attr-defined]
render_context_table = chat_cli_module.render_context_table  # type: ignore[attr-defined]
save_transcript = chat_cli_module.save_transcript  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = settings.DEFAULT_EMBED_MODEL
CHROMA_DIR = settings.CHROMA_DIR

_CHAT_WRAP = 64  # text wrap width inside the chat panel


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class DebugChatTurn:
    user_message: str
    answer: str
    result: TurnResult


@dataclass
class DebugState:
    """Every turn (including error turns) plus the metrics of the latest one."""

    turns: List[DebugChatTurn] = field(default_factory=list)

    def add_turn(self, user_message: str, answer: str, result: TurnResult) -> None:
        self.turns.append(DebugChatTurn(user_message, answer, result))

    @property
    def last_result(self) -> Optional[TurnResult]:
        return self.turns[-1].result if self.turns else None


# ── panel builders ─────────────────────────────────────────────────────────────

def _ms(val: float) -> str:
    return f"{val:,.0f} ms"


def build_metrics_panel(result: Optional[TurnResult], history: SummaryBufferHistory) -> Panel:
    """Right panel: router, retrieval, timing, memory."""
    if result is None:
        return Panel(
            Text("─ waiting for first turn ─", style="dim"),
            title="[bold]Pipeline Metrics[/bold]",
            border_style="bright_black",
        )

    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column()

    # ── Router ──────────────────────────────────────────────────────────────
    t.add_row(Text("ROUTER", style="bold yellow underline"), "")
    route = result.route
    if route is not None:
        on_icon = "[green]✓[/green]" if route.on_topic else "[red]✗[/red]"
        rag_icon = "[green]✓[/green]" if route.use_rag else "[red]✗[/red]"
        t.add_row("on_topic", on_icon)
        t.add_row("use_rag", rag_icon)
        t.add_row("confidence", f"{route.confidence:.2f}")
        if route.search_query:
            t.add_row("query", textwrap.shorten(route.search_query, 34, placeholder="…"))
        if route.keywords:
            t.add_row("keywords", ", ".join(route.keywords[:4]))
        t.add_row("reason", textwrap.shorten(route.reason, 34, placeholder="…"))
    else:
        t.add_row(Text("(no route)", style="dim"), "")

    # ── Retrieval ────────────────────────────────────────────────────────────
    t.add_row("", "")
    chunk_count = len(result.results)
    t.add_row(Text(f"RETRIEVAL  k={chunk_count}", style="bold yellow underline"), "")
    if result.results:
        chunk_table = Table(box=None, show_header=True, padding=(0, 1))
        chunk_table.add_column("#", style="dim", width=2)
        chunk_table.add_column("source", style="cyan", max_width=14, no_wrap=True)
        chunk_table.add_column("score", justify="right", width=5)
        chunk_table.add_column("rerank", justify="right", width=6)
        chunk_table.add_column("lex", justify="right", width=4)
        for idx, (doc, score) in enumerate(result.results):
            meta = getattr(doc, "metadata", {}) or {}
            source = Path(meta.get("source", "?")).name
            combined = meta.get("combined_score")
            lexical = meta.get("lexical_overlap")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            rerank_str = f"{combined:.3f}" if isinstance(combined, (int, float)) else "—"
            lex_str = f"{lexical:.2f}" if isinstance(lexical, (int, float)) else "—"
            chunk_table.add_row(str(idx), source, score_str, rerank_str, lex_str)
        t.add_row("", chunk_table)
    elif route is not None and route.use_rag:
        # RAG was attempted but retrieval came back empty (result.used_rag is
        # False in that case, so check the route decision itself).
        t.add_row(Text("(no chunks returned)", style="dim"), "")
    else:
        t.add_row(Text("(RAG not used)", style="dim"), "")

    # ── Timing ───────────────────────────────────────────────────────────────
    t.add_row("", "")
    t.add_row(Text("TIMING", style="bold yellow underline"), "")
    timings = result.timings
    if timings:
        if "router_ms" in timings:
            t.add_row("router", _ms(timings["router_ms"]))
        if timings.get("retrieval_ms", 0.0) > 0:
            t.add_row("retrieval", _ms(timings["retrieval_ms"]))
        if "llm_ms" in timings:
            t.add_row("llm", _ms(timings["llm_ms"]))
        if "total_ms" in timings:
            t.add_row("total", f"[bold]{_ms(timings['total_ms'])}[/bold]")
    else:
        t.add_row(Text("(no timing data)", style="dim"), "")

    if result.error:
        t.add_row("", "")
        t.add_row(Text("ERROR", style="bold red"), Text(textwrap.shorten(result.error, 36, placeholder="…"), style="red"))

    # ── Memory ───────────────────────────────────────────────────────────────
    t.add_row("", "")
    t.add_row(Text("MEMORY", style="bold yellow underline"), "")
    raw = getattr(history, "raw_messages", []) or []
    summary = getattr(history, "moving_summary_buffer", "") or ""
    t.add_row("turns", str(len(raw) // 2))
    t.add_row("raw msgs", str(len(raw)))
    t.add_row("summary", f"{len(summary)} chars")

    return Panel(t, title="[bold]Pipeline Metrics[/bold]", border_style="bright_black")


def build_chat_panel(turns: List[DebugChatTurn], max_lines: int = 40) -> Panel:
    """Left panel: rolling conversation history (error turns shown in red)."""
    if not turns:
        return Panel(
            Text("Chat will appear here. Type a message below.", style="dim"),
            title="[bold]Conversation[/bold]",
            border_style="bright_black",
        )

    lines: List[str] = []
    for turn in turns:
        you_lines = textwrap.wrap(turn.user_message, width=_CHAT_WRAP) or [""]
        asst_lines = textwrap.wrap(turn.answer, width=_CHAT_WRAP) or [""]
        asst_style = "bold red" if turn.result.error else "bold green"
        lines.append(f"[bold cyan]You:[/bold cyan] {you_lines[0]}")
        for l in you_lines[1:]:
            lines.append(f"     {l}")
        lines.append(f"[{asst_style}]Asst:[/{asst_style}] {asst_lines[0]}")
        for l in asst_lines[1:]:
            lines.append(f"      {l}")
        lines.append("")

    # Trim to the most recent max_lines
    visible = lines[-max_lines:] if len(lines) > max_lines else lines

    return Panel(
        Text.from_markup("\n".join(visible)),
        title="[bold]Conversation[/bold]",
        border_style="bright_black",
    )


def render_screen(
    console: Console,
    chat_panel: Panel,
    metrics_panel: Panel,
    status_line: str = "",
) -> None:
    console.clear()
    if status_line:
        console.print(Text(status_line, style="dim cyan"))
    layout = Layout()
    layout.split_row(
        Layout(chat_panel, name="chat", ratio=3),
        Layout(metrics_panel, name="metrics", ratio=2),
    )
    console.print(layout)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug chat TUI with RAG pipeline metrics")
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
    parser.add_argument("--chat-lines", type=int, default=40, help="Max lines of chat history visible in the left panel")
    parser.add_argument("--save-transcript", dest="transcript_path", help="Optional file path to write the chat transcript on exit")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to stderr")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    console = Console()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    runtime = create_chat_runtime(args, console)
    session = runtime.session
    history = runtime.history

    state = DebugState()
    transcript_log = TranscriptLog()
    # Keep provider/model visible even after the layout repaints the screen.
    status_line = (
        f"simple-RAG debug  |  provider: {runtime.provider}  model: {runtime.llm_model}"
        f"  k={args.retrieval_k}  |  /help /exit /reset /showctx"
    )

    console.print(
        Panel(
            Text(
                f"Debug chat TUI  |  provider: {runtime.provider}  model: {runtime.llm_model}  k={args.retrieval_k}\n"
                "Commands: /help  /exit  /reset  /showctx",
                style="cyan",
            ),
            title="[bold]simple-RAG debug[/bold]",
        )
    )

    show_full_ctx = False

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Session ended.[/red]")
            break

        if not user_message.strip():
            continue

        if user_message.startswith("/"):
            # Strip exactly one slash so "//exit" stays an unknown command.
            cmd = user_message[1:].lower()
            if cmd in {"exit", "quit"}:
                break
            if cmd == "help":
                console.print("Commands: /help  /exit  /reset  /showctx (toggle full context table)")
                continue
            if cmd == "reset":
                history.clear()
                state = DebugState()
                transcript_log.reset()
                console.print("[green]Cleared.[/green]")
                continue
            if cmd == "showctx":
                show_full_ctx = not show_full_ctx
                console.print(f"[green]Full context table {'on' if show_full_ctx else 'off'}.[/green]")
                continue
            console.print(f"[yellow]Unknown command '{cmd}'. Try /help.[/yellow]")
            continue

        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            result = session.handle_turn(user_message)

        # Error turns are shown too (in red) so the user always sees a
        # response and the transcript reflects the whole conversation.
        state.add_turn(user_message, result.answer, result)
        transcript_log.add_turn(user_message, result.answer, result.context_blocks)

        render_screen(
            console,
            build_chat_panel(state.turns, max_lines=args.chat_lines),
            build_metrics_panel(state.last_result, history),
            status_line=status_line,
        )

        if show_full_ctx and result.results:
            render_context_table(result.results, console)

    if args.transcript_path:
        save_transcript(history, transcript_log, Path(args.transcript_path), console)


if __name__ == "__main__":
    main()
