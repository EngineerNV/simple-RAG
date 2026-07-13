"""07_debug_chat.py — debug chat TUI with live RAG pipeline metrics.

Same RAG session as ``05_chat_cli.py`` but renders a two-pane Rich layout after
each turn: the left pane shows the rolling conversation and the right pane shows
router decisions, per-chunk retrieval scores, per-stage timing, and memory stats.

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
from typing import List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_orchestration_helper import (
    ChatSession,
    TurnResult,
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

# ── width constants ────────────────────────────────────────────────────────────

_METRICS_WIDTH = 44   # right panel character width (approx)
_CHAT_WRAP = 64       # text wrap width inside the chat panel


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class DebugChatTurn:
    user_message: str
    answer: str
    result: TurnResult


@dataclass
class DebugState:
    turns: List[DebugChatTurn] = field(default_factory=list)
    last_result: Optional[TurnResult] = None

    def add_turn(self, user_message: str, answer: str, result: TurnResult) -> None:
        self.turns.append(DebugChatTurn(user_message, answer, result))
        self.last_result = result


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
    elif result.used_rag:
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
    raw = list(getattr(history, "raw_messages", []))
    summary = getattr(history, "moving_summary_buffer", "") or ""
    turn_count = len(raw) // 2
    t.add_row("turns", str(turn_count))
    t.add_row("raw msgs", str(len(raw)))
    t.add_row("summary", f"{len(summary)} chars")

    return Panel(t, title="[bold]Pipeline Metrics[/bold]", border_style="bright_black")


def build_chat_panel(turns: List[DebugChatTurn], max_lines: int = 40) -> Panel:
    """Left panel: rolling conversation history."""
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
        lines.append(f"[bold cyan]You:[/bold cyan] {you_lines[0]}")
        for l in you_lines[1:]:
            lines.append(f"     {l}")
        lines.append(f"[bold green]Asst:[/bold green] {asst_lines[0]}")
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
) -> None:
    console.clear()
    layout = Layout()
    layout.split_row(
        Layout(chat_panel, name="chat", ratio=3),
        Layout(metrics_panel, name="metrics", ratio=2),
    )
    console.print(layout)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug chat TUI with RAG pipeline metrics")
    parser.add_argument("--retrieval-k", type=int, default=3)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR))
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-key", dest="api_key")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-lines", type=int, default=40, help="Max lines of chat history visible in left panel")
    parser.add_argument("--save-transcript", dest="transcript_path")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    console = Console()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        _, store = create_retrieval_store(model_name=args.embedding_model, persist_dir=Path(args.persist_dir))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    provider, api_key = resolve_provider_and_key(args.api_key, args.provider)
    if not api_key:
        console.print("[red]No API key. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY.[/red]")
        sys.exit(1)

    llm_model = resolve_model(provider, args.llm_model)

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

    state = DebugState()

    # Initial screen: show empty panels with welcome text
    console.print(
        Panel(
            Text(
                f"Debug chat TUI  |  provider: {provider}  model: {llm_model}  k={args.retrieval_k}\n"
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
            cmd = user_message.lstrip("/").lower()
            if cmd in {"exit", "quit"}:
                break
            if cmd == "help":
                console.print("Commands: /help  /exit  /reset  /showctx (toggle full context table)")
                continue
            if cmd == "reset":
                history.clear()
                state = DebugState()
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

        answer = result.answer
        if not result.error:
            state.add_turn(user_message, answer, result)
        else:
            # Still show the error turn in the metrics panel
            state.last_result = result

        render_screen(
            console,
            build_chat_panel(state.turns, max_lines=args.chat_lines),
            build_metrics_panel(state.last_result, history),
        )

        if show_full_ctx and result.results:
            _render_full_ctx(result.results, console)

    if args.transcript_path:
        _save_transcript(history, state, Path(args.transcript_path), console)


def _render_full_ctx(results: Sequence[Tuple], console: Console) -> None:
    table = Table(title="Retrieved chunks", box=None)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Snippet", style="white")
    for idx, (doc, score) in enumerate(results):
        snippet = clean_snippet(doc.page_content) or "(empty)"
        meta = doc.metadata or {}
        meta_bits = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "no metadata"
        combined = meta.get("combined_score")
        header = (
            f"[{idx}] score={score:.3f} rerank={combined:.3f}\n{meta_bits}"
            if combined is not None
            else f"[{idx}] score={score:.3f}\n{meta_bits}"
        )
        table.add_row(header, textwrap.fill(snippet, width=80))
    console.print(table)


def _save_transcript(
    history: SummaryBufferHistory,
    state: DebugState,
    path: Path,
    console: Console,
) -> None:
    try:
        lines = []
        summary_text = history.moving_summary_buffer.strip()
        if summary_text:
            lines.append("# Conversation summary\n" + summary_text + "\n")
        for turn in state.turns:
            lines.append(f"## User\n{turn.user_message}\n")
            lines.append(f"## Assistant\n{turn.answer}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Transcript written to {path}[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to save transcript: {exc}[/red]")


if __name__ == "__main__":
    main()
