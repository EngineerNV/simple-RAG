"""05_chat_cli.py — playful terminal chat interface backed by the RAG pipeline.

This script turns the retrieval pipeline into an interactive chat experience. It
keeps a running conversation, periodically summarises the dialogue so the LLM can
carry context, and surfaces the retrieved snippets that ground each answer.
"""

from __future__ import annotations

import argparse
import os
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

query_module = import_module("scripts.02_query")

LLMInvocationError = query_module.LLMInvocationError  # type: ignore[attr-defined]
MissingAPIKeyError = query_module.MissingAPIKeyError  # type: ignore[attr-defined]
call_chat_model = query_module.call_chat_model  # type: ignore[attr-defined]
clean_snippet = query_module.clean_snippet  # type: ignore[attr-defined]
compose_messages = query_module.compose_messages  # type: ignore[attr-defined]
create_retrieval_store = query_module.create_retrieval_store  # type: ignore[attr-defined]
retrieve_contexts = query_module.retrieve_contexts  # type: ignore[attr-defined]


DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly research assistant. Use the retrieved knowledge snippets "
    "to answer the user, cite sources as [source #], keep the tone conversational, "
    "and acknowledge when information is missing."
)
SUMMARY_SYSTEM_PROMPT = (
    "Summarise the ongoing conversation between the assistant and the user. "
    "Capture key topics, decisions, and follow-ups in 3 bullet points or fewer."
)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5-mini"
CHROMA_DIR = Path("data") / "chroma"


@dataclass
class ConversationTurn:
    speaker: str
    content: str
    contexts: Sequence[str] | None = None


@dataclass
class ConversationState:
    summary_every: int
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    user_turns_since_summary: int = 0

    def add_user_turn(self, message: str, contexts: Sequence[str]) -> None:
        self.turns.append(ConversationTurn("user", message, contexts))
        self.user_turns_since_summary += 1

    def add_assistant_turn(self, message: str) -> None:
        self.turns.append(ConversationTurn("assistant", message))

    def reset(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.user_turns_since_summary = 0

    def needs_summary(self) -> bool:
        if self.summary_every <= 0:
            return False
        return self.user_turns_since_summary >= self.summary_every

    def mark_summarised(self, new_summary: str) -> None:
        self.summary = new_summary.strip()
        self.user_turns_since_summary = 0

    def recent_dialogue(self, window: int = 4) -> List[str]:
        excerpt: List[str] = []
        for turn in self.turns[-window:]:
            prefix = "User" if turn.speaker == "user" else "Assistant"
            excerpt.append(f"{prefix}: {turn.content}")
        return excerpt

    def transcript_text(self) -> str:
        lines = []
        for turn in self.turns:
            prefix = "User" if turn.speaker == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal RAG chat playground")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Number of context chunks to retrieve per question")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model to load for retrieval")
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR), help="Path to the persisted Chroma directory")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model identifier for responses")
    parser.add_argument("--provider", default="openai", help="LLM provider identifier")
    parser.add_argument("--api-key", dest="api_key", help="Explicit API key override for the LLM provider")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    parser.add_argument("--max-tokens", type=int, default=700, help="Maximum tokens per LLM response")
    parser.add_argument("--base-url", dest="base_url", help="Optional base URL for OpenAI-compatible endpoints")
    parser.add_argument("--summary-every", type=int, default=5, help="Summarise the chat after this many user turns (0 disables)")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt that governs assistant behaviour")
    parser.add_argument("--show-context", action="store_true", help="Display retrieved snippets for each answer")
    parser.add_argument("--save-transcript", dest="transcript_path", help="Optional file path to write the chat transcript on exit")
    return parser.parse_args(argv)


def resolve_api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("RAG_LLM_API_KEY")


def format_contexts(results) -> List[str]:
    formatted: List[str] = []
    for idx, (doc, score) in enumerate(results):
        snippet = clean_snippet(doc.page_content)
        source_label = f"[source {idx}] score={score:.3f}"
        if snippet:
            formatted.append(f"{source_label}\n{snippet}")
        else:
            formatted.append(f"{source_label}\n(empty snippet)")
    return formatted


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
        header = f"[{idx}] score={score:.3f}\n{meta_bits}"
        table.add_row(header, textwrap.fill(snippet, width=80))
    console.print(table)


def build_question_payload(user_message: str, state: ConversationState) -> str:
    sections: List[str] = []
    if state.summary:
        sections.append("Conversation summary so far:\n" + state.summary)
    recent_dialogue = state.recent_dialogue()
    if recent_dialogue:
        sections.append("Recent exchanges:\n" + "\n".join(recent_dialogue))
    sections.append("Current user message:\n" + user_message)
    return "\n\n".join(sections)


def summarise_history(state: ConversationState, console: Console, provider: str, model_name: str,
                      api_key: str | None, temperature: float, max_tokens: int, base_url: str | None) -> str:
    transcript = state.transcript_text()
    if not transcript.strip():
        return state.summary

    question = "Please produce an updated running summary of the conversation so far."
    instructions = SUMMARY_SYSTEM_PROMPT

    try:
        messages = compose_messages(question, [], system_prompt=SUMMARY_SYSTEM_PROMPT)
        # Override the human message content with the summary-specific payload.
        messages[-1].content = f"{instructions}\n\nConversation transcript:\n{transcript}"
        summary_text, _, _ = call_chat_model(
            messages=messages,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        return summary_text or state.summary
    except MissingAPIKeyError:
        console.print("[yellow]No API key available; storing a compact manual summary instead.[/yellow]")
    except (LLMInvocationError, RuntimeError) as exc:
        console.print(f"[yellow]Summary generation failed ({exc}). Using heuristic fallback.[/yellow]")

    # Fallback heuristic: keep the last few turns as a lightweight summary.
    recent = " ".join(state.recent_dialogue(window=6))
    return textwrap.shorten(recent, width=320, placeholder="…")


def save_transcript(state: ConversationState, path: Path, console: Console) -> None:
    try:
        lines = []
        if state.summary:
            lines.append("# Conversation summary\n" + state.summary + "\n")
        for turn in state.turns:
            prefix = "User" if turn.speaker == "user" else "Assistant"
            lines.append(f"## {prefix}\n{turn.content}\n")
            if turn.contexts:
                lines.append("Retrieved contexts:\n")
                for ctx in turn.contexts:
                    lines.append(f"- {ctx}")
                lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Transcript written to {path}[/green]")
    except Exception as exc:  # pragma: no cover - filesystem failure
        console.print(f"[red]Failed to save transcript: {exc}[/red]")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    console = Console()

    try:
        _, store = create_retrieval_store(model_name=args.embedding_model, persist_dir=Path(args.persist_dir))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    api_key = resolve_api_key(args.api_key)
    provider = args.provider

    if not api_key:
        console.print(
            "[red]No API key available. Set --api-key or the OPENAI_API_KEY/RAG_LLM_API_KEY environment variable.[/red]"
        )
        sys.exit(1)

    console.print(
        Panel(
            Text(
                "Type your message to converse with the RAG assistant. Commands: /help, /exit, /reset, /summary, /showctx",
                style="cyan",
            ),
            title="simple-RAG chat",
        )
    )

    state = ConversationState(summary_every=args.summary_every)
    show_context = args.show_context

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Session ended.[/red]")
            break

        if not user_message.strip():
            continue

        if user_message.startswith("/"):
            command = user_message.lstrip("/").lower()
            if command in {"exit", "quit"}:
                break
            if command == "help":
                console.print(
                    "Commands: /help, /exit, /reset, /summary, /showctx (toggle context display)"
                )
                continue
            if command == "reset":
                state.reset()
                console.print("[green]Cleared conversation history.[/green]")
                continue
            if command == "summary":
                if state.summary:
                    console.print(Panel(state.summary, title="Conversation summary", style="magenta"))
                else:
                    console.print("[yellow]No summary generated yet.[/yellow]")
                continue
            if command == "showctx":
                show_context = not show_context
                console.print(f"[green]Context display {'enabled' if show_context else 'disabled'}.[/green]")
                continue
            console.print(f"[yellow]Unknown command '{command}'. Try /help.[/yellow]")
            continue

        results = retrieve_contexts(store, user_message, args.retrieval_k)
        context_blocks = format_contexts(results)
        state.add_user_turn(user_message, context_blocks)

        if show_context:
            render_context_table(results, console)

        question_payload = build_question_payload(user_message, state)
        try:
            messages = compose_messages(question_payload, results, system_prompt=args.system_prompt)
            answer, raw_response, _ = call_chat_model(
                messages=messages,
                provider=provider,
                model_name=args.llm_model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
            )
        except MissingAPIKeyError:
            console.print(
                "[red]No API key available during response generation. "
                "Set --api-key or configure OPENAI_API_KEY/RAG_LLM_API_KEY.[/red]"
            )
            sys.exit(1)
        except (LLMInvocationError, RuntimeError) as exc:
            console.print(f"[red]LLM call failed: {exc}[/red]")
            sys.exit(1)

        state.add_assistant_turn(answer)
        console.print(Panel(answer, title="Assistant", style="green"))

        if state.needs_summary():
            summary_text = summarise_history(
                state,
                console,
                provider=provider,
                model_name=args.llm_model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
            )
            state.mark_summarised(summary_text)
            console.print(Panel(summary_text, title="Updated summary", style="magenta"))

    if args.transcript_path:
        save_transcript(state, Path(args.transcript_path), console)


if __name__ == "__main__":
    main()
