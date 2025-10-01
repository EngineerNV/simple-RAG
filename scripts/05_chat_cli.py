"""05_chat_cli.py — playful terminal chat interface backed by the RAG pipeline.

This script turns the retrieval pipeline into an interactive chat experience. It
keeps a running conversation, maintains a rolling summary via LangChain memory so
the LLM can carry context, and surfaces the retrieved snippets that ground each
answer.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Any

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

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core._api.deprecation import LangChainDeprecationWarning

from agent_orchestration_helper import (
    RAG_TOPIC_INVENTORY,
    SPECIALIZATION_LIST,
    build_decider,
    decide_use_rag,
    build_rewriter,
    apply_rewrite_and_retrieve,
    build_user_payload,
)
from utils import (
    build_rejection_writer,
    build_topic_guard,
    generate_rejection,
    topic_gate,
)

query_module = import_module("scripts.02_query")

clean_snippet = query_module.clean_snippet  # type: ignore[attr-defined]
create_retrieval_store = query_module.create_retrieval_store  # type: ignore[attr-defined]
load_chat_model = query_module.load_chat_model  # type: ignore[attr-defined]
compose_user_prompt = query_module.compose_user_prompt  # type: ignore[attr-defined]
retrieve_contexts = query_module.retrieve_contexts  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


DEFAULT_SYSTEM_PROMPT = (
    "Stay in expert research assistant mode. Follow the persona and context provided in the user message."
)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5-mini"
CHROMA_DIR = Path("data") / "chroma"


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


class SummaryHistoryAdapter(BaseChatMessageHistory):
    """Adapts ``ConversationSummaryBufferMemory`` for runnable message history."""

    def __init__(self, memory: ConversationSummaryBufferMemory):
        self._memory = memory

    @property
    def memory(self) -> ConversationSummaryBufferMemory:
        return self._memory

    @property
    def messages(self) -> List[BaseMessage]:
        stored = self._memory.load_memory_variables({}).get(self._memory.memory_key, [])
        return list(stored) if isinstance(stored, Iterable) else []

    def add_message(self, message: BaseMessage) -> None:
        self._memory.chat_memory.add_message(message)
        self._memory.prune()

    def add_messages(self, messages: Iterable[BaseMessage]) -> None:
        for message in messages:
            self._memory.chat_memory.add_message(message)
        self._memory.prune()

    def clear(self) -> None:
        self._memory.clear()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal RAG chat playground")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Number of context chunks to retrieve per question")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model to load for retrieval")
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR), help="Path to the persisted Chroma directory")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model identifier for responses")
    parser.add_argument("--provider", default="openai", help="LLM provider identifier")
    parser.add_argument("--api-key", dest="api_key", help="Explicit API key override for the LLM provider")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the chat model")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens per LLM response")
    parser.add_argument("--base-url", dest="base_url", help="Optional base URL for OpenAI-compatible endpoints")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt that governs assistant behaviour")
    parser.add_argument("--show-context", action="store_true", help="Display retrieved snippets for each answer")
    parser.add_argument("--save-transcript", dest="transcript_path", help="Optional file path to write the chat transcript on exit")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to stderr")
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


def save_transcript(
    memory: ConversationSummaryBufferMemory,
    transcript_log: TranscriptLog,
    path: Path,
    console: Console,
) -> None:
    try:
        lines = []
        summary_text = memory.moving_summary_buffer.strip()
        if summary_text:
            lines.append("# Conversation summary\n" + summary_text + "\n")

        user_contexts = list(transcript_log.iter_user_contexts())
        user_index = 0
        history_messages = memory.load_memory_variables({}).get(memory.memory_key, [])

        for message in history_messages:
            if (
                isinstance(message, memory.summary_message_cls)
                and message.content == memory.moving_summary_buffer
            ):
                # The summary is already recorded above; skip the placeholder message.
                continue

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

    api_key = resolve_api_key(args.api_key)
    provider = args.provider

    if not api_key:
        console.print(
            "[red]No API key available. Set --api-key or the OPENAI_API_KEY/RAG_LLM_API_KEY environment variable.[/red]"
        )
        sys.exit(1)

    logger.debug("Using provider '%s' with model '%s'.", provider, args.llm_model)

    try:
        llm = load_chat_model(
            provider=provider,
            model_name=args.llm_model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    decider_llm = build_decider(llm)
    rewriter_llm = build_rewriter(llm)
    topic_guard_llm = build_topic_guard(llm)
    rejection_llm = build_rejection_writer(llm)

    logger.debug(
        "LLM loaded; structured helpers ready (decider=%s, rewriter=%s, topic_guard=%s, rejection_writer=%s).",
        type(decider_llm).__name__,
        type(rewriter_llm).__name__,
        type(topic_guard_llm).__name__,
        type(rejection_llm).__name__,
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=max(args.max_tokens * 2, 1200),
    )
    history_adapter = SummaryHistoryAdapter(memory)

    logger.debug("Conversation memory initialised with max_token_limit=%s.", max(args.max_tokens * 2, 1200))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_message}"),
        ]
    )
    response_chain = prompt | llm | StrOutputParser()
    chat_with_history = RunnableWithMessageHistory(
        response_chain,
        lambda _session_id: history_adapter,
        input_messages_key="user_message",
        history_messages_key="chat_history",
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
    session_id = "cli-session"

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Session ended.[/red]")
            break

        if not user_message.strip():
            continue

        logger.debug("Received user message: %s", user_message)

        normalized = user_message.strip().lower()
        if any(phrase in normalized for phrase in ("who are you", "what are you", "what do you do")):
            answer = f"I’m an expert research assistant specializing in: {SPECIALIZATION_LIST}."
            history_adapter.add_messages(
                [HumanMessage(content=user_message), AIMessage(content=answer)]
            )
            transcript_log.add_user_context([])
            console.print(Panel(answer, title="Assistant", style="green"))
            continue

        if user_message.startswith("/"):
            command = user_message.lstrip("/").lower()
            if command in {"exit", "quit"}:
                break
            if command == "help":
                console.print("Commands: /help, /exit, /reset, /showctx (toggle context display)")
                continue
            if command == "reset":
                history_adapter.clear()
                transcript_log.reset()
                console.print("[green]Cleared conversation history.[/green]")
                continue
            if command == "showctx":
                show_context = not show_context
                console.print(f"[green]Context display {'enabled' if show_context else 'disabled'}.[/green]")
                continue
            console.print(f"[yellow]Unknown command '{command}'. Try /help.[/yellow]")
            continue

        results: Sequence[Tuple[Any, float]] = []
        context_blocks: Sequence[str] = []
        answer = ""
        use_rag = False
        decision = None
        manual_history_update = False

        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            gate_decision = topic_gate(
                guard_llm=topic_guard_llm,
                user_message=user_message,
                specialization_list=SPECIALIZATION_LIST,
            )
            gate_payload = gate_decision.model_dump() if hasattr(gate_decision, "model_dump") else gate_decision.dict()
            logger.debug("Topic gate verdict: %s", gate_payload)

            if not gate_decision.on_topic:
                answer = generate_rejection(
                    llm=rejection_llm,
                    specialization_list=SPECIALIZATION_LIST,
                    user_message=user_message,
                )
                logger.debug("Off-topic gate triggered (reason: %s).", gate_decision.reason)
                manual_history_update = True
            else:
                decision = decide_use_rag(
                    decider_llm=decider_llm,
                    history_adapter=history_adapter,
                    user_message=user_message,
                    rag_topics=RAG_TOPIC_INVENTORY,
                    memory=history_adapter.memory,
                    min_conf=0.70,
                )
                use_rag = decision.use_rag

                decision_payload = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()
                logger.debug("Decider verdict: %s", decision_payload)

                if use_rag:
                    rewrite, results = apply_rewrite_and_retrieve(
                        rewriter_llm=rewriter_llm,
                        retrieve_contexts_fn=retrieve_contexts,
                        store=store,
                        user_message=user_message,
                        history_adapter=history_adapter,
                        rag_topics=RAG_TOPIC_INVENTORY,
                        k=args.retrieval_k,
                        use_namespace_filter=False,
                    )
                    rewrite_payload = rewrite.model_dump() if hasattr(rewrite, "model_dump") else rewrite.dict()
                    logger.debug("Query rewrite produced: %s", rewrite_payload)
                    logger.debug(
                        "Retrieved %d context chunks via rewritten query '%s'.",
                        len(results),
                        rewrite.query or user_message,
                    )
                    context_blocks = format_contexts(results)
                else:
                    results = []
                    context_blocks = []
                    logger.debug("RAG skipped for this turn (reason: %s).", decision.reason)

                try:
                    user_payload = build_user_payload(
                        user_message=user_message,
                        results=results,
                        compose_user_prompt_fn=compose_user_prompt,
                        use_rag=use_rag,
                        decision=decision,
                    )
                    logger.debug("User payload forwarded to LLM (truncated): %s", user_payload[:500])
                    answer = chat_with_history.invoke(
                        {
                            "system_prompt": args.system_prompt,
                            "user_message": user_payload,
                        },
                        config={"configurable": {"session_id": session_id}},
                    )
                    logger.debug("LLM output (truncated): %s", answer[:500])
                except Exception as exc:
                    logger.exception("LLM call failed.")
                    console.print(f"[red]LLM call failed: {exc}[/red]")
                    sys.exit(1)

        if manual_history_update:
            history_adapter.add_messages(
                [HumanMessage(content=user_message), AIMessage(content=answer)]
            )

        transcript_log.add_user_context(context_blocks)
        if use_rag and show_context:
            render_context_table(results, console)

        console.print(Panel(answer, title="Assistant", style="green"))

    if args.transcript_path:
        save_transcript(memory, transcript_log, Path(args.transcript_path), console)


if __name__ == "__main__":
    main()
