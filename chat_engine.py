"""chat_engine.py — UI-agnostic core of the RAG chat pipeline.

Extracted from ``scripts/05_chat_cli.py`` so the plain Rich CLI and the
Textual TUI (``scripts/06_chat_tui.py``) share one implementation of the
turn-processing pipeline (topic gate -> RAG decider -> optional rewrite ->
retrieve -> rerank -> compose -> LLM -> history/transcript commit) instead of
each maintaining its own copy. Nothing here imports ``rich`` or ``textual``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:
    from langchain.memory import ConversationSummaryBufferMemory
except ImportError:
    try:
        from langchain_classic.memory import ConversationSummaryBufferMemory
    except ImportError:
        from langchain_community.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent_orchestration_helper import (
    RAG_TOPIC_INVENTORY,
    SPECIALIZATION_LIST,
    apply_rewrite_and_retrieve,
    build_decider,
    build_rewriter,
    build_user_payload,
    decide_use_rag,
)
from utils import (
    build_rejection_writer,
    build_topic_guard,
    generate_rejection,
    topic_gate,
)
from utils.llm_provider import resolve_provider_and_key
from utils.semantic_cache import SemanticCache, cached_retrieve_and_rerank

query_module = import_module("scripts.02_query")

create_retrieval_store = query_module.create_retrieval_store  # type: ignore[attr-defined]
load_chat_model = query_module.load_chat_model  # type: ignore[attr-defined]
compose_user_prompt = query_module.compose_user_prompt  # type: ignore[attr-defined]
retrieve_contexts = query_module.retrieve_contexts  # type: ignore[attr-defined]
rerank_results = query_module.rerank_results  # type: ignore[attr-defined]
clean_snippet = query_module.clean_snippet  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Stay in expert research assistant mode. Follow the persona and context provided in the user message."
)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-5-mini"
CHROMA_DIR = Path("data") / "chroma"

# Exact-match (not substring) so genuine questions like "what do you do when a
# Pikachu faints?" aren't hijacked by this canned identity reply.
IDENTITY_PHRASES = {"who are you", "what are you", "what do you do"}


@dataclass
class ChatEngineConfig:
    """Everything needed to stand up a chat session, independent of any UI."""

    persist_dir: Path = CHROMA_DIR
    embedding_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    provider: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2000
    base_url: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    retrieval_k: int = 3
    enable_semantic_cache: bool = True
    semantic_cache_size: int = 20
    semantic_cache_similarity_threshold: float = 0.93


@dataclass
class TranscriptTurn:
    """A single user/assistant exchange, with the contexts retrieved for it."""

    user_message: str
    answer: str
    contexts: Sequence[str] = field(default_factory=list)


@dataclass
class TranscriptLog:
    """Tracks each user/assistant turn, independent of chat-memory pruning."""

    turns: List[TranscriptTurn] = field(default_factory=list)

    def add_turn(self, user_message: str, answer: str, contexts: Sequence[str]) -> None:
        self.turns.append(TranscriptTurn(user_message=user_message, answer=answer, contexts=contexts))

    def reset(self) -> None:
        self.turns.clear()


@dataclass
class TurnResult:
    """The outcome of one ``ChatEngine.process_turn`` call."""

    answer: str
    use_rag: bool = False
    results: Sequence[Tuple[Any, float]] = field(default_factory=list)
    context_blocks: Sequence[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


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


def format_contexts(results: Sequence[Tuple[Any, float]]) -> List[str]:
    """Render retrieved (doc, score) pairs as plain-text blocks for prompts/transcripts."""

    formatted: List[str] = []
    for idx, (doc, score) in enumerate(results):
        snippet = clean_snippet(doc.page_content)
        combined = None
        try:
            combined = getattr(doc, "metadata", {}).get("combined_score")
        except Exception:
            combined = None
        score_display = f"{score:.3f}" if score is not None else "n/a"
        source_label = (
            f"[source {idx}] score={score_display} rerank={combined:.3f}" if combined is not None else f"[source {idx}] score={score_display}"
        )
        if snippet:
            formatted.append(f"{source_label}\n{snippet}")
        else:
            formatted.append(f"{source_label}\n(empty snippet)")
    return formatted


class ChatEngine:
    """Owns the retrieval store, LLM, memory, and per-turn RAG pipeline.

    UI layers (the Rich CLI, the Textual TUI) construct one of these, call
    ``process_turn`` for each user message, and render the returned
    ``TurnResult`` however suits their surface.
    """

    def __init__(self, config: ChatEngineConfig):
        self.config = config

        self.embedder, self.store = create_retrieval_store(
            model_name=config.embedding_model, persist_dir=Path(config.persist_dir)
        )

        # Semantic result cache: skips retrieval+rerank for near-duplicate
        # queries. Only built when there's an embedder to drive the
        # similarity lookup -- e.g. tests stub `create_retrieval_store` to
        # return `(None, ...)`, and the pipeline degrades to uncached
        # retrieval in that case rather than failing. See
        # utils/semantic_cache.py for how/why this cache is designed the
        # way it is.
        self.semantic_cache: Optional[SemanticCache] = None
        self._cached_retrieve = None
        if config.enable_semantic_cache and self.embedder is not None:
            self.semantic_cache = SemanticCache(
                max_size=config.semantic_cache_size,
                similarity_threshold=config.semantic_cache_similarity_threshold,
            )
            self._cached_retrieve = cached_retrieve_and_rerank(
                cache=self.semantic_cache,
                embed_fn=self.embedder.embed_query,
                retrieve_fn=retrieve_contexts,
                rerank_fn=rerank_results,
            )

        provider, api_key = resolve_provider_and_key(config.api_key, config.provider)
        if not api_key:
            raise RuntimeError(
                "No API key available. Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY environment variable."
            )
        self.provider = provider

        self.llm = load_chat_model(
            provider=provider,
            model_name=config.llm_model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=config.base_url,
        )

        self.decider_llm = build_decider(self.llm)
        self.rewriter_llm = build_rewriter(self.llm)
        self.topic_guard_llm = build_topic_guard(self.llm)
        self.rejection_llm = build_rejection_writer(self.llm)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max(config.max_tokens * 2, 1200),
        )
        self.history_adapter = SummaryHistoryAdapter(self.memory)
        self.transcript_log = TranscriptLog()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_message}"),
            ]
        )
        self.response_chain = prompt | self.llm | StrOutputParser()

    def reset(self) -> None:
        self.history_adapter.clear()
        self.transcript_log.reset()

    def process_turn(self, user_message: str) -> TurnResult:
        """Run one user message through the full RAG pipeline and commit it to history."""

        normalized = user_message.strip().lower().rstrip("?!.")
        if normalized in IDENTITY_PHRASES:
            answer = f"I’m an expert research assistant specializing in: {SPECIALIZATION_LIST}."
            self.history_adapter.add_messages(
                [HumanMessage(content=user_message), AIMessage(content=answer)]
            )
            self.transcript_log.add_turn(user_message, answer, [])
            return TurnResult(answer=answer)

        gate_decision = topic_gate(
            guard_llm=self.topic_guard_llm,
            user_message=user_message,
            specialization_list=SPECIALIZATION_LIST,
        )
        gate_payload = gate_decision.model_dump() if hasattr(gate_decision, "model_dump") else gate_decision.dict()
        logger.debug("Topic gate verdict: %s", gate_payload)

        results: Sequence[Tuple[Any, float]] = []
        context_blocks: Sequence[str] = []
        use_rag = False
        decision = None
        answer = ""

        if not gate_decision.on_topic:
            answer = generate_rejection(
                llm=self.rejection_llm,
                specialization_list=SPECIALIZATION_LIST,
                user_message=user_message,
            )
            logger.debug("Off-topic gate triggered (reason: %s).", gate_decision.reason)
        else:
            decision = decide_use_rag(
                decider_llm=self.decider_llm,
                history_adapter=self.history_adapter,
                user_message=user_message,
                rag_topics=RAG_TOPIC_INVENTORY,
                memory=self.history_adapter.memory,
                min_conf=0.70,
            )
            use_rag = decision.use_rag
            decision_payload = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()
            logger.debug("Decider verdict: %s", decision_payload)

            if use_rag:
                # When the semantic cache is active it already reranks
                # internally on a miss (see cached_retrieve_and_rerank), so
                # rerank_fn is only passed here for the uncached fallback.
                retrieve_fn = self._cached_retrieve or retrieve_contexts
                rerank_fn = None if self._cached_retrieve else rerank_results
                rewrite, results = apply_rewrite_and_retrieve(
                    rewriter_llm=self.rewriter_llm,
                    retrieve_contexts_fn=retrieve_fn,
                    store=self.store,
                    user_message=user_message,
                    history_adapter=self.history_adapter,
                    rag_topics=RAG_TOPIC_INVENTORY,
                    k=self.config.retrieval_k,
                    use_namespace_filter=False,
                    rerank_fn=rerank_fn,
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
                # Invoked directly (not via RunnableWithMessageHistory) so the
                # persona/context-laden payload sent to the LLM never gets
                # written into chat_history as if it were what the user typed.
                answer = self.response_chain.invoke(
                    {
                        "system_prompt": self.config.system_prompt,
                        "chat_history": self.history_adapter.messages,
                        "user_message": user_payload,
                    }
                )
                logger.debug("LLM output (truncated): %s", answer[:500])
            except Exception as exc:
                logger.exception("LLM call failed.")
                return TurnResult(answer="", use_rag=use_rag, results=results, context_blocks=context_blocks, error=str(exc))

        self.history_adapter.add_messages(
            [HumanMessage(content=user_message), AIMessage(content=answer)]
        )
        self.transcript_log.add_turn(user_message, answer, context_blocks)
        return TurnResult(answer=answer, use_rag=use_rag, results=results, context_blocks=context_blocks)

    def save_transcript(self, path: Path) -> None:
        """Write the conversation transcript to ``path``. Raises on failure."""

        lines: List[str] = []
        summary_text = self.memory.moving_summary_buffer.strip()
        if summary_text:
            lines.append("# Conversation summary (older turns folded in by memory)\n" + summary_text + "\n")

        # Read straight from transcript_log rather than reconstructing from
        # ``memory``: ConversationSummaryBufferMemory prunes older messages into
        # the summary above, so its message list is a moving suffix of the full
        # conversation and can't be zipped positionally against a same-length
        # per-turn context list.
        for turn in self.transcript_log.turns:
            lines.append(f"## User\n{turn.user_message}\n")
            if turn.contexts:
                lines.append("Retrieved contexts:\n")
                for ctx in turn.contexts:
                    lines.append(f"- {ctx}")
                lines.append("")
            lines.append(f"## Assistant\n{turn.answer}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
