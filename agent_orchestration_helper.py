# agent_orchestration_helper.py — RAG-aware orchestration helpers (no score gate).

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Any, List
import warnings
import json
from pathlib import Path
from pydantic import BaseModel, Field

try:
    from langchain_core.messages import BaseMessage
except Exception:
    BaseMessage = Any  # type: ignore

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

from utils.inventory_view import build_specialization_list
from utils.persona import build_persona_preamble
from utils.text_sanitize import validate_output

# To customize the content inventory without editing code, edit the JSON file
# placed next to this module: rag_content.json
# Schema:
# {
#   "rag_topic_inventory": "...multi-line text...",
#   "specialization_topics": ["topic 1", "topic 2", ...]
# }

DEFAULT_RAG_TOPIC_INVENTORY: str = (
    """
RAG covers:
- Kanto region field notes for Generation I Pokémon (focus on Pikachu)
- Species bios, habitats, abilities, typings, base stats, movesets
- Trainer tips, battle tactics, evolutionary paths, item interactions
- No real-time events; canonical up to the Indigo League era (circa 1998)
""".strip()
)

DEFAULT_SPECIALIZATION_TOPICS = [
    "Generation I Pokémon field research across the Kanto region",
    "Species bios, habitats, abilities, typings, base stats, and movesets",
    "Trainer strategies, battle tactics, evolutionary paths, and item interactions",
    "Lore through the Indigo League era (circa 1998)",
]


def _load_agent_content_from_json() -> Tuple[str, List[str]]:
    """Load inventory/topics from rag_content.json with safe fallbacks.

    Returns:
        (rag_topic_inventory, specialization_topics)
    """
    cfg_path = Path(__file__).with_name("rag_content.json")
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
        rag_text = str(data.get("rag_topic_inventory") or "").strip()
        topics = data.get("specialization_topics")
        if not rag_text:
            rag_text = DEFAULT_RAG_TOPIC_INVENTORY
        if not isinstance(topics, list) or not topics:
            topics = DEFAULT_SPECIALIZATION_TOPICS
        # Coerce items to str
        topics = [str(t) for t in topics]
        return rag_text, topics  # type: ignore[return-value]
    except FileNotFoundError:
        return DEFAULT_RAG_TOPIC_INVENTORY, DEFAULT_SPECIALIZATION_TOPICS
    except Exception:
        # On any parse/IO error, fall back to baked-in defaults
        return DEFAULT_RAG_TOPIC_INVENTORY, DEFAULT_SPECIALIZATION_TOPICS


RAG_TOPIC_INVENTORY, SPECIALIZATION_TOPICS = _load_agent_content_from_json()

SPECIALIZATION_LIST = build_specialization_list(SPECIALIZATION_TOPICS)

class RetrievalDecision(BaseModel):
    use_rag: bool = Field(..., description="True only if RAG likely improves factual accuracy.")
    reason: str = Field(..., description="One-sentence rationale for audit/logging.")
    overlaps_rag_topics: bool = Field(False, description="LLM believes user intent overlaps RAG inventory.")
    need_fresh_facts: bool = Field(False, description="User asks for specific facts/steps/names likely in RAG.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Subjective confidence 0..1")

DECIDER_SYSTEM: str = (
    "You are a classifier that decides whether to consult a retrieval database (RAG). "
    "Say True ONLY if the answer depends on specific facts/steps/names/APIs contained in RAG, "
    "or the user is following up on something the AI previously offered that is in RAG. "
    "If the question is general knowledge, subjective, or creative, say False. "
    "If the topic is outside RAG inventory or requires real-time web, say False."
)

DECIDER_TEMPLATE: str = """\
RAG topic inventory:
{rag_topics}

Previous AI message (if any):
{prev_ai}

Latest user message:
{user_msg}

Chat summary (if available):
{chat_summary}

Rules:
- Prefer False unless you're confident RAG adds factual grounding.
- If the user refers to "that link/that code/that API you mentioned" and it's in RAG, prefer True.
- Do NOT use RAG for real-time info or anything outside the inventory.

Return a compact decision.
"""

def get_prev_ai_message_text(history_adapter: Any, max_lookback: int = 1) -> str:
    try:
        msgs: List[Any] = list(getattr(history_adapter, "messages", []) or [])
    except Exception:
        return ""
    if not msgs:
        return ""
    count = 0
    for m in reversed(msgs):
        mtype = getattr(m, "type", None) or m.__class__.__name__.lower()
        if "ai" in mtype:
            count += 1
            if count == max_lookback:
                return getattr(m, "content", "") or ""
    return ""

def get_summary_text(memory: Any) -> str:
    if memory is None:
        return ""
    return getattr(memory, "moving_summary_buffer", "") or ""

def build_decider(llm: Any):
    try:
        constrained = llm.bind(max_tokens=2000, max_completion_tokens=2000)
        return constrained.with_structured_output(RetrievalDecision)
    except AttributeError as exc:
        raise RuntimeError(
            "The provided llm does not support with_structured_output. "
            "Use a LangChain ChatModel (e.g., ChatOpenAI/ChatAnthropic)."
        ) from exc

def decide_use_rag(
    decider_llm: Any,
    history_adapter: Any,
    user_message: str,
    rag_topics: Optional[str] = None,
    memory: Optional[Any] = None,
    min_conf: float = 0.55,
) -> RetrievalDecision:
    prev_ai = get_prev_ai_message_text(history_adapter) or "(none)"
    chat_summary = get_summary_text(getattr(history_adapter, "memory", memory)) or "(none)"
    rag_topics = (rag_topics or RAG_TOPIC_INVENTORY).strip()

    prompt = DECIDER_TEMPLATE.format(
        rag_topics=rag_topics,
        prev_ai=prev_ai,
        user_msg=user_message,
        chat_summary=(chat_summary[:400] + " …") if chat_summary and len(chat_summary) > 400 else chat_summary,
    )

    decision: RetrievalDecision = decider_llm.invoke([
        {"role": "system", "content": DECIDER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    if decision.confidence < min_conf:
        return RetrievalDecision(
            use_rag=False,
            reason=f"below min_conf={min_conf:.2f}: {decision.reason}",
            overlaps_rag_topics=decision.overlaps_rag_topics,
            need_fresh_facts=decision.need_fresh_facts,
            confidence=decision.confidence,
        )
    return decision

class QueryRewrite(BaseModel):
    query: str = Field(..., description="Single best query string to embed for cosine similarity.")
    keywords: List[str] = Field(default_factory=list, description="Key terms, entities, API names, file paths.")
    entities: List[str] = Field(default_factory=list, description="Named entities (teams, services, components).")
    suspected_topic: Optional[str] = Field(None, description="Optional topic/namespace to filter the retriever.")
    rationale: str = Field("", description="Short note for debugging/tuning.")

REWRITER_SYSTEM: str = (
    "Rewrite for cosine-similarity retrieval. Return ONLY the JSON fields per schema; keep them concise. "
    "Preserve concrete nouns, API/class names, file paths, error strings, IDs, and dates. Resolve references using history when possible. "
    "Limit the 'query' field to 30 words or fewer. Do not output anything outside the JSON schema."
)

REWRITER_TEMPLATE: str = """\
If the user message is a follow-up to the previous AI message, resolve anaphora like
"that code", "step 2", "that API", etc., using the history excerpt.

History excerpt (optional):
{history_excerpt}

RAG topic inventory (to help choose terms):
{rag_topics}

User message:
{user_msg}

Return a compact rewrite suited for embedding, plus any keywords/entities that aid retrieval.
"""

def build_rewriter(llm: Any):
    try:
        constrained = llm.bind(max_tokens=2000, max_completion_tokens=2000)
        return constrained.with_structured_output(QueryRewrite)
    except AttributeError:
        class _PlainRewriter:
            def invoke(self, msgs):
                content = ""
                for m in reversed(msgs):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = str(m.get("content", ""))
                        break
                q = content.split("User message:")[-1].strip() or content.strip()
                return QueryRewrite(query=q, rationale="plain fallback", keywords=[], entities=[], suspected_topic=None)
        return _PlainRewriter()

def rewrite_for_retrieval(
    rewriter_llm: Any,
    user_message: str,
    history_adapter: Optional[Any] = None,
    rag_topics: Optional[str] = None,
    max_hist_chars: int = 400,
) -> QueryRewrite:
    history_excerpt = ""
    if history_adapter:
        try:
            lines = [getattr(m, "content", "") for m in getattr(history_adapter, "messages", [])[-8:]]
            joined = "\n".join(x for x in lines if x)
            clamp = min(max_hist_chars, 400)
            history_excerpt = (joined[:clamp] + " …") if len(joined) > clamp else joined
        except Exception:
            history_excerpt = ""
    rag_topics = (rag_topics or RAG_TOPIC_INVENTORY).strip()
    prompt = REWRITER_TEMPLATE.format(
        history_excerpt=history_excerpt or "(none)",
        rag_topics=rag_topics or "(none)",
        user_msg=user_message,
    )
    try:
        result: QueryRewrite = rewriter_llm.invoke([
            {"role": "system", "content": REWRITER_SYSTEM},
            {"role": "user", "content": prompt},
        ])
    except Exception as exc:
        return QueryRewrite(
            query=user_message,
            keywords=[],
            entities=[],
            suspected_topic=None,
            rationale=f"rewriter fallback due to error: {exc}",
        )
    return result

def apply_rewrite_and_retrieve(
    rewriter_llm: Any,
    retrieve_contexts_fn,
    store: Any,
    user_message: str,
    history_adapter: Optional[Any] = None,
    rag_topics: Optional[str] = None,
    k: int = 3,
    use_namespace_filter: bool = False,
) -> Tuple[QueryRewrite, Sequence[Tuple[Any, float]]]:
    rewrite = rewrite_for_retrieval(
        rewriter_llm=rewriter_llm,
        user_message=user_message,
        history_adapter=history_adapter,
        rag_topics=rag_topics,
    )
    query = rewrite.query or user_message
    results = retrieve_contexts_fn(store, query, k)
    return rewrite, results

def build_user_payload(
    user_message: str,
    results: Sequence[Tuple[Any, float]],
    compose_user_prompt_fn,
    use_rag: bool,
    decision: Optional[RetrievalDecision] = None,
    specialization_list: str = SPECIALIZATION_LIST,
    allow_suggestions: bool = False,
) -> str:
    persona = build_persona_preamble(specialization_list)

    if use_rag:
        prompt = compose_user_prompt_fn(user_message, results)
        return validate_output(persona + "\n" + prompt, allow_suggestions)

    overlaps_topics = bool(decision and getattr(decision, "overlaps_rag_topics", False))
    if overlaps_topics:
        prompt = (
            persona
            + "\nNo directly matching notes found. Answer concisely from your own knowledge; do not cite.\n"
            f"User question: {user_message}"
        )
        return validate_output(prompt, allow_suggestions)

    prompt = (
        persona
        + "\nThis appears outside my specialization. Provide a brief refusal—no suggestions—and state the specialization list.\n"
        f"User question: {user_message}"
    )
    return validate_output(prompt, allow_suggestions)
