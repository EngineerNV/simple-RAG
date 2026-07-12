"""Offline smoke tests for the chat turn logic — no API keys required."""

from __future__ import annotations

from langchain_core.documents import Document

import agent_orchestration_helper as aoh
from utils.chat_history import SummaryBufferHistory
from utils.rejections import RejectionOut
from tests.fakes import FakeAnswerChain, FakeStructuredLLM, StubStore, stub_retrieve_contexts


def _route(**overrides) -> aoh.RouteDecision:
    payload = dict(
        on_topic=True,
        use_rag=True,
        search_query="pikachu thunderbolt",
        keywords=[],
        reason="test",
        confidence=0.95,
    )
    payload.update(overrides)
    return aoh.RouteDecision(**payload)


def _results():
    return [
        (Document(page_content="Try using Thunderbolt against water types.", metadata={"source": "pikachu.md"}), 0.83),
        (Document(page_content="Pikachu evolves into Raichu.", metadata={"source": "pikachu.md"}), 0.61),
    ]


def _session(router_responses, answers=("grounded answer",), rejection_responses=(), store=None):
    history = SummaryBufferHistory(max_token_limit=1200)
    chain = FakeAnswerChain(answers, history=history)
    store = store if store is not None else StubStore(_results())
    session = aoh.ChatSession(
        router_llm=FakeStructuredLLM(router_responses),
        rejection_llm=FakeStructuredLLM(rejection_responses),
        chat_with_history=chain,
        history_adapter=history,
        store=store,
        retrieve_contexts_fn=stub_retrieve_contexts,
        system_prompt=aoh.build_system_prompt(),
        retrieval_k=2,
    )
    return session, history, chain, store


def test_full_rag_turn() -> None:
    session, history, chain, store = _session([_route()])
    result = session.handle_turn("What move should Pikachu use?")

    assert result.answer == "grounded answer"
    assert result.used_rag is True
    assert result.error is None
    # Retrieval used the router's rewritten query, not the raw message.
    assert store.queries == ["pikachu thunderbolt"]

    [inputs] = chain.calls
    context_block = inputs["context_block"]
    assert '<document index="0" source="pikachu.md" score="0.830">' in context_block
    # Most-relevant chunk comes first and evidence is NOT sanitized:
    # lines starting with "Try ..." must survive intact.
    assert context_block.index("Thunderbolt") < context_block.index("Raichu")
    assert "Try using Thunderbolt" in context_block


def test_history_stores_clean_user_message() -> None:
    session, history, chain, _ = _session([_route()])
    session.handle_turn("What move should Pikachu use?")

    human_messages = [m for m in history.raw_messages if m.type == "human"]
    assert len(human_messages) == 1
    assert human_messages[0].content == "What move should Pikachu use?"
    assert "<documents>" not in human_messages[0].content


def test_off_topic_turn_uses_rejection_and_appends_history() -> None:
    session, history, chain, store = _session(
        [_route(on_topic=False, use_rag=False, confidence=0.95)],
        rejection_responses=[RejectionOut(text="That's outside my focus.")],
    )
    result = session.handle_turn("What's the best pizza in town?")

    assert result.answer == "That's outside my focus."
    assert result.used_rag is False
    assert store.queries == []
    assert chain.calls == []
    contents = [m.content for m in history.raw_messages]
    assert contents == ["What's the best pizza in town?", "That's outside my focus."]


def test_rejection_writer_failure_uses_static_fallback() -> None:
    session, _, _, _ = _session(
        [_route(on_topic=False, use_rag=False, confidence=0.95)],
        rejection_responses=[RuntimeError("rejection writer down")],
    )
    result = session.handle_turn("What's the best pizza in town?")
    assert "outside my focus" in result.answer
    assert aoh.SPECIALIZATION_LIST in result.answer


def test_llm_failure_keeps_session_alive() -> None:
    session, history, chain, _ = _session(
        [_route(), _route()],
        answers=[RuntimeError("model exploded"), "recovered answer"],
    )

    first = session.handle_turn("What move should Pikachu use?")
    assert first.error == "model exploded"
    assert "error" in first.answer.lower() or "Sorry" in first.answer
    # A failed turn leaves no residue in history.
    assert history.raw_messages == []

    second = session.handle_turn("What move should Pikachu use?")
    assert second.answer == "recovered answer"
    assert second.error is None
    assert len(history.raw_messages) == 2


def test_no_rag_turn_has_empty_context_block() -> None:
    session, _, chain, store = _session([_route(use_rag=False, search_query="")])
    result = session.handle_turn("Say hi!")

    assert result.used_rag is False
    assert store.queries == []
    [inputs] = chain.calls
    assert inputs["context_block"] == ""


def test_rag_turn_with_no_hits_notes_missing_evidence() -> None:
    session, _, chain, _ = _session([_route()], store=StubStore([]))
    result = session.handle_turn("What move should Pikachu use?")

    assert result.used_rag is False  # nothing actually retrieved
    [inputs] = chain.calls
    assert "No matching notes" in inputs["context_block"]


def test_retrieval_failure_answers_without_contexts() -> None:
    class ExplodingStore:
        def similarity_search_with_relevance_scores(self, question: str, k: int):
            raise RuntimeError("chroma offline")

    session, _, chain, _ = _session([_route()], store=ExplodingStore())
    result = session.handle_turn("What move should Pikachu use?")

    assert result.answer == "grounded answer"
    assert result.error is None
    [inputs] = chain.calls
    assert "No matching notes" in inputs["context_block"]
