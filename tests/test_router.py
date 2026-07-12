from __future__ import annotations

import agent_orchestration_helper as aoh
from tests.fakes import FakeStructuredLLM


def _decision(**overrides) -> aoh.RouteDecision:
    payload = dict(
        on_topic=True,
        use_rag=True,
        search_query="pikachu evolution levels",
        keywords=["pikachu"],
        reason="asks for indexed facts",
        confidence=0.9,
    )
    payload.update(overrides)
    return aoh.RouteDecision(**payload)


def test_confident_decision_passes_through() -> None:
    router = FakeStructuredLLM([_decision()])
    decision = aoh.route_turn(router, "How does Pikachu evolve?")
    assert decision.use_rag is True
    assert decision.search_query == "pikachu evolution levels"
    assert decision.on_topic is True


def test_low_confidence_falls_back_to_retrieve_and_abstain() -> None:
    router = FakeStructuredLLM([_decision(on_topic=False, use_rag=False, search_query="", confidence=0.3)])
    decision = aoh.route_turn(router, "Tell me about that thing", min_conf=0.6)
    assert decision.on_topic is True
    assert decision.use_rag is True
    assert decision.search_query == "Tell me about that thing"
    assert "low confidence" in decision.reason


def test_router_exception_falls_back() -> None:
    router = FakeStructuredLLM([RuntimeError("boom")])
    decision = aoh.route_turn(router, "How does Pikachu evolve?")
    assert decision.on_topic is True
    assert decision.use_rag is True
    assert decision.search_query == "How does Pikachu evolve?"
    assert "router error" in decision.reason


def test_use_rag_without_query_defaults_to_user_message() -> None:
    router = FakeStructuredLLM([_decision(search_query="")])
    decision = aoh.route_turn(router, "How does Pikachu evolve?")
    assert decision.search_query == "How does Pikachu evolve?"


def test_router_prompt_contains_context_sections() -> None:
    router = FakeStructuredLLM([_decision()])
    aoh.route_turn(
        router,
        "What about its speed stat?",
        history_excerpt="ai: Pikachu evolves with a Thunder Stone.",
        chat_summary="User is researching Pikachu.",
    )
    [messages] = router.prompts
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "routing classifier" in system
    assert aoh.RAG_TOPIC_INVENTORY.strip() in user
    assert aoh.SPECIALIZATION_LIST in user
    assert "Thunder Stone" in user
    assert "researching Pikachu" in user
    assert "What about its speed stat?" in user
