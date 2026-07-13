"""Offline tests for the debug chat TUI panels and turn timings — no API keys."""

from __future__ import annotations

from importlib import import_module

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

import agent_orchestration_helper as aoh
from utils.chat_history import SummaryBufferHistory
from tests.fakes import FakeAnswerChain, FakeStructuredLLM, StubStore, stub_retrieve_contexts

dbg = import_module("scripts.07_debug_chat")
chat_cli = import_module("scripts.05_chat_cli")


def _render(renderable) -> str:
    console = Console(record=True, width=100)
    console.print(renderable)
    return console.export_text()


def _route(**overrides) -> aoh.RouteDecision:
    payload = dict(
        on_topic=True,
        use_rag=True,
        search_query="pikachu thunderbolt",
        keywords=["pikachu"],
        reason="test route",
        confidence=0.92,
    )
    payload.update(overrides)
    return aoh.RouteDecision(**payload)


def _results():
    return [
        (
            Document(
                page_content="Try using Thunderbolt against water types.",
                metadata={"source": "pikachu.md", "combined_score": 0.81, "lexical_overlap": 0.4},
            ),
            0.83,
        ),
    ]


def _history(turns: int = 1, summary: str = "") -> SummaryBufferHistory:
    history = SummaryBufferHistory(max_token_limit=100_000)
    for i in range(turns):
        history.add_messages([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])
    if summary:
        history.moving_summary_buffer = summary
    return history


def _session(router_responses, answers=("grounded answer",), store=None):
    history = SummaryBufferHistory(max_token_limit=1200)
    chain = FakeAnswerChain(answers, history=history)
    store = store if store is not None else StubStore(_results())
    session = aoh.ChatSession(
        router_llm=FakeStructuredLLM(router_responses),
        rejection_llm=FakeStructuredLLM([RuntimeError("no rejection llm in tests")]),
        chat_with_history=chain,
        history_adapter=history,
        store=store,
        retrieve_contexts_fn=stub_retrieve_contexts,
        system_prompt=aoh.build_system_prompt(),
        retrieval_k=2,
    )
    return session, history


# ---------------------------- metrics panel ----------------------------


def test_metrics_panel_waits_before_first_turn() -> None:
    text = _render(dbg.build_metrics_panel(None, _history(0)))
    assert "waiting for first turn" in text


def test_metrics_panel_shows_router_retrieval_timing_memory() -> None:
    result = aoh.TurnResult(
        answer="ok",
        results=_results(),
        used_rag=True,
        route=_route(),
        timings={"router_ms": 210.4, "retrieval_ms": 45.2, "llm_ms": 1234.0, "total_ms": 1489.6},
    )
    text = _render(dbg.build_metrics_panel(result, _history(2, summary="short recap")))

    assert "0.92" in text                       # router confidence
    assert "pikachu thunderbolt" in text        # rewritten query
    assert "pikachu.md" in text                 # chunk source
    assert "0.830" in text and "0.810" in text  # score and rerank
    assert "210 ms" in text and "1,234 ms" in text and "1,490 ms" in text
    assert "2" in text                          # turn count
    assert "11 chars" in text                   # summary length


def test_metrics_panel_distinguishes_empty_retrieval_from_no_rag() -> None:
    empty = aoh.TurnResult(answer="ok", results=(), used_rag=False, route=_route(use_rag=True))
    assert "no chunks returned" in _render(dbg.build_metrics_panel(empty, _history(0)))

    skipped = aoh.TurnResult(answer="ok", results=(), used_rag=False, route=_route(use_rag=False))
    assert "RAG not used" in _render(dbg.build_metrics_panel(skipped, _history(0)))


def test_metrics_panel_survives_non_numeric_scores() -> None:
    doc = Document(page_content="x", metadata={"source": "a.md", "combined_score": "0.85"})
    result = aoh.TurnResult(answer="ok", results=[(doc, None)], used_rag=True, route=_route())
    text = _render(dbg.build_metrics_panel(result, _history(0)))
    assert "n/a" in text


def test_metrics_panel_shows_error() -> None:
    result = aoh.TurnResult(answer="Sorry", route=_route(), error="model exploded")
    assert "model exploded" in _render(dbg.build_metrics_panel(result, _history(0)))


# ---------------------------- chat panel ----------------------------


def test_chat_panel_shows_error_turns() -> None:
    state = dbg.DebugState()
    ok = aoh.TurnResult(answer="fine", route=_route())
    failed = aoh.TurnResult(answer="Sorry — I hit an error talking to the model. Please try again.",
                            route=_route(), error="boom")
    state.add_turn("q1", ok.answer, ok)
    state.add_turn("q2", failed.answer, failed)

    text = _render(dbg.build_chat_panel(state.turns))
    assert "q1" in text and "fine" in text
    assert "q2" in text and "Sorry" in text     # error answer is visible
    assert state.last_result is failed


# ---------------------------- context table guard ----------------------------


def test_render_context_table_handles_none_score() -> None:
    doc = Document(page_content="snippet", metadata={"source": "a.md", "combined_score": "not-a-float"})
    console = Console(record=True, width=100)
    chat_cli.render_context_table([(doc, None)], console)
    text = console.export_text()
    assert "n/a" in text


# ---------------------------- turn timings ----------------------------


def test_handle_turn_populates_timings() -> None:
    session, _ = _session([_route()])
    result = session.handle_turn("What move should Pikachu use?")
    assert set(result.timings) == {"router_ms", "retrieval_ms", "llm_ms", "total_ms"}
    assert result.timings["total_ms"] >= result.timings["router_ms"]


def test_error_turn_still_reports_timings() -> None:
    session, _ = _session([_route()], answers=[RuntimeError("model exploded")])
    result = session.handle_turn("What move should Pikachu use?")
    assert result.error == "model exploded"
    assert "total_ms" in result.timings and "router_ms" in result.timings


def test_rejection_turn_reports_router_and_total() -> None:
    session, _ = _session([_route(on_topic=False, use_rag=False)])
    result = session.handle_turn("Best pizza in town?")
    assert set(result.timings) == {"router_ms", "total_ms"}
