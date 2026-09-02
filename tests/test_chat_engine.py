from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from chat_engine import (
    ChatEngine,
    ChatEngineConfig,
    SummaryHistoryAdapter,
    TranscriptLog,
    TranscriptTurn,
    TurnResult,
    format_contexts,
)


class MockMemory:
    def __init__(self) -> None:
        self.memory_key = "chat_history"
        self.messages: list[Any] = []
        self.moving_summary_buffer = "Previous conversation summary."
        self.chat_memory = self

    def load_memory_variables(self, _inputs: dict) -> dict:
        return {self.memory_key: self.messages}

    def add_message(self, msg: Any) -> None:
        self.messages.append(msg)

    def prune(self) -> None:
        pass

    def clear(self) -> None:
        self.messages.clear()
        self.moving_summary_buffer = ""


def test_transcript_log() -> None:
    log = TranscriptLog()
    assert len(log.turns) == 0

    log.add_turn("User Q", "Assistant A", ["ctx 1"])
    assert len(log.turns) == 1
    assert log.turns[0].user_message == "User Q"
    assert log.turns[0].answer == "Assistant A"
    assert log.turns[0].contexts == ["ctx 1"]

    log.reset()
    assert len(log.turns) == 0


def test_summary_history_adapter() -> None:
    mem = MockMemory()
    adapter = SummaryHistoryAdapter(mem)

    assert adapter.memory is mem
    assert adapter.messages == []

    msg1 = HumanMessage(content="Hello")
    msg2 = AIMessage(content="Hi there")
    adapter.add_message(msg1)
    adapter.add_messages([msg2])

    assert len(adapter.messages) == 2
    assert adapter.messages[0].content == "Hello"
    assert adapter.messages[1].content == "Hi there"

    adapter.clear()
    assert len(adapter.messages) == 0


def test_format_contexts() -> None:
    doc1 = Document(page_content=" First context snippet ", metadata={"combined_score": 0.85})
    doc2 = Document(page_content="", metadata={})

    results = [(doc1, 0.9), (doc2, 0.4)]
    formatted = format_contexts(results)

    assert len(formatted) == 2
    assert "[source 0] score=0.900 rerank=0.850" in formatted[0]
    assert "First context snippet" in formatted[0]
    assert "[source 1] score=0.400" in formatted[1]
    assert "(empty snippet)" in formatted[1]


def test_turn_result_ok() -> None:
    tr_ok = TurnResult(answer="Good response")
    assert tr_ok.ok is True

    tr_err = TurnResult(answer="", error="LLM API error")
    assert tr_err.ok is False


class MockGateDecision:
    def __init__(self, on_topic: bool, reason: str = "ok"):
        self.on_topic = on_topic
        self.reason = reason

    def model_dump(self):
        return {"on_topic": self.on_topic, "reason": self.reason}


class MockRAGDecision:
    def __init__(self, use_rag: bool, reason: str = "ok"):
        self.use_rag = use_rag
        self.reason = reason

    def model_dump(self):
        return {"use_rag": self.use_rag, "reason": self.reason}


class MockRewrite:
    def __init__(self, query: str = "rewritten query"):
        self.query = query

    def model_dump(self):
        return {"query": self.query}


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult


class MockLLM(BaseChatModel):
    answer: str = "Mocked LLM answer"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.answer)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self

    @property
    def _llm_type(self) -> str:
        return "mock"


def test_chat_engine_identity_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chat_engine.create_retrieval_store", lambda **kw: (None, object()))
    monkeypatch.setattr("chat_engine.resolve_provider_and_key", lambda k, p: ("openai", "sk-mock"))
    monkeypatch.setattr("chat_engine.load_chat_model", lambda **kw: MockLLM())

    config = ChatEngineConfig(persist_dir=Path("data/chroma"))
    engine = ChatEngine(config)

    result = engine.process_turn("who are you")
    assert result.ok is True
    assert "expert research assistant" in result.answer
    assert len(engine.transcript_log.turns) == 1
    assert engine.transcript_log.turns[0].user_message == "who are you"


def test_chat_engine_off_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chat_engine.create_retrieval_store", lambda **kw: (None, object()))
    monkeypatch.setattr("chat_engine.resolve_provider_and_key", lambda k, p: ("openai", "sk-mock"))
    monkeypatch.setattr("chat_engine.load_chat_model", lambda **kw: MockLLM())
    monkeypatch.setattr("chat_engine.topic_gate", lambda **kw: MockGateDecision(on_topic=False, reason="Irrelevant"))
    monkeypatch.setattr("chat_engine.generate_rejection", lambda **kw: "Sorry, I can only answer questions about supported topics.")

    config = ChatEngineConfig(persist_dir=Path("data/chroma"))
    engine = ChatEngine(config)

    result = engine.process_turn("What is the recipe for chocolate cake?")
    assert result.ok is True
    assert "only answer questions about supported topics" in result.answer
    assert result.use_rag is False


def test_chat_engine_rag_turn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("chat_engine.create_retrieval_store", lambda **kw: (None, object()))
    monkeypatch.setattr("chat_engine.resolve_provider_and_key", lambda k, p: ("openai", "sk-mock"))
    monkeypatch.setattr("chat_engine.load_chat_model", lambda **kw: MockLLM(answer="Generated RAG response."))
    monkeypatch.setattr("chat_engine.topic_gate", lambda **kw: MockGateDecision(on_topic=True))
    monkeypatch.setattr("chat_engine.decide_use_rag", lambda **kw: MockRAGDecision(use_rag=True))

    doc = Document(page_content="Relevant info snippet", metadata={"combined_score": 0.9})
    monkeypatch.setattr("chat_engine.apply_rewrite_and_retrieve", lambda **kw: (MockRewrite("query"), [(doc, 0.9)]))

    config = ChatEngineConfig(persist_dir=Path("data/chroma"))
    engine = ChatEngine(config)

    result = engine.process_turn("Tell me about topic")
    assert result.ok is True
    assert result.answer == "Generated RAG response."
    assert result.use_rag is True
    assert len(result.context_blocks) == 1

    # Test saving transcript
    transcript_file = tmp_path / "transcript.md"
    engine.save_transcript(transcript_file)
    content = transcript_file.read_text(encoding="utf-8")
    assert "## User\nTell me about topic" in content
    assert "## Assistant\nGenerated RAG response." in content


def test_chat_engine_no_rag_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chat_engine.create_retrieval_store", lambda **kw: (None, object()))
    monkeypatch.setattr("chat_engine.resolve_provider_and_key", lambda k, p: ("openai", "sk-mock"))
    monkeypatch.setattr("chat_engine.load_chat_model", lambda **kw: MockLLM(answer="Chitchat response."))
    monkeypatch.setattr("chat_engine.topic_gate", lambda **kw: MockGateDecision(on_topic=True))
    monkeypatch.setattr("chat_engine.decide_use_rag", lambda **kw: MockRAGDecision(use_rag=False))

    config = ChatEngineConfig(persist_dir=Path("data/chroma"))
    engine = ChatEngine(config)

    result = engine.process_turn("Hello!")
    assert result.ok is True
    assert result.answer == "Chitchat response."
    assert result.use_rag is False
    assert len(result.context_blocks) == 0

    engine.reset()
    assert len(engine.transcript_log.turns) == 0


def test_chat_engine_reset_preserves_semantic_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbedder:
        def embed_query(self, text: str):
            return [1.0, 0.0]

    monkeypatch.setattr("chat_engine.create_retrieval_store", lambda **kw: (DummyEmbedder(), object()))
    monkeypatch.setattr("chat_engine.resolve_provider_and_key", lambda k, p: ("openai", "sk-mock"))
    monkeypatch.setattr("chat_engine.load_chat_model", lambda **kw: MockLLM())

    config = ChatEngineConfig(persist_dir=Path("data/chroma"), enable_semantic_cache=True)
    engine = ChatEngine(config)

    assert engine.semantic_cache is not None
    engine.semantic_cache.put("cached query", [1.0, 0.0], k=3, value="cached answer")
    assert len(engine.semantic_cache) == 1

    engine.reset()
    assert len(engine.semantic_cache) == 1  # Cache preserved across user reset for economy
