from __future__ import annotations

import importlib

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

query_module = importlib.import_module("scripts.02_query")


class StubStore:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.questions.append(question)
        docs = [
            (Document(page_content=" first  snippet ", metadata={"#": "Title"}), 0.9),
            (Document(page_content="second snippet", metadata={"section": "details"}), "0.4"),
        ]
        return docs[:k]


def test_clean_snippet_trims_whitespace() -> None:
    assert query_module.clean_snippet("  spaced   text\n") == "spaced text"


def test_format_metadata_handles_missing() -> None:
    assert query_module.format_metadata(None) == "metadata: none"
    assert "section=details" in query_module.format_metadata({"section": "details"})


def test_retrieve_contexts_coerces_scores() -> None:
    store = StubStore()
    results = query_module.retrieve_contexts(store, "what is rag?", 2)
    assert store.questions == ["what is rag?"]
    assert results[0][1] == pytest.approx(0.9)
    assert isinstance(results[1][1], float)


def test_compose_messages_returns_chat_objects() -> None:
    doc = Document(page_content="content", metadata={})
    messages = query_module.compose_messages("Question?", [(doc, 0.5)])
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_run_llm_mode_falls_back_without_key(capsys: pytest.CaptureFixture[str]) -> None:
    doc = Document(page_content="answer chunk", metadata={"id": 1})
    results = [(doc, 0.8)]
    query_module.run_llm_mode(
        question="Explain the pipeline",
        results=results,
        provider="openai",
        model_name="gpt-test",
        api_key=None,
        temperature=0.1,
        max_tokens=20,
        base_url=None,
        show_usage=False,
    )
    captured = capsys.readouterr()
    assert "OpenAI API key missing" in captured.err
    assert "(mock) Answer" in captured.out
