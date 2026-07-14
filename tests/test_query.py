from __future__ import annotations

import importlib
import json
import warnings

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

query_module = importlib.import_module("scripts.02_query")


class FakeChroma:
    def __init__(self, persist_directory: str, embedding_function) -> None:
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function


class NamedEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def test_load_vector_store_warns_on_embedding_model_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    (persist_dir / "_index_meta.json").write_text(
        json.dumps({"embedding_model": "model-a"}), encoding="utf-8"
    )
    monkeypatch.setattr(query_module, "Chroma", FakeChroma)

    with pytest.warns(RuntimeWarning, match="model-a"):
        query_module.load_vector_store(persist_dir, NamedEmbeddings("model-b"))


def test_load_vector_store_silent_when_embedding_model_matches(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    (persist_dir / "_index_meta.json").write_text(
        json.dumps({"embedding_model": "model-a"}), encoding="utf-8"
    )
    monkeypatch.setattr(query_module, "Chroma", FakeChroma)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        query_module.load_vector_store(persist_dir, NamedEmbeddings("model-a"))


def test_load_vector_store_silent_without_sidecar_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setattr(query_module, "Chroma", FakeChroma)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        query_module.load_vector_store(persist_dir, NamedEmbeddings("model-a"))


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
    assert "API key missing for provider 'openai'" in captured.err
    assert "(mock) Answer" in captured.out


def test_rerank_promotes_lexical_match() -> None:
    # Build results where doc[0] has higher retriever score but no lexical
    # overlap, and doc[1] contains the query token but has lower retriever score.
    doc_a = Document(page_content="completely unrelated content", metadata={})
    doc_b = Document(page_content="This chunk mentions RAG and retrieval.", metadata={})
    results = [(doc_a, 0.9), (doc_b, 0.2)]

    # Use alpha=0 to prioritize lexical overlap for this test case.
    reranked = query_module.rerank_results(results, "What is RAG?", alpha=0.0)

    # Expect the lexical match (doc_b) to be promoted above doc_a when lexical
    # overlap is combined with retriever score.
    assert reranked[0][0].page_content.startswith("This chunk mentions RAG")


def test_rerank_handles_empty_question() -> None:
    doc = Document(page_content="content", metadata={})
    results = [(doc, 0.7)]
    # Empty question should not raise and should return the same single result
    out = query_module.rerank_results(results, "", alpha=0.5)
    assert len(out) == 1
    assert out[0][0].page_content == "content"
