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


def test_retrieve_contexts_handles_vector_query() -> None:
    class StubVectorStore:
        def __init__(self) -> None:
            self.vectors = []

        def similarity_search_by_vector_with_relevance_scores(self, embedding, k: int):
            self.vectors.append(embedding)
            return [(Document(page_content="vector doc"), 1.6)]

        def _select_relevance_score_fn(self):
            return lambda d: 1.0 - d / 2.0

    store = StubVectorStore()
    results = query_module.retrieve_contexts(store, [0.5, 0.5], 1)
    assert store.vectors == [[0.5, 0.5]]
    assert results[0][0].page_content == "vector doc"
    assert results[0][1] == pytest.approx(0.2)  # 1.0 - 1.6 / 2.0 = 0.2


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


def test_rerank_promotes_lexical_match() -> None:
    # Build results where doc[0] has higher retriever score but no lexical
    # overlap, and doc[1] contains the query token but has lower retriever score.
    doc_a = Document(page_content="completely unrelated content", metadata={})
    doc_b = Document(page_content="This chunk mentions RAG and retrieval.", metadata={})
    results = [(doc_a, 0.9), (doc_b, 0.2)]

    # Force the lexical blend (alpha=0 prioritizes lexical overlap) so this
    # test doesn't depend on downloading the cross-encoder model.
    reranked = query_module.rerank_results(results, "What is RAG?", alpha=0.0, use_cross_encoder=False)

    # Expect the lexical match (doc_b) to be promoted above doc_a when lexical
    # overlap is combined with retriever score.
    assert reranked[0][0].page_content.startswith("This chunk mentions RAG")


def test_rerank_handles_empty_question() -> None:
    doc = Document(page_content="content", metadata={})
    results = [(doc, 0.7)]
    # Empty question should not raise and should return the same single result
    out = query_module.rerank_results(results, "", alpha=0.5, use_cross_encoder=False)
    assert len(out) == 1
    assert out[0][0].page_content == "content"


def test_rerank_cross_encoder_used_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cross-encoder path is tried first; a stub avoids the real model download."""

    class StubCrossEncoder:
        def predict(self, pairs):
            # Score the second pair (the lexically relevant one) higher.
            return [-5.0, 5.0]

    monkeypatch.setattr(query_module, "_load_cross_encoder", lambda: StubCrossEncoder())
    doc_a = Document(page_content="completely unrelated content", metadata={})
    doc_b = Document(page_content="This chunk mentions RAG and retrieval.", metadata={})
    results = [(doc_a, 0.9), (doc_b, 0.2)]

    reranked = query_module.rerank_results(results, "What is RAG?")

    assert reranked[0][0].page_content.startswith("This chunk mentions RAG")
    assert reranked[0][0].metadata["combined_score"] > reranked[1][0].metadata["combined_score"]


def test_rerank_falls_back_to_lexical_when_cross_encoder_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_module, "_load_cross_encoder", lambda: None)
    doc_a = Document(page_content="completely unrelated content", metadata={})
    doc_b = Document(page_content="This chunk mentions RAG and retrieval.", metadata={})
    results = [(doc_a, 0.9), (doc_b, 0.2)]

    # use_cross_encoder defaults to True, but loading returns None, so this
    # must silently fall back to the lexical blend rather than erroring.
    reranked = query_module.rerank_results(results, "What is RAG?", alpha=0.0)

    assert reranked[0][0].page_content.startswith("This chunk mentions RAG")
