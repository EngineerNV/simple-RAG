"""Shared fakes for exercising the chat/orchestration layer offline."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from langchain_core.documents import Document


class FakeStructuredLLM:
    """Stands in for a structured-output LLM: pops queued responses or raises.

    Records every prompt it receives so tests can assert on prompt contents.
    """

    def __init__(self, responses: Sequence[Any] = ()) -> None:
        self.responses: List[Any] = list(responses)
        self.prompts: List[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.prompts.append(messages)
        if not self.responses:
            raise AssertionError("FakeStructuredLLM has no queued responses left.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAnswerChain:
    """Stands in for the RunnableWithMessageHistory answer chain."""

    def __init__(self, answers: Sequence[Any] = ("stub answer",), history: Any = None) -> None:
        self.answers: List[Any] = list(answers)
        self.calls: List[dict] = []
        self.history = history

    def invoke(self, inputs: dict, config: dict | None = None) -> str:
        self.calls.append(inputs)
        if not self.answers:
            raise AssertionError("FakeAnswerChain has no queued answers left.")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        # Mirror RunnableWithMessageHistory: persist the exchange on success.
        if self.history is not None:
            from langchain_core.messages import AIMessage, HumanMessage

            self.history.add_messages(
                [HumanMessage(content=inputs["user_message"]), AIMessage(content=answer)]
            )
        return answer


class StubStore:
    """Vector store double returning canned (Document, score) results."""

    def __init__(self, results: Sequence[Tuple[Document, float]] | None = None) -> None:
        self.results = list(results or [])
        self.queries: List[str] = []

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.queries.append(question)
        return self.results[:k]


def stub_retrieve_contexts(store: StubStore, question: str, k: int):
    return store.similarity_search_with_relevance_scores(question, k=k)
