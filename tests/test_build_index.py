from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, List

import pytest
from langchain_core.documents import Document

build_index = importlib.import_module("scripts.01_build_index")


class DummyEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface placeholder
        return [[0.1] * 3 for _ in texts]

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - interface placeholder
        return [0.1] * 3


class FakeCollection:
    def __init__(self, metadatas: List[dict]):
        self._metadatas = metadatas
        self.name = "fake"

    def count(self) -> int:
        return len(self._metadatas)


class FakeChroma:
    def __init__(self, texts: List[str], metadatas: List[dict], persist_directory: str):
        self.texts = texts
        self.metadatas = metadatas
        self.persist_directory = persist_directory
        self.persist_called = False
        self._collection = FakeCollection(metadatas)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding,
        metadatas: List[dict],
        persist_directory: str,
    ) -> "FakeChroma":
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        instance = cls(texts, metadatas, persist_directory)
        instance.embedding = embedding
        return instance

    def persist(self) -> None:
        self.persist_called = True

    def get(self, include: Iterable[str] | None = None, limit: int = 1) -> dict:
        return {"metadatas": self.metadatas[:limit]}


@pytest.fixture(autouse=True)
def reset_run_metadata() -> None:
    build_index._RUN_METADATA.clear()


def _make_docs() -> List[Document]:
    return [
        Document(page_content="chunk 1", metadata={"#": "Title", "tags": {"python", "rag"}}),
        Document(page_content="chunk 2", metadata={"#": "Title", "##": "Section", "notes": [1, 2, 3]}),
    ]


def test_persist_chroma_sanitizes_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(build_index, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(build_index, "Chroma", FakeChroma)
    docs = _make_docs()
    store = build_index.persist_chroma(docs, DummyEmbeddings())

    assert isinstance(store, FakeChroma)
    assert store.persist_called
    assert all(isinstance(md["tags"], str) for md in store.metadatas if "tags" in md)
    assert chroma_dir.exists()
    assert build_index._RUN_METADATA["doc_count"] == len(docs)


def test_persist_chroma_recreates_existing_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "old.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(build_index, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(build_index, "Chroma", FakeChroma)

    store = build_index.persist_chroma(_make_docs(), DummyEmbeddings())
    assert store.persist_directory == str(chroma_dir)
    assert build_index._RUN_METADATA["rebuilt"] is True


def test_summarize_run_outputs_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(build_index, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(build_index, "Chroma", FakeChroma)

    docs = _make_docs()
    store = build_index.persist_chroma(docs, DummyEmbeddings())
    build_index.summarize_run(store)

    captured = capsys.readouterr().out
    assert "Chroma index build complete" in captured
    assert "Documents embedded" in captured
