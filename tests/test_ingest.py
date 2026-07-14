import importlib
from pathlib import Path

import pytest

ingest_module = importlib.import_module("scripts.00_ingest")


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    hidden = tmp_path / ".hidden.md"
    hidden.write_text("# Hidden\n\nIgnore me", encoding="utf-8")
    visible = tmp_path / "lesson.md"
    visible.write_text("# Title\n\nWelcome\n\n## Part\n\nDetails", encoding="utf-8")
    return tmp_path


def test_list_corpus_files_skips_hidden(monkeypatch: pytest.MonkeyPatch, corpus_dir: Path) -> None:
    monkeypatch.setattr(ingest_module, "CORPUS_DIR", str(corpus_dir))
    files = ingest_module.list_corpus_files(str(corpus_dir))
    assert all(Path(f).name != ".hidden.md" for f in files)
    assert {Path(f).name for f in files} == {"lesson.md"}


def test_ingest_returns_documents(monkeypatch: pytest.MonkeyPatch, corpus_dir: Path) -> None:
    monkeypatch.setattr(ingest_module, "CORPUS_DIR", str(corpus_dir))
    docs = ingest_module.ingest()
    assert len(docs) >= 2
    headers = [doc.metadata.get("#") for doc in docs if doc.metadata.get("#")]
    assert "Title" in headers
    contents = {doc.page_content.strip() for doc in docs}
    assert "Welcome" in contents


def test_list_corpus_files_skips_non_markdown(monkeypatch: pytest.MonkeyPatch, corpus_dir: Path) -> None:
    (corpus_dir / "notes.txt").write_text("not markdown", encoding="utf-8")
    (corpus_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(ingest_module, "CORPUS_DIR", str(corpus_dir))
    files = ingest_module.list_corpus_files(str(corpus_dir))
    assert {Path(f).name for f in files} == {"lesson.md"}


def test_ingest_skips_undecodable_file_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, corpus_dir: Path
) -> None:
    # A .md file with bytes that aren't valid UTF-8 shouldn't take down the whole run.
    (corpus_dir / "broken.md").write_bytes(b"\xff\xfe# Not UTF-8\n")
    monkeypatch.setattr(ingest_module, "CORPUS_DIR", str(corpus_dir))
    docs = ingest_module.ingest()
    sources = {doc.metadata.get("source") for doc in docs}
    assert "broken.md" not in sources
    assert "lesson.md" in sources
