from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

quiz_module = importlib.import_module("scripts.06_quiz")


def test_main_refuses_to_overwrite_existing_output_without_resume(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([{"id": "q1", "question": "What is RAG?"}]), encoding="utf-8")

    out_path = tmp_path / "reviews.jsonl"
    out_path.write_text(json.dumps({"id": "q1", "question": "What is RAG?"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already has"):
        quiz_module.main(["--questions", str(questions_path), "--out", str(out_path)])


def test_main_allows_empty_existing_output_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([{"id": "q1", "question": "What is RAG?"}]), encoding="utf-8")

    out_path = tmp_path / "reviews.jsonl"
    out_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(quiz_module, "HuggingFaceEmbeddings", lambda model_name: object())

    class FakeQueryModule:
        CHROMA_DIR = tmp_path / "chroma"

        @staticmethod
        def load_vector_store(persist_dir, embed):
            raise FileNotFoundError("no index at that path")

    monkeypatch.setattr(quiz_module, "load_query_module", lambda: FakeQueryModule())

    # An empty output file has no saved reviews, so it shouldn't trip the
    # overwrite guard; main() should reach (and gracefully exit at) the
    # missing-vector-store check instead of raising the RuntimeError above.
    quiz_module.main(["--questions", str(questions_path), "--out", str(out_path)])
    assert "no index at that path" in capsys.readouterr().err
