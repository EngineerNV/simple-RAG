from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

eval_module = importlib.import_module("scripts.03_eval")


def test_compute_overlap_ratio_detects_shared_tokens() -> None:
    context = ["RAG pipelines combine retrieval and generation."]
    ratio = eval_module.compute_overlap_ratio("Retrieval augmented generation", context)
    assert ratio > 0


def test_is_faithful_uses_threshold() -> None:
    context = ["RAG uses retrieval."]
    assert eval_module.is_faithful("RAG uses retrieval.", context, threshold=0.5)
    assert not eval_module.is_faithful("Unrelated answer", context, threshold=0.5)


def test_should_abstain_on_short_context() -> None:
    assert eval_module.should_abstain(["tiny"])


def test_evaluate_qa_pair_parses_json_context() -> None:
    qa = {"question": "Q", "answer": "RAG uses retrieval", "context": json.dumps(["RAG uses retrieval context"])}
    result = eval_module.evaluate_qa_pair(qa)
    assert result["faithful"] in {True, False}
    assert isinstance(result["context_list"], list)
    assert result["context_length"] > 0


def test_load_eval_data_supports_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "eval.json"
    payload = [{"question": "Q", "answer": "A", "context": "C"}]
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    data_json = eval_module.load_eval_data(json_path)
    assert data_json == payload

    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("question,answer,context\nq,a,c\n", encoding="utf-8")
    data_csv = eval_module.load_eval_data(csv_path)
    assert data_csv[0]["question"] == "q"


def test_load_question_file_handles_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text("question\nWhat is RAG?\n\n", encoding="utf-8")
    questions = eval_module.load_question_file(csv_path)
    assert questions == ["What is RAG?"]
