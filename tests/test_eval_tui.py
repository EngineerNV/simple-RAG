from __future__ import annotations

import importlib
import pytest

pytest.importorskip("textual")

eval_tui_module = importlib.import_module("scripts.08_eval_tui")
build_ragas_command = eval_tui_module.build_ragas_command
build_lexical_command = eval_tui_module.build_lexical_command
build_quiz_command = eval_tui_module.build_quiz_command


def test_build_ragas_command() -> None:
    cmd = build_ragas_command(
        golden_set="data/eval/golden_qa.json",
        out="data/eval/ragas_report.json",
        k="3",
        provider="openai",
        llm_model="gpt-5-mini",
        api_key="sk-test",
        judge_provider="claude",
        judge_model="claude-haiku-4-5",
        judge_api_key="sk-judge",
        temperature="0.2",
        max_tokens="2000",
    )
    assert sys.executable in cmd[0]
    assert "07_ragas_eval.py" in cmd[1]
    assert "--golden-set" in cmd
    assert "data/eval/golden_qa.json" in cmd
    assert "--provider" in cmd
    assert "openai" in cmd
    assert "--judge-provider" in cmd
    assert "claude" in cmd
    assert "--judge-model" in cmd
    assert "claude-haiku-4-5" in cmd


def test_build_lexical_command() -> None:
    cmd = build_lexical_command(
        dataset_path="data/eval/golden_qa.json",
        out="data/eval_report.json",
        k="5",
        agent_mode="pretend",
        provider="gemini",
        llm_model="gemini-1.5-flash",
        api_key="",
        rebuild_index=True,
    )
    assert "03_eval.py" in cmd[1]
    assert "--in" in cmd
    assert "data/eval/golden_qa.json" in cmd
    assert "--k" in cmd
    assert "5" in cmd
    assert "--provider" in cmd
    assert "gemini" in cmd
    assert "--rebuild-index" in cmd


def test_build_quiz_command() -> None:
    cmd = build_quiz_command(
        questions_path="data/eval/golden_qa.json",
        out="data/human_review.jsonl",
        k="3",
        agent_mode="pretend",
        provider="",
        llm_model="",
        api_key="",
        resume=True,
        shuffle=True,
    )
    assert "03_quiz.py" in cmd[1]
    assert "--questions" in cmd
    assert "data/eval/golden_qa.json" in cmd
    assert "--resume" in cmd
    assert "--shuffle" in cmd
