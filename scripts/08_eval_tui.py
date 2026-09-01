"""08_eval_tui.py — Textual TUI for configuring and running simple-RAG evaluations.

Provides a full-screen dashboard to configure and trigger:
1. RAGAS LLM-Judge Evaluation (scripts/07_ragas_eval.py)
2. Lexical Heuristic Evaluation (scripts/03_eval.py)
3. Human Quiz / Review Setup (scripts/03_quiz.py)

Outputs real-time logs and metrics into an interactive terminal interface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

try:  # Optional dependency for convenient local development.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    TabbedContent,
    TabPane,
)


def build_ragas_command(
    golden_set: str,
    out: str,
    k: str,
    provider: str,
    llm_model: str,
    api_key: str,
    judge_provider: str,
    judge_model: str,
    judge_api_key: str,
    temperature: str,
    max_tokens: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "07_ragas_eval.py"),
        "--golden-set",
        golden_set or "data/eval/golden_qa.json",
        "--out",
        out or "data/eval/ragas_report.json",
        "--retrieval-k",
        k or "3",
        "--temperature",
        temperature or "0.2",
        "--max-tokens",
        max_tokens or "2000",
    ]
    if provider:
        cmd.extend(["--provider", provider])
    if llm_model:
        cmd.extend(["--llm-model", llm_model])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if judge_provider:
        cmd.extend(["--judge-provider", judge_provider])
    if judge_model:
        cmd.extend(["--judge-model", judge_model])
    if judge_api_key:
        cmd.extend(["--judge-api-key", judge_api_key])
    return cmd


def build_lexical_command(
    dataset_path: str,
    out: str,
    k: str,
    agent_mode: str,
    provider: str,
    llm_model: str,
    api_key: str,
    rebuild_index: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "03_eval.py"),
        "--out",
        out or "data/eval_report.json",
        "--k",
        k or "3",
        "--agent-mode",
        agent_mode or "pretend",
    ]
    if dataset_path:
        if dataset_path.endswith(".json") or dataset_path.endswith(".csv"):
            cmd.extend(["--in", dataset_path])
        else:
            cmd.extend(["--questions", dataset_path])
    else:
        cmd.extend(["--in", "data/eval/golden_qa.json"])

    if provider:
        cmd.extend(["--provider", provider])
    if llm_model:
        cmd.extend(["--llm-model", llm_model])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if rebuild_index:
        cmd.append("--rebuild-index")
    return cmd


def build_quiz_command(
    questions_path: str,
    out: str,
    k: str,
    agent_mode: str,
    provider: str,
    llm_model: str,
    api_key: str,
    resume: bool,
    shuffle: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "03_quiz.py"),
        "--questions",
        questions_path or "data/eval/golden_qa.json",
        "--out",
        out or "data/human_review.jsonl",
        "--k",
        k or "3",
        "--agent-mode",
        agent_mode or "pretend",
    ]
    if provider:
        cmd.extend(["--provider", provider])
    if llm_model:
        cmd.extend(["--llm-model", llm_model])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if resume:
        cmd.append("--resume")
    if shuffle:
        cmd.append("--shuffle")
    return cmd


class EvalTUI(App):
    """Textual app to configure and execute simple-RAG evaluations."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-container {
        height: 1fr;
    }
    #left-pane {
        width: 1fr;
        border: round $accent;
        padding: 1;
    }
    #right-pane {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    .form-group {
        margin: 0 0 1 0;
    }
    .form-label {
        text-style: bold;
        color: $accent;
    }
    Button {
        margin-top: 1;
        width: 100%;
    }
    #status-label {
        padding: 1;
        text-style: bold;
        color: $success;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.is_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with VerticalScroll(id="left-pane"):
                with TabbedContent(initial="tab-ragas"):
                    with TabPane("RAGAS LLM-Judge Eval", id="tab-ragas"):
                        yield Label("Golden Set QA Path", classes="form-label")
                        yield Input(value="data/eval/golden_qa.json", id="ragas-golden-set")

                        yield Label("Output Report Path", classes="form-label")
                        yield Input(value="data/eval/ragas_report.json", id="ragas-out")

                        yield Label("Retrieval k", classes="form-label")
                        yield Input(value="3", id="ragas-k")

                        yield Label("Provider", classes="form-label")
                        yield Select(
                            [("Auto-detect", ""), ("OpenAI", "openai"), ("Gemini", "gemini"), ("Claude", "claude")],
                            value="",
                            id="ragas-provider",
                        )

                        yield Label("Pipeline LLM Model", classes="form-label")
                        yield Input(value="gpt-5-mini", id="ragas-model")

                        yield Label("API Key Override (Optional)", classes="form-label")
                        yield Input(placeholder="Auto-detected if blank", password=True, id="ragas-api-key")

                        yield Label("Judge Provider (Optional)", classes="form-label")
                        yield Select(
                            [("Default (Same as Pipeline)", ""), ("OpenAI", "openai"), ("Gemini", "gemini"), ("Claude", "claude")],
                            value="",
                            id="ragas-judge-provider",
                        )

                        yield Label("Judge Model (Optional)", classes="form-label")
                        yield Input(placeholder="Defaults to pipeline model", id="ragas-judge-model")

                        yield Button("Run RAGAS Evaluation", variant="primary", id="btn-run-ragas")

                    with TabPane("Lexical Heuristic Eval", id="tab-lexical"):
                        yield Label("Input Dataset / Questions Path", classes="form-label")
                        yield Input(value="data/eval/golden_qa.json", id="lexical-dataset")

                        yield Label("Output Report Path", classes="form-label")
                        yield Input(value="data/eval_report.json", id="lexical-out")

                        yield Label("Retrieval k", classes="form-label")
                        yield Input(value="3", id="lexical-k")

                        yield Label("Agent Mode", classes="form-label")
                        yield Select(
                            [("Pretend (Mock Answer)", "pretend"), ("None (Concatenate)", "none"), ("LLM (Live Model)", "llm")],
                            value="pretend",
                            id="lexical-agent-mode",
                        )

                        yield Label("Provider", classes="form-label")
                        yield Select(
                            [("Auto-detect", ""), ("OpenAI", "openai"), ("Gemini", "gemini"), ("Claude", "claude")],
                            value="",
                            id="lexical-provider",
                        )

                        yield Label("LLM Model", classes="form-label")
                        yield Input(value="gpt-5-mini", id="lexical-model")

                        yield Checkbox("Rebuild Index Before Eval", value=False, id="lexical-rebuild-index")

                        yield Button("Run Lexical Evaluation", variant="primary", id="btn-run-lexical")

                    with TabPane("Human Quiz Review", id="tab-quiz"):
                        yield Label("Questions File Path", classes="form-label")
                        yield Input(value="data/eval/golden_qa.json", id="quiz-questions")

                        yield Label("Output Review JSONL Path", classes="form-label")
                        yield Input(value="data/human_review.jsonl", id="quiz-out")

                        yield Label("Retrieval k", classes="form-label")
                        yield Input(value="3", id="quiz-k")

                        yield Label("Agent Mode", classes="form-label")
                        yield Select(
                            [("Pretend", "pretend"), ("None", "none"), ("LLM", "llm")],
                            value="pretend",
                            id="quiz-agent-mode",
                        )

                        yield Checkbox("Resume Previous Progress", value=True, id="quiz-resume")
                        yield Checkbox("Shuffle Questions", value=False, id="quiz-shuffle")

                        yield Button("Setup / Generate Quiz Output", variant="primary", id="btn-run-quiz")

            with Vertical(id="right-pane"):
                yield Label("Evaluation Status & Output Log", classes="form-label")
                yield Label("Status: Ready", id="status-label")
                yield RichLog(id="log-view", highlight=True, markup=True)

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.is_running:
            self.query_one("#log-view", RichLog).write("[yellow]An evaluation is already running. Please wait...[/yellow]")
            return

        button_id = event.button.id
        if button_id == "btn-run-ragas":
            cmd = build_ragas_command(
                golden_set=self.query_one("#ragas-golden-set", Input).value,
                out=self.query_one("#ragas-out", Input).value,
                k=self.query_one("#ragas-k", Input).value,
                provider=self.query_one("#ragas-provider", Select).value,
                llm_model=self.query_one("#ragas-model", Input).value,
                api_key=self.query_one("#ragas-api-key", Input).value,
                judge_provider=self.query_one("#ragas-judge-provider", Select).value,
                judge_model=self.query_one("#ragas-judge-model", Input).value,
                judge_api_key="",
                temperature="0.2",
                max_tokens="2000",
            )
            self.run_eval_job("RAGAS LLM-Judge Evaluation", cmd)

        elif button_id == "btn-run-lexical":
            cmd = build_lexical_command(
                dataset_path=self.query_one("#lexical-dataset", Input).value,
                out=self.query_one("#lexical-out", Input).value,
                k=self.query_one("#lexical-k", Input).value,
                agent_mode=self.query_one("#lexical-agent-mode", Select).value,
                provider=self.query_one("#lexical-provider", Select).value,
                llm_model=self.query_one("#lexical-model", Input).value,
                api_key="",
                rebuild_index=self.query_one("#lexical-rebuild-index", Checkbox).value,
            )
            self.run_eval_job("Lexical Heuristic Evaluation", cmd)

        elif button_id == "btn-run-quiz":
            cmd = build_quiz_command(
                questions_path=self.query_one("#quiz-questions", Input).value,
                out=self.query_one("#quiz-out", Input).value,
                k=self.query_one("#quiz-k", Input).value,
                agent_mode=self.query_one("#quiz-agent-mode", Select).value,
                provider="",
                llm_model="",
                api_key="",
                resume=self.query_one("#quiz-resume", Checkbox).value,
                shuffle=self.query_one("#quiz-shuffle", Checkbox).value,
            )
            self.run_eval_job("Human Quiz Setup", cmd)

    @work(exclusive=True, thread=True)
    def run_eval_job(self, title: str, cmd: List[str]) -> None:
        self.is_running = True
        status = self.query_one("#status-label", Label)
        status.update(f"[cyan]Status: Running {title}...[/cyan]")
        log = self.query_one("#log-view", RichLog)
        log.write(f"[bold green]=== Starting {title} ===[/bold green]")
        log.write(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=dict(os.environ),
            )

            if process.stdout:
                for line in process.stdout:
                    log.write(line.rstrip())

            process.wait()
            if process.returncode == 0:
                status.update(f"[green]Status: {title} Completed Successfully![/green]")
                log.write(f"\n[bold green]✓ {title} finished with code 0.[/bold green]")
            else:
                status.update(f"[red]Status: {title} Failed (Code {process.returncode})[/red]")
                log.write(f"\n[bold red]✗ {title} failed with return code {process.returncode}.[/bold red]")
        except Exception as exc:
            status.update(f"[red]Status: Error launching {title}[/red]")
            log.write(f"[bold red]Execution error: {exc}[/bold red]")
        finally:
            self.is_running = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Textual RAG evaluation dashboard TUI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    _args = parse_args(argv)
    app = EvalTUI()
    app.run()


if __name__ == "__main__":
    main()
