from __future__ import annotations

import importlib
from pathlib import Path

import pytest

delete_chroma = importlib.import_module("scripts.delete_chroma")


def test_ensure_under_allows_nested_path(tmp_path: Path) -> None:
    parent = tmp_path / "data"
    child = parent / "chroma"
    parent.mkdir()
    delete_chroma.ensure_under(parent, child)  # should not raise


def test_ensure_under_rejects_outside_path(tmp_path: Path) -> None:
    parent = tmp_path / "data"
    parent.mkdir()
    outside = tmp_path / "elsewhere"
    with pytest.raises(ValueError, match="Refusing to operate outside"):
        delete_chroma.ensure_under(parent, outside)


def test_clear_directory_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert delete_chroma.clear_directory(tmp_path / "does-not-exist") == []


def test_clear_directory_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    target.mkdir()
    (target / "file.bin").write_text("data", encoding="utf-8")
    (target / "subdir").mkdir()

    deleted = delete_chroma.clear_directory(target, dry_run=True)

    assert {p.name for p in deleted} == {"file.bin", "subdir"}
    assert (target / "file.bin").exists()
    assert (target / "subdir").exists()


def test_clear_directory_removes_files_and_subdirs_but_keeps_parent(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    target.mkdir()
    (target / "file.bin").write_text("data", encoding="utf-8")
    nested = target / "subdir"
    nested.mkdir()
    (nested / "inner.txt").write_text("inner", encoding="utf-8")

    deleted = delete_chroma.clear_directory(target)

    assert len(deleted) == 2
    assert target.exists()
    assert list(target.iterdir()) == []


def test_resolve_repo_root_uses_explicit_root(tmp_path: Path) -> None:
    assert delete_chroma.resolve_repo_root(str(tmp_path)) == tmp_path.resolve()


def test_resolve_repo_root_defaults_to_repo_root() -> None:
    root = delete_chroma.resolve_repo_root(None)
    assert (root / "scripts" / "delete_chroma.py").exists()


def test_main_missing_data_dir_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = delete_chroma.main(["--root", str(tmp_path), "--force"])
    assert rc == 1
    assert "Data directory not found" in capsys.readouterr().out


def test_main_nothing_to_delete_returns_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "data" / "chroma").mkdir(parents=True)
    rc = delete_chroma.main(["--root", str(tmp_path), "--force"])
    assert rc == 0
    assert "Nothing to delete" in capsys.readouterr().out


def test_main_dry_run_does_not_delete(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chroma_dir = tmp_path / "data" / "chroma"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "index.bin").write_text("data", encoding="utf-8")

    rc = delete_chroma.main(["--root", str(tmp_path), "--dry-run"])

    assert rc == 0
    assert (chroma_dir / "index.bin").exists()
    assert "Dry run complete" in capsys.readouterr().out


def test_main_force_deletes_contents(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chroma_dir = tmp_path / "data" / "chroma"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "index.bin").write_text("data", encoding="utf-8")

    rc = delete_chroma.main(["--root", str(tmp_path), "--force"])

    assert rc == 0
    assert chroma_dir.exists()
    assert list(chroma_dir.iterdir()) == []
    assert "Done. Deleted 1 items." in capsys.readouterr().out


def test_main_without_force_prompts_and_aborts_on_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chroma_dir = tmp_path / "data" / "chroma"
    chroma_dir.mkdir(parents=True)
    (chroma_dir / "index.bin").write_text("data", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    rc = delete_chroma.main(["--root", str(tmp_path)])

    assert rc == 0
    assert (chroma_dir / "index.bin").exists()
    assert "Aborted." in capsys.readouterr().out
