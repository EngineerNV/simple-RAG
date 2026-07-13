#!/usr/bin/env python3
"""
Delete Chroma artifacts to reset the workspace for a fresh index build.

By default, this script empties data/chroma (without removing the directory
itself).

Usage examples:
  - Dry run (show what would be deleted):
      python scripts/delete_chroma.py --dry-run
  - Delete without interactive prompt:
      python scripts/delete_chroma.py --force
  - Custom project root:
      python scripts/delete_chroma.py --root "C:/path/to/simple-RAG" --force

Notes:
- Uses only Python standard library.
- Handles Windows read-only files by making them writable before deletion.
- Safe-guards to ensure deletion is scoped under the repository's data directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable, List


DATA_DIR_NAME = "data"
DEFAULT_TARGETS = ("chroma",)


def _on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree to handle read-only files (especially on Windows).

    If removal fails due to permission error, try to make the file writable and retry once.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        # Re-raise original exception if recovery fails
        raise


def _iter_children(p: Path) -> Iterable[Path]:
    try:
        return list(p.iterdir())
    except FileNotFoundError:
        return []


def clear_directory(dir_path: Path, *, dry_run: bool = False) -> List[Path]:
    """Remove all children of dir_path, leaving dir_path in place.

    Returns a list of paths that were deleted (or would be deleted in dry-run).
    """
    deleted: List[Path] = []

    if not dir_path.exists():
        # Nothing to clear
        return deleted

    for child in _iter_children(dir_path):
        deleted.append(child)
        if dry_run:
            continue
        if child.is_dir():
            shutil.rmtree(child, onerror=_on_rm_error)
        else:
            try:
                # Make writable then unlink
                os.chmod(child, stat.S_IWRITE)
            except Exception:
                pass
            child.unlink(missing_ok=True)
    return deleted


def resolve_repo_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    # scripts/ is one level under repo root
    return Path(__file__).resolve().parent.parent


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empty data/chroma for a fresh index build.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root (defaults to the repository root inferred from this script).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(DEFAULT_TARGETS),
        default=list(DEFAULT_TARGETS),
        help="Which directories under data/ to clear (default: chroma).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without making changes.",
    )
    parser.add_argument(
        "--force",
        "-y",
        action="store_true",
        help="Do not prompt for confirmation.",
    )
    return parser.parse_args(argv)


def ensure_under(parent: Path, child: Path) -> None:
    """Ensure child is within parent to prevent accidental deletions."""
    try:
        child.resolve().relative_to(parent.resolve())
    except Exception:
        raise ValueError(f"Refusing to operate outside {parent}: {child}")


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    repo_root = resolve_repo_root(args.root)
    data_dir = repo_root / DATA_DIR_NAME

    targets: List[Path] = [data_dir / t for t in args.targets]

    # Safety checks
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    for t in targets:
        ensure_under(data_dir, t)

    planned: List[Path] = []
    for t in targets:
        planned.extend(_iter_children(t))

    if not planned:
        print("Nothing to delete. Targets are already empty or do not exist.")
        return 0

    # Summary
    print("Targets:")
    for t in targets:
        print(f"  - {t}")

    print("\nItems to delete:")
    for p in planned:
        suffix = "/" if p.is_dir() else ""
        print(f"  - {p}{suffix}")

    if not args.force and not args.dry_run:
        ans = input("\nProceed with deletion? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return 0

    actually_deleted: List[Path] = []
    for t in targets:
        actually_deleted.extend(clear_directory(t, dry_run=args.dry_run))

    if args.dry_run:
        print(f"\nDry run complete. {len(actually_deleted)} items would be deleted.")
    else:
        print(f"\nDone. Deleted {len(actually_deleted)} items.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
