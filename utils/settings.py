"""Central location for filesystem and model defaults shared across scripts."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):  # type: ignore[return-type]
        return False

# Entry-point scripts call load_dotenv() again themselves, but CHROMA_DIR below
# is resolved at import time — often before those calls run — so load here too,
# otherwise a .env-only CHROMA_DB_PATH would be silently ignored.
load_dotenv()


def _resolve_chroma_dir() -> Path:
    # Treat an empty CHROMA_DB_PATH (e.g. "CHROMA_DB_PATH=" in .env) as unset;
    # Path("") resolves to the current directory, not our default.
    raw = os.environ.get("CHROMA_DB_PATH", "").strip()
    return Path(raw) if raw else Path("data", "chroma")


# Honour the CHROMA_DB_PATH documented in .env.example; default matches the
# path the pipeline has always used.
CHROMA_DIR = _resolve_chroma_dir()

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
