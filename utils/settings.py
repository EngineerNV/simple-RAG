"""Central location for filesystem and model defaults shared across scripts."""

import os
from pathlib import Path

# Honour the CHROMA_DB_PATH documented in .env.example; default matches the
# path the pipeline has always used.
CHROMA_DIR = Path(os.environ.get("CHROMA_DB_PATH", os.path.join("data", "chroma")))

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
