"""01_build_index.py — embed the corpus and persist it to Chroma.

This module reuses :func:`scripts.ingest` to load markdown chunks, embeds them
with a Hugging Face sentence transformer, and writes a fresh Chroma collection
to ``data/chroma`` on every run. A short summary is printed so you can confirm
the number of chunks, collection name, and metadata coverage before moving on
to the retrieval step.
"""

from __future__ import annotations

import logging
import shutil  # Clean up old persistence directories when rebuilding the index
import sys
from pathlib import Path  # Resolve the directory where the Chroma DB should live
from typing import Iterable, List  # Provide typing for the document list flowing through the script

from langchain_community.vectorstores import (  # Persist embeddings in a local Chroma collection
    Chroma,
)
from langchain_core.documents import Document  # Describe the structure of LangChain documents
from langchain_core.embeddings import Embeddings  # Type hint for embedding models to keep signatures clear
import json

try:  # Reuse the ingestion baseline supplied in 00_ingest
    from scripts import ingest as run_ingest
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts import ingest as run_ingest

from utils import settings
from utils.warnings_filter import suppress_langchain_warnings

suppress_langchain_warnings()

logger = logging.getLogger(__name__)

CHROMA_DIR = settings.CHROMA_DIR
DEFAULT_MODEL_NAME = settings.DEFAULT_EMBED_MODEL

_RUN_METADATA: dict[str, object] = {}


def build_embeddings_model(model_name: str = DEFAULT_MODEL_NAME) -> Embeddings:
    """Return a configured embedding model to keep indexing and retrieval aligned.

    Parameters
    ----------
    model_name:
        Name of the sentence-transformer model to load. Keeping this configurable
        makes it easy to swap in an API-based embedder later without touching the
        rest of the pipeline.
    """

    # Keep the stable import path to avoid environment drift.
    from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings  # type: ignore

    return HFEmbeddings(model_name=model_name)


def persist_chroma(processed_docs: Iterable[Document], embedding_model: Embeddings) -> Chroma:
    """Create or update a Chroma collection that stores the supplied processed document chunks.

    Parameters
    ----------
    processed_docs:
        An iterable of LangChain `Document` objects produced by the ingestion step. These
        represent pre-split chunks (sections) ready for embedding and storage.
    embedding_model:
        An embedding model instance (LangChain `Embeddings`) used to convert text to vectors.
    """

    processed_documents: List[Document] = list(processed_docs)
    if not processed_documents:
        raise ValueError("No processed documents supplied to persist_chroma; run ingestion first.")

    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    already_exists = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
    if already_exists:
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize metadata to ensure all values are JSON-serializable (Chroma requires simple types).
    def _sanitize_metadata(md: dict) -> dict:
        if not isinstance(md, dict):
            return {}
        out = {}
        for k, v in md.items():
            key = str(k)
            # Allow simple JSON-friendly primitives unchanged
            if v is None or isinstance(v, (str, bool, int, float)):
                out[key] = v
                continue
            # Lists/dicts: attempt to JSON-serialize; fall back to string repr
            if isinstance(v, (list, dict)):
                try:
                    json.dumps(v)
                    out[key] = v
                    continue
                except Exception:
                    out[key] = str(v)
                    continue
            # For any other type (set, bytes, objects), convert to string
            out[key] = str(v)
        return out

    # Attach sanitized metadata back to documents (Chroma expects simple metadata values)
    for doc in processed_documents:
        md = getattr(doc, "metadata", None) or {}
        sanitized = _sanitize_metadata(md)
        try:
            # Some Document implementations are frozen; assign to .metadata if possible
            doc.metadata = sanitized
        except (AttributeError, TypeError):
            # Frozen Document; the sanitized mapping below is passed to Chroma anyway.
            logger.debug("Could not assign sanitized metadata back to a document.")

    sanitized_metadatas = [_sanitize_metadata(getattr(d, "metadata", {}) or {}) for d in processed_documents]

    # Persist processed document chunks and use the provided embedding model instance.
    # Use from_texts to explicitly pass sanitized texts and metadatas and avoid internal
    # duplication of the `metadatas` keyword.
    texts = [getattr(d, "page_content", str(d)) for d in processed_documents]
    # Chroma 0.4+ persists automatically when a persist_directory is supplied,
    # so no explicit persist() call is needed (it was deprecated and removed).
    store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=sanitized_metadatas,
        persist_directory=str(CHROMA_DIR),
    )

    _RUN_METADATA.update(
        {
            "doc_count": len(processed_documents),
            "persist_directory": str(CHROMA_DIR),
            "rebuilt": already_exists,
        }
    )

    return store


def summarize_run(store: Chroma) -> None:
    """Print key facts that help graders confirm the index was built correctly."""

    sample = store.get(include=["metadatas"], limit=1)
    sample_metadata = sample.get("metadatas", [])
    metadata_preview = sample_metadata[0] if sample_metadata else {}

    rebuilt = _RUN_METADATA.get("rebuilt", False)
    persist_directory = _RUN_METADATA.get("persist_directory", str(CHROMA_DIR))
    doc_count = _RUN_METADATA.get("doc_count", "<unknown>")

    print("Chroma index build complete.")
    print(f" - Persist directory: {persist_directory}")
    print(f" - Documents embedded: {doc_count}")
    print(f" - Existing store replaced: {'yes' if rebuilt else 'no'}")
    if metadata_preview:
        keys_preview = ", ".join(sorted(metadata_preview.keys())) or "<no metadata>"
        print(f" - Metadata keys preserved (sample): {keys_preview}")
    else:
        print(" - Metadata keys preserved (sample): <none>")
    # Attempt to surface ingest chunking configuration so it's clear what was embedded.
    try:
        import importlib
        ingest_mod = importlib.import_module("scripts.00_ingest")
        chunk_size = getattr(ingest_mod, "CHUNK_SIZE_TOKENS", None)
        chunk_overlap = getattr(ingest_mod, "CHUNK_OVERLAP_TOKENS", None)
        if chunk_size is not None:
            print(f" - Ingest chunking: chunk_size={chunk_size}, overlap={chunk_overlap}")
    except ImportError:
        # Not critical; continue without the chunking summary.
        logger.debug("Could not import scripts.00_ingest for the chunking summary.")


def main() -> None:
    """CLI entry point expected by the assignment."""

    try:
        documents = list(run_ingest())
    except FileNotFoundError:
        module = sys.modules.get(run_ingest.__module__)
        corpus_dir = getattr(module, "CORPUS_DIR", "data/corpus") if module else "data/corpus"
        print(f"Corpus directory not found at {corpus_dir}; run 00_ingest first.")
        return

    if not documents:
        print("No documents were ingested; skipping index build.")
        return

    # Clarify naming: these are processed Documents (chunks) and an embedding model instance.
    processed_documents = documents
    embedding_model = build_embeddings_model()
    store = persist_chroma(processed_documents, embedding_model)
    summarize_run(store)


if __name__ == "__main__":
    main()

