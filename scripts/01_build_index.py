"""01_build_index.py — embed the corpus and persist it to Chroma.

Teacher briefing
-----------------
By the end of this milestone you should have a reproducible script that transforms
the ingested ``Document`` objects into a Chroma collection. Every later step in
the lab assumes this store exists, so focus on correctness and repeatability.

Implementation checklist
------------------------
1. Import ``ingest`` from ``scripts`` (the helper is re-exported from ``00_ingest``).
2. Choose an embedding model from LangChain (``HuggingFaceEmbeddings`` works offline;
   ``OpenAIEmbeddings`` is an option if you have an API key).
3. Create or connect to a Chroma collection persisted under ``data/chroma/`` using
   the same embedding function you will use during retrieval.
4. Add the documents and confirm their metadata is preserved.
5. Print a concise run report (documents embedded, collection name, persistence dir,
   and whether the call rebuilt an existing store).

Stretch goals
-------------
- Accept CLI flags for ``--collection-name``, ``--force`` rebuilds, and embedding choice.
- Log simple timing stats to highlight slow stages.
- Optionally serialize the chunk list to JSON for debugging or unit tests.
"""

from __future__ import annotations

import shutil  # Clean up old persistence directories when rebuilding the index
import sys
from pathlib import Path  # Resolve the directory where the Chroma DB should live
from typing import Iterable, List  # Provide typing for the document list flowing through the script

from langchain_community.embeddings import (  # Generate vector representations of text chunks
    HuggingFaceEmbeddings,
)
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

CHROMA_DIR = Path("data") / "chroma"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

    return HuggingFaceEmbeddings(model_name=model_name)


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
        except Exception:
            # Fallback: nothing to do; Chroma.from_documents will use the sanitized mapping below
            pass

    sanitized_metadatas = [_sanitize_metadata(getattr(d, "metadata", {}) or {}) for d in processed_documents]

    # Persist processed document chunks and use the provided embedding model instance.
    # Use from_texts to explicitly pass sanitized texts and metadatas and avoid internal
    # duplication of the `metadatas` keyword.
    texts = [getattr(d, "page_content", str(d)) for d in processed_documents]
    store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=sanitized_metadatas,
        persist_directory=str(CHROMA_DIR),
    )
    store.persist()

    collection = store._collection  # type: ignore[attr-defined]
    _RUN_METADATA.update(
        {
            "doc_count": len(processed_documents),
            "collection_name": getattr(collection, "name", "default"),
            "persist_directory": str(CHROMA_DIR),
            "rebuilt": already_exists,
        }
    )

    return store


def summarize_run(store: Chroma) -> None:
    """Print key facts that help graders confirm the index was built correctly."""

    collection = store._collection  # type: ignore[attr-defined]
    count = collection.count()
    sample = store.get(include=["metadatas"], limit=1)
    sample_metadata = sample.get("metadatas", [])
    metadata_preview = sample_metadata[0] if sample_metadata else {}

    rebuilt = _RUN_METADATA.get("rebuilt", False)
    collection_name = _RUN_METADATA.get("collection_name", getattr(collection, "name", "default"))
    persist_directory = _RUN_METADATA.get("persist_directory", str(CHROMA_DIR))
    doc_count = _RUN_METADATA.get("doc_count", count)

    print("Chroma index build complete.")
    print(f" - Collection name: {collection_name}")
    print(f" - Persist directory: {persist_directory}")
    print(f" - Documents embedded: {doc_count}")
    print(f" - Existing store replaced: {'yes' if rebuilt else 'no'}")
    if metadata_preview:
        keys_preview = ", ".join(sorted(metadata_preview.keys())) or "<no metadata>"
        print(f" - Metadata keys preserved (sample): {keys_preview}")
    else:
        print(" - Metadata keys preserved (sample): <none>")


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

