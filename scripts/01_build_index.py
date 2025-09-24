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

from pathlib import Path  # Resolve the directory where the Chroma DB should live
from typing import Iterable  # Provide typing for the document list flowing through the script

from langchain_community.embeddings import (  # Generate vector representations of text chunks
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import (  # Persist embeddings in a local Chroma collection
    Chroma,
)
from langchain_core.documents import Document  # Describe the structure of LangChain documents
from langchain_core.embeddings import Embeddings  # Type hint for embedding models to keep signatures clear

from scripts import ingest  # Reuse the ingestion baseline supplied in 00_ingest

CHROMA_DIR = Path("data") / "chroma"


def build_embeddings_model() -> Embeddings:
    """Return a configured embedding model to keep indexing and retrieval aligned."""
    # TODO: Instantiate HuggingFaceEmbeddings() or another Embeddings implementation.
    raise NotImplementedError


def load_documents() -> Iterable[Document]:
    """Run the ingestion baseline and surface the resulting documents."""
    # TODO: Call ingest() and add any extra filtering or metadata adjustments.
    raise NotImplementedError


def persist_chroma(docs: Iterable[Document], embeddings: Embeddings) -> Chroma:
    """Create or update a Chroma collection that stores the supplied documents."""
    # TODO: Use Chroma.from_documents(...) or Chroma(persist_directory=...) to write vectors.
    raise NotImplementedError


def summarize_run(store: Chroma) -> None:
    """Print key facts that help graders confirm the index was built correctly."""
    # TODO: Inspect store._collection or store.get() to report counts and persistence paths.
    raise NotImplementedError


def main() -> None:
    """CLI entry point expected by the assignment."""
    # TODO: Parse CLI args, orchestrate embedding + persistence, and call summarize_run.
    raise NotImplementedError


if __name__ == "__main__":
    main()

