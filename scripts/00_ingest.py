"""00_ingest.py — baseline ingestion utility for the RAG homework.

Teacher briefing
-----------------
This is the only fully implemented step. It ensures every learner begins with the
same chunking strategy before experimenting with embeddings, retrieval, and agents.

What the script already provides
--------------------------------
* Reads markdown-like files from ``data/corpus/`` (configurable via ``CORPUS_DIR``).
* Splits documents into token-budgeted sections with sensible overlap so chunks align
  with embedding context windows.
* Returns ``List[Document]`` objects annotated with source metadata ready for embedding.

Learner guidance
----------------
- Extend the script only if you need additional preprocessing. Preserve the public
  ``ingest()`` contract so later milestones can import it directly.
- Use ``preview()`` to inspect the chunk hierarchy and confirm metadata is set up for
  retrieval explanations.
"""

import os  # Handle filesystem navigation for the corpus directory
from typing import Iterable, List  # Describe the list of LangChain Document objects returned

from langchain_core.documents import Document  # Represent individual text chunks with metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------- Config ----------------------------
CORPUS_DIR = os.environ.get("CORPUS_DIR", os.path.join("data", "corpus"))

# Configure chunking in token units so the ingestion step aligns with embedding models.
DEFAULT_TIKTOKEN_MODEL = os.environ.get("INGEST_TIKTOKEN_MODEL", "text-embedding-3-small")
CHUNK_SIZE_TOKENS = int(os.environ.get("INGEST_CHUNK_SIZE", "400"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("INGEST_CHUNK_OVERLAP", "80"))


# ---------------------------- Helpers ----------------------------
def list_corpus_files(corpus_dir: str) -> List[str]:
    return [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)
            if os.path.isfile(os.path.join(corpus_dir, f)) and not f.startswith('.')]


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _approximate_token_length(text: str) -> int:
    """Fallback token estimate when a real tokenizer is unavailable."""

    return max(1, len(text.split()))


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Return a token-aware text splitter with a sensible amount of overlap."""

    # Guard against accidental misconfiguration that sets overlap >= chunk size.
    overlap = max(0, min(CHUNK_OVERLAP_TOKENS, max(CHUNK_SIZE_TOKENS - 1, 0)))
    chunk_size = max(1, CHUNK_SIZE_TOKENS)

    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=DEFAULT_TIKTOKEN_MODEL,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    except Exception:  # pragma: no cover - network restricted environments fall back here
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=_approximate_token_length,
        )


_SPLITTER = _build_splitter()


def split_markdown(markdown_text: str, *, source_path: str) -> List[Document]:
    """Split text into token-bounded chunks while retaining document metadata."""

    base_metadata = {
        "source": os.path.basename(source_path),
        "source_path": source_path,
    }
    documents = _SPLITTER.create_documents(
        [markdown_text],
        metadatas=[base_metadata],
    )

    total_chunks = len(documents)
    for idx, doc in enumerate(documents):
        metadata = getattr(doc, "metadata", {}) or {}
        metadata.update({
            "chunk_index": idx,
            "chunk_count": total_chunks,
        })
        doc.metadata = metadata

    return documents


# (Persistence helpers removed – not needed now.)


# ---------------------------- Core processing ----------------------------
def process_file(path: str) -> Iterable[Document]:
    return split_markdown(read_text(path), source_path=path)


def ingest() -> List:
    files = list_corpus_files(CORPUS_DIR)
    if not files:
        print(f"No files in {CORPUS_DIR}")
        return []
    docs_all = []
    for p in files:
        docs = process_file(p)
        print(f"{os.path.basename(p)} -> {len(docs)} sections")
        docs_all.extend(docs)
    print(f"Total sections: {len(docs_all)}")
    return docs_all


def preview(docs, n: int = 3):
    for i, d in enumerate(docs[:n]):
        metadata = getattr(d, "metadata", {}) or {}
        source = metadata.get("source", "<unknown source>")
        idx = metadata.get("chunk_index")
        count = metadata.get("chunk_count")
        chunk_label = f"chunk {idx + 1}/{count}" if isinstance(idx, int) and isinstance(count, int) else "chunk"
        print(f"[{i}] {source} • {chunk_label}")
        snippet = d.page_content.strip().replace('\n', ' ')
        print(snippet[:200] + ('...' if len(snippet) > 200 else ''))


if __name__ == '__main__':
    preview(ingest())


