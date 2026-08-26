"""00_ingest.py — baseline ingestion utility for the RAG homework.

This module reads markdown files from the ``data/corpus`` directory, splits them
into section-scoped, token-bounded chunks, and returns a list of LangChain Document
objects. Each Document contains the chunked text plus metadata about its source
file, its position within that file, and the heading(s) it was found under.
"""

import os
import re
import warnings
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Suppress noisy deprecation warnings without changing packages.
try:  # Best-effort: some environments provide this warning class
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    # Fallback to message-based filters if the class isn't importable
    warnings.filterwarnings(
        "ignore",
        message=r".*HuggingFaceEmbeddings.*was deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*manual persistence method is no longer supported.*",
    )

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

# Pass 1: split at heading boundaries. Metadata keys intentionally mirror the
# raw heading markers ("#" through "####") so a chunk's section title is
# visible directly in doc.metadata alongside source/chunk_index. H4 covers
# entries nested under a thematic H3 subgroup (e.g. a species under a
# "Rock & Ground Dwellers" grouping); most sections only go two or three
# levels deep, which the splitter handles fine by omitting unused keys.
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "#"), ("##", "##"), ("###", "###"), ("####", "####")],
    strip_headers=True,
)

# Horizontal rules ("---", "***", "___") are pure visual separators in our
# corpus files; strip them before header-splitting so they don't bleed into
# a section's content or form their own near-empty chunk.
_HORIZONTAL_RULE_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)


def split_markdown(markdown_text: str, *, source_path: str) -> List[Document]:
    """Split markdown into section-scoped, token-bounded chunks.

    Pass 1 (``_HEADER_SPLITTER``) splits the document at heading boundaries so
    every chunk carries its enclosing heading(s) as metadata. Pass 2 re-runs
    the token-aware ``_SPLITTER`` on any section that still exceeds
    ``CHUNK_SIZE_TOKENS``, so long entries still get overlap-preserving
    sub-chunks instead of one oversized chunk.
    """

    base_metadata = {
        "source": os.path.basename(source_path),
        "source_path": source_path,
    }

    cleaned_text = _HORIZONTAL_RULE_RE.sub("", markdown_text)
    sections = [s for s in _HEADER_SPLITTER.split_text(cleaned_text) if s.page_content.strip()]

    documents: List[Document] = []
    for section in sections:
        section_metadata = {**base_metadata, **section.metadata}
        documents.extend(
            _SPLITTER.create_documents([section.page_content.strip()], metadatas=[section_metadata])
        )

    if not documents:
        # Degenerate case (no headings, or headings with no body text at all):
        # fall back to splitting the raw text so a file is never silently dropped.
        stripped = cleaned_text.strip()
        if stripped:
            documents = _SPLITTER.create_documents([stripped], metadatas=[base_metadata])

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
        docs = list(process_file(p))
        print(
            f"{os.path.basename(p)} -> {len(docs)} chunks (section-scoped, max chunk_size={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP_TOKENS})"
        )
        docs_all.extend(docs)
    print(f"Total chunks: {len(docs_all)}")
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


