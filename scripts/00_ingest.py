"""00_ingest.py — baseline ingestion utility for the RAG homework.

Teacher briefing
-----------------
This is the only fully implemented step. It ensures every learner begins with the
same chunking strategy before experimenting with embeddings, retrieval, and agents.

What the script already provides
--------------------------------
* Reads markdown-like files from ``data/corpus/`` (configurable via ``CORPUS_DIR``).
* Splits documents into sections keyed by headers (#, ##, ###) using
  ``MarkdownHeaderTextSplitter``.
* Returns ``List[Document]`` objects with header metadata intact—ready for embedding.

Learner guidance
----------------
- Extend the script only if you need additional preprocessing. Preserve the public
  ``ingest()`` contract so later milestones can import it directly.
- Use ``preview()`` to inspect the chunk hierarchy and confirm metadata is set up for
  retrieval explanations.
"""

import os  # Handle filesystem navigation for the corpus directory
from typing import List  # Describe the list of LangChain Document objects returned

from langchain_text_splitters import (  # Split markdown files into header-aware chunks
    MarkdownHeaderTextSplitter,
)

# ---------------------------- Config ----------------------------
CORPUS_DIR = os.environ.get("CORPUS_DIR", os.path.join("data", "corpus"))
HEADER_LEVELS = ["#", "##", "###"]  # order matters


# ---------------------------- Helpers ----------------------------
def list_corpus_files(corpus_dir: str) -> List[str]:
    return [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)
            if os.path.isfile(os.path.join(corpus_dir, f)) and not f.startswith('.')]


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def split_markdown(markdown_text: str):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[(h, h) for h in HEADER_LEVELS])
    return splitter.split_text(markdown_text)


# (Persistence helpers removed – not needed now.)


# ---------------------------- Core processing ----------------------------
def process_file(path: str):
    return split_markdown(read_text(path))


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
        headers = [d.metadata.get(h) for h in HEADER_LEVELS if d.metadata.get(h)]
        print(f"[{i}] {' > '.join(headers) if headers else '<no header>'}")
        snippet = d.page_content.strip().replace('\n', ' ')
        print(snippet[:200] + ('...' if len(snippet) > 200 else ''))


if __name__ == '__main__':
    preview(ingest())


