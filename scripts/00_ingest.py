"""00_ingest.py

Ultra‑simple ingestion:
    * Split markdown into sections by headers (#, ##, ###) using MarkdownHeaderTextSplitter.
    * Keep ALL sections (including empty or 'Meta').
    * Return list[Document]; no extra metadata manipulation, no persistence by default.
    * Designed for piping directly into LangChain vector store creation.

Minimal surface – easy to extend later (e.g., filtering, token metrics) without clutter now.
"""

import os
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

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


## Token counting intentionally omitted (keep simple per plan)


## Removed pagination logic for simplicity (each section == one chunk)


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

