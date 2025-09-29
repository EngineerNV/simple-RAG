# data/

This directory holds all runtime assets for the retrieval pipeline. The repository includes empty `corpus/` and `chroma/` folders so you can start experimenting immediately—no need to create them by hand.

## Structure

- `corpus/` — Drop your source documents here (markdown or plain text work best with the default ingestion script). Files are ignored by version control so you can keep your own notes locally.
- `chroma/` — `scripts/01_build_index.py` recreates this directory on each run and stores the persisted Chroma collection. Delete it between experiments to rebuild from scratch.

## Workflow tips

1. **Populate the corpus:** copy markdown files into `corpus/`, then run `python scripts/00_ingest.py` to verify chunking.
2. **Persist embeddings:** execute `python scripts/01_build_index.py` after updating the corpus so the retriever uses the latest content.
3. **Keep things tidy:** it is safe to clear `chroma/` or replace files under `corpus/` whenever you want to iterate on the dataset.

The `.gitkeep` placeholders ensure the directories stay in git even when they are empty—you can ignore or delete them during local runs.
