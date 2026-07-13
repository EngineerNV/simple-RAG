# data/

This directory holds all runtime assets for the retrieval pipeline. The repository ships `corpus/` (with a sample `Pokémon.MD` knowledge base) so you can start experimenting immediately; `chroma/` is **generated** by the index build — it is not committed.

## Structure

- `corpus/` — Drop your source documents here (markdown works best with the default ingestion script). Your own files are ignored by version control, so you can keep local notes here; only the sample `Pokémon.MD` is tracked.
- `chroma/` — Created (and recreated) by `scripts/01_build_index.py` on each run to store the persisted Chroma collection. Delete it between experiments to rebuild from scratch (`python scripts/delete_chroma.py`).

## Workflow tips

1. **Populate the corpus:** copy markdown files into `corpus/`, then run `python scripts/00_ingest.py` to verify chunking.
2. **Persist embeddings:** execute `python scripts/01_build_index.py` after updating the corpus so the retriever uses the latest content.
3. **Keep things tidy:** it is safe to clear `chroma/` or replace files under `corpus/` whenever you want to iterate on the dataset.

The Chroma path can be moved with the `CHROMA_DB_PATH` environment variable (see `.env.example`).
