# data/

This folder is for all data-related assets in your RAG project.

## Structure

- `corpus/` — Place your source documents here. These can be PDFs, text files, or any data you want to use for retrieval.
- `chroma/` — This is where the Chroma vector database will store its persistent data. You don't need to manually edit anything here.

## How to use

1. **Add Documents:**
   - Drop your documents into the `corpus/` folder. Try different file types and see how ingestion scripts handle them.
   - Experiment with adding, removing, or modifying files to see how it affects the RAG pipeline.

2. **Chroma Persistence:**
   - After running the indexing script, check the `chroma/` folder to see what files are created.
   - Try deleting the `chroma/` folder and re-running the index script to observe what happens.

## Learning Opportunities

- Try to write a script that lists all files in `corpus/`.
- Experiment with reading a file from `corpus/` in Python.
- Look up how to check if `chroma/` exists and create it if missing.

> **Tip:** Use comments in your scripts to document what each step does. This will help you and others learn more effectively.
