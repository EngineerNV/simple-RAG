
"""
01_build_index.py

This script is the second step in the RAG pipeline. It is responsible for embedding the text chunks and storing them in a Chroma vector database.

---
LEARNING GUIDE (do not delete):
- Your goal: Take the text chunks from the previous step and embed them using a model, then store them in Chroma.
- Try to:
    * Load your text chunks from disk (from 00_ingest.py output).
    * Look up a simple embedding model (hint: sentence-transformers, OpenAI, etc.).
    * Install and import Chroma (chromadb).
    * Store the embeddings in a persistent Chroma DB (use data/chroma/ as the persist directory).
- You may need to install extra packages (see requirements.txt).
- Add comments to explain your code and what you learned.
"""

# Your code starts below
"""Build index step (scaffold)

Plan:
 1. Import ingest.ingest() to get Document objects in-memory.
 2. Choose embedding approach:
            a) Use LangChain HuggingFaceEmbeddings (model same as later query) OR
            b) Manually use SentenceTransformer and pass embeddings to Chroma client.
 3. Create / load persistent Chroma vector store at data/chroma.
 4. Add documents with their metadata (headers already in doc.metadata).
 5. Persist store and print basic stats (count of vectors).
 6. (Optional later) add idempotency: skip if already embedded unless --force.

Next actions (to be implemented):
    - from scripts.00_ingest import ingest
    - from langchain_community.vectorstores import Chroma
    - from langchain_community.embeddings import HuggingFaceEmbeddings
    - call docs = ingest(save=False)
    - embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    - vs = Chroma.from_documents(docs, embedding=embeddings, persist_directory="data/chroma")
    - vs.persist()
    - print(len(vs.get()['ids']))

NOTE: Implementation deferred until confirmation to proceed.
"""

from pathlib import Path

CHROMA_DIR = Path("data") / "chroma"

def main():
        print("Index build scaffold ready. Implement logic once confirmed.")

if __name__ == "__main__":
        main()

# Try running this script and see what happens!
