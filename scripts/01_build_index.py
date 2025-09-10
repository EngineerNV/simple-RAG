
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

def load_chunks(path):
    """Load chunked data from the given path."""
    # TODO: Implement loading logic
    pass

def embed_chunks(chunks, embedding_model):
    """Embed the text chunks using the specified embedding model."""
    # TODO: Implement embedding logic
    pass

def store_embeddings(embeddings, db_path):
    """Store the embeddings in a vector database at db_path."""
    # TODO: Implement storage logic
    pass

if __name__ == "__main__":
    # Example usage (replace with your own logic)
    # chunks = load_chunks("path/to/chunks")
    # embeddings = embed_chunks(chunks, embedding_model="your-model")
    # store_embeddings(embeddings, db_path="./data/chroma")
    pass

# Try running this script and see what happens!
