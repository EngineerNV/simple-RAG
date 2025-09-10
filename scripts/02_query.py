"""
02_query.py

This script is the third step in the RAG pipeline. It is responsible for querying the Chroma vector database with a user question and retrieving relevant chunks, then sending them to a cloud LLM for answer generation.

---
LEARNING GUIDE (do not delete):
- Your goal: Query your Chroma DB with a question and get relevant context back.
- Try to:
    * Connect to your Chroma DB (see chromadb docs).
    * Accept a user question (input() or hardcoded for now).
    * Embed the question using the same model as before.
    * Retrieve the most similar chunks from Chroma.
    * (Optional) Send the context and question to a cloud LLM (OpenAI, Azure, etc.) and print the answer.
- Look up how to use chromadb for similarity search.
- Add comments to explain your code and what you learned.

"""

def connect_chroma(db_path):
    """Connect to the Chroma vector database at db_path."""
    # TODO: Implement connection logic
    pass

def get_user_query():
    """Get a query from the user (e.g., via input or function argument)."""
    # TODO: Implement user query input
    pass

def embed_query(query, embedding_model):
    """Embed the user query using the specified embedding model."""
    # TODO: Implement embedding logic
    pass

def retrieve_chunks(query_embedding, db):
    """Retrieve relevant chunks from the vector database using the query embedding."""
    # TODO: Implement retrieval logic
    pass

def generate_answer(query, context, llm):
    """Generate an answer using the LLM and retrieved context."""
    # TODO: Implement answer generation logic
    pass

if __name__ == "__main__":
    # Example usage (replace with your own logic)
    # db = connect_chroma("./data/chroma")
    # query = get_user_query()
    # query_embedding = embed_query(query, embedding_model="your-model")
    # context = retrieve_chunks(query_embedding, db)
    # answer = generate_answer(query, context, llm="your-llm")
    # print(answer)
    pass

# Try running this script and see what happens!
