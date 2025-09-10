"""
00_ingest.py

This script is the first step in the RAG pipeline. It is responsible for loading documents from the data/corpus/ directory, converting them to text (if needed), and chunking them for later embedding.

---
LEARNING GUIDE (do not delete):
- Your goal: Load all files from data/corpus/ and split them into text chunks.
- Try to:
    * List all files in the corpus directory (hint: use os.listdir or pathlib).
    * Read the contents of each file (try open(), .read(), etc.).
    * Write a function to split text into chunks (look up 'text chunking python').
    * Save the chunks to disk (as .txt, .json, or .pkl files in a new folder if you want).
- Don't worry about embeddings or Chroma yet!
- Add comments to explain your code and what you learned.
"""


import os

# Use environment variable for corpus directory, fallback to default if not set
CORPUS_DIR = os.environ.get('CORPUS_DIR', os.path.join('data', 'corpus'))

def list_corpus_files(corpus_dir):
    """List all files in the given corpus directory. Returns a list of file paths."""
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass

def read_file(filepath):
    """Read the contents of a file and return as a string."""
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass

def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately chunk_size characters. Returns a list of text chunks."""
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass
    # TODO: Implement this function
    pass

def save_chunks(chunks, out_dir, base_filename):
    """Save the list of chunks to disk (e.g., as .txt or .json)."""
    # TODO: Implement this function (optional)
    pass
    # TODO: Implement this function (optional)
    pass
    # TODO: Implement this function (optional)
    pass

def main():
    # 1. List all files in the corpus directory
    files = list_corpus_files(CORPUS_DIR)
    # 2. Read each file and print the first 100 characters
    for file in files:
        text = read_file(file)
        print(f"First 100 chars of {file}: {text[:100]}")
        # 3. Chunk the text
        chunks = chunk_text(text)
        # 4. Save your chunks somewhere (optional)
        # save_chunks(chunks, 'data/chunks', os.path.basename(file))

if __name__ == "__main__":
    main()

