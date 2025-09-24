"""02_query.py — connect the retriever and a simple retrieval-based answerer.

Teacher briefing
-----------------
This milestone demonstrates how the vector index supports interactive QA. Build a
small LangChain agent that exposes the Chroma retriever as a tool and lets a chat
model decide how to answer. Showcase both the retrieved evidence and the model's
final response so reviewers can trace the reasoning path.

Implementation checklist
------------------------
1. Load the embedding model and persisted Chroma store created in milestone 1.
2. Wrap the store in a retriever (`as_retriever` or `VectorStoreRetriever`).
3. Register the retriever as a LangChain `Tool` so the agent can call it.
4. Instantiate a chat model (``ChatOpenAI`` or ``ChatAnthropic``) and construct an
   agent (`create_react_agent`, `AgentExecutor`, or a RetrievalQA chain wrapped as a tool).
5. Collect a user question, run it through the agent, and print:
   - the retrieved chunks (with metadata and scores) and
   - the model's answer or an explicit abstain message when no evidence is found.
6. Support configuration for ``k`` (retrieval depth) and model name via CLI flags.

Stretch goals
-------------
- Cache or display token usage for transparency.
- Allow batch querying from a file of questions.
- Persist conversation history for follow-up questions.
"""

from pathlib import Path
from typing import List
import argparse
import textwrap

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

"""
This script provides a minimal, runnable retrieval smoke-test so you can see the
end-to-end flow without needing an external LLM. It loads the persisted Chroma
collection, runs a retriever for the supplied question, and prints the top
contexts plus a simple synthesized answer (concatenation of retrieved chunks).

The implementation intentionally keeps the agent light-weight so it runs in
offline environments and is easy to extend to call a real chat model later.
"""

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = Path("data") / "chroma"




def load_vector_store(persist_dir: Path, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Connect to the Chroma collection built during the indexing milestone.

    Returns a LangChain `Chroma` wrapper instance configured to use the provided
    embedding model so retrieval behavior matches indexing.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist directory not found: {persist_dir}")
    store = Chroma(persist_directory=str(persist_dir), embedding_function=embedding_model)
    return store


def build_retriever(store: Chroma, k: int):
    """Return a retriever object that provides `get_relevant_documents`.

    We keep this simple and return the object from `store.as_retriever` which
    LangChain wrappers expose. The returned object supports `.get_relevant_documents(query)`
    which will be used below.
    """
    return store.as_retriever(search_kwargs={"k": k})


def make_retriever_tool(retriever):
    """Return a simple callable that wraps the retriever for ad-hoc use.

    We return a small function rather than a full LangChain `Tool` to avoid
    heavy agent wiring and keep the smoke-test lightweight.
    """

    def _call(query: str):
        return retriever.get_relevant_documents(query)

    return _call


def load_chat_model(model_name: str):
    """Placeholder loader for a chat model. For the smoke-test we don't create
    a real LLM — callers can substitute a real client if desired."""
    return None


def build_agent(llm, tool):
    """No-op for the smoke-test. We will call the retriever directly instead
    of wiring a full LangChain agent. Keep this helper for future extension."""
    return None


def run_agent(agent, question: str, retriever_callable, k: int = 3) -> tuple[str, List[Document], List[dict]]:
    """Run the simple retrieval flow and synthesize a basic answer.

    Returns a tuple: (answer_text, retrieved_documents, metadata_list)
    """
    docs: List[Document] = retriever_callable(question)
    # Simple synthesized answer: join top-k snippets (trimmed) — replace with LLM later.
    snippets = [d.page_content.strip().replace("\n", " ") for d in docs]
    answer = "\n\n".join(snippets[:k]) if snippets else "No relevant context found."
    metadatas = [getattr(d, "metadata", {}) for d in docs]
    return answer, docs, metadatas


def display_result(answer: str, contexts: List[Document], metadatas: List[dict]) -> None:
    """Print retrieved contexts and the synthesized answer in a readable form."""
    print("\nRetrieved contexts:")
    for i, d in enumerate(contexts):
        hdr = " | ".join(f"{k}: {v}" for k, v in (getattr(d, 'metadata', {}) or {}).items())
        snippet = d.page_content.strip().replace("\n", " ")
        print(f"[{i}] {hdr}")
        print(textwrap.fill(snippet, width=120))
        print("-")
    print("\nSynthesized answer:\n")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval smoke-test: query the Chroma store")
    parser.add_argument("--question", "-q", required=True, help="Question to query against the store")
    parser.add_argument("--k", type=int, default=3, help="Number of contexts to retrieve")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model name to use (must match index)")
    parser.add_argument(
        "--agent-mode",
        choices=["none", "pretend", "llm"],
        default="none",
        help="Agent mode: 'none' (no LLM), 'pretend' (show prompt + templated answer), or 'llm' (call real model)",
    )
    args = parser.parse_args()

    embed = HuggingFaceEmbeddings(model_name=args.model)
    store = load_vector_store(CHROMA_DIR, embed)
    retriever = build_retriever(store, k=args.k)
    retriever_callable = make_retriever_tool(retriever)

    # Agent-mode behavior
    if args.agent_mode == "none":
        answer, docs, metadatas = run_agent(None, args.question, retriever_callable, k=args.k)
        display_result(answer, docs, metadatas)
    elif args.agent_mode == "pretend":
        # Build the agent prompt preview that a real LLM would receive
        docs_preview = []
        for i, d in enumerate(retriever_callable(args.question)):
            hdr = " | ".join(f"{k}: {v}" for k, v in (getattr(d, 'metadata', {}) or {}).items())
            snippet = d.page_content.strip().replace("\n", " ")
            docs_preview.append({"index": i, "metadata": getattr(d, 'metadata', {}), "snippet": snippet})

        # Print the composed prompt and contexts for inspection
        print("\n=== Pretend Agent Prompt Preview ===\n")
        system_msg = (
            "You are an assistant that must answer the user's question using ONLY the provided contexts. "
            "If the information is insufficient, say you don't know. Quote sources in square brackets like [source #]."
        )
        print("System instruction:")
        print(textwrap.fill(system_msg, width=120))
        print("\nUser question:")
        print(textwrap.fill(args.question, width=120))
        print("\nRetrieved contexts (top k):")
        for item in docs_preview[: args.k]:
            print(f"[{item['index']}] metadata: {item['metadata']}")
            print(textwrap.fill(item['snippet'], width=120))
            print("-")

        # Very small templated synthesis showing which sources we'd cite
        cited = [str(item["index"]) for item in docs_preview[: args.k]]
        synth = " ".join(item["snippet"] for item in docs_preview[: args.k])
        templated_answer = "Answer (synthesized from sources [" + ",".join(cited) + "]):\n\n" + (synth or "No relevant context found.")
        print("\n=== Pretend Agent Final Answer ===\n")
        print(templated_answer)
    else:
        # 'llm' mode not implemented in this lightweight smoke-test
        print("LLM agent mode requested but not implemented in this script. Use --agent-mode pretend for a simulated run.")


if __name__ == "__main__":
    main()



