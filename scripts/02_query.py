"""02_query.py — connect the retriever and chat model through a LangChain agent.

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

from pathlib import Path  # Locate the persisted Chroma directory on disk
from typing import List  # Describe collections of documents and tool outputs

from langchain.agents import (  # Build and execute a retrieval-augmented agent
    AgentExecutor,
    create_react_agent,
)
from langchain_core.language_models.chat_models import (  # Type hint for chat-based LLM clients
    BaseChatModel,
)
from langchain_core.prompts import ChatPromptTemplate  # Compose prompts for the agent and tools
from langchain_core.runnables import Runnable  # Represent retrievers/tools in a generic way
from langchain_community.embeddings import (  # Recreate the embedding model used for indexing
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import (  # Load the persisted Chroma collection
    Chroma,
)
from langchain_core.documents import Document  # Provide structure for retrieved context snippets
from langchain.tools import Tool  # Register the retriever so the agent can invoke it

from scripts import ingest  # Optional: reuse ingestion for ad-hoc checks or rebuilding


def load_vector_store(persist_dir: Path, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Connect to the Chroma collection built during the indexing milestone."""
    # TODO: Use Chroma(persist_directory=..., embedding_function=...) to load vectors.
    raise NotImplementedError


def build_retriever(store: Chroma, k: int) -> Runnable:
    """Expose the vector store as a retriever runnable for the agent."""
    # TODO: Call store.as_retriever(search_kwargs={"k": k}) or wrap manually.
    raise NotImplementedError


def make_retriever_tool(retriever: Runnable) -> Tool:
    """Wrap the retriever in a LangChain Tool with a descriptive name and docstring."""
    # TODO: Provide Tool.from_function(...) or Tool(...) with callable=...
    raise NotImplementedError


def load_chat_model(model_name: str) -> BaseChatModel:
    """Instantiate the chat model that will drive the agent's reasoning."""
    # TODO: Return ChatOpenAI(model=model_name, api_key=...) or an Anthropic equivalent.
    raise NotImplementedError


def build_agent(llm: BaseChatModel, tool: Tool) -> AgentExecutor:
    """Create an agent that can consult the retriever tool before answering."""
    # TODO: Create a prompt template, call create_react_agent, and wrap it in AgentExecutor.
    raise NotImplementedError


def run_agent(agent: AgentExecutor, question: str) -> tuple[str, List[Document], List[dict]]:
    """Execute the agent on a question and collect intermediate steps for reporting."""
    # TODO: agent.invoke(..., return_intermediate_steps=True) to capture answer and retrieved docs.
    raise NotImplementedError


def display_result(answer: str, contexts: List[Document], steps: List[dict]) -> None:
    """Pretty-print retrieved evidence and the final answer for manual grading."""
    # TODO: Format intermediate steps, retrieved documents, and the final response.
    raise NotImplementedError


def main() -> None:
    """CLI entry point: wire args -> retriever -> agent -> output."""
    # TODO: Parse arguments, instantiate dependencies above, and call display_result.
    raise NotImplementedError


if __name__ == "__main__":
    main()

