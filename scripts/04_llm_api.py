"""04_llm_api.py — format prompts and execute LLM calls for the LangChain agent.

Teacher briefing
-----------------
This helper consolidates the glue required for milestone 2's agent and milestone 3's
prompt design. It loads API credentials, instantiates a chat client, assembles a
prompt from question + retrieved context, and executes the call while collecting
metadata you can surface in reports.

Implementation checklist
------------------------
1. Read the API key from ``RAG_LLM_API_KEY`` (or a CLI-specified override) and raise
   a helpful error if it is missing.
2. Instantiate the provider's chat model (``ChatOpenAI``, ``ChatAnthropic``, etc.) in a
   dedicated helper so other scripts can reuse it.
3. Build a prompt template that combines system instructions, the user's question, and
   retrieved chunks. Return either a formatted string or a sequence of messages.
4. Call the model, capture response text plus metadata (model, token usage, latency).
5. Provide a CLI entry point that accepts a question and JSON/markdown context list for
   quick manual smoke tests.

Stretch goals
-------------
- Implement retry/backoff handling for rate limits.
- Support streaming output for interactive demos.
- Offer multiple prompt templates (concise vs. verbose) toggled by CLI flags.
"""

from __future__ import annotations

import json  # Allow CLI users to pass context snippets via JSON payloads
import os  # Access environment variables for API keys and configuration
import time  # Capture latency metrics for reporting
from pathlib import Path  # Load context snippets from files during CLI tests
from typing import Iterable, Optional  # Type hints for prompt assembly and inputs

from langchain_core.language_models.chat_models import (  # Provide a common interface for chat models
    BaseChatModel,
)
from langchain_core.messages import AIMessage  # Represent the structured response returned by chat models
from langchain_core.prompts import ChatPromptTemplate  # Build message templates for question + context prompts

LLM_API_KEY_ENV = "RAG_LLM_API_KEY"
DEFAULT_MODEL_NAME = "gpt-3.5-turbo"


def load_api_key(env_var: str = LLM_API_KEY_ENV) -> str:
    """Fetch the LLM API key from the environment.

    Raise a helpful error when the key is missing so the caller knows to set up
    their ``.env`` file or export the variable manually.
    """

    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(
            f"Set the {env_var} environment variable before calling the LLM."
        )
    return api_key


def build_llm_client(api_key: str, model_name: str = DEFAULT_MODEL_NAME) -> BaseChatModel:
    """Instantiate your provider's chat client with the supplied key."""

    # TODO: Return ChatOpenAI(api_key=api_key, model=model_name) or a similar client.
    raise NotImplementedError


def assemble_prompt(
    question: str,
    contexts: Iterable[str],
    instructions: Optional[str] = None,
) -> ChatPromptTemplate:
    """Combine the question, supporting contexts, and optional system message."""

    # TODO: Create a ChatPromptTemplate with system/human messages referencing {context} and {question}.
    raise NotImplementedError


def call_llm(prompt: ChatPromptTemplate, client: BaseChatModel) -> tuple[AIMessage, float]:
    """Send the prompt to the configured client and return the structured response."""

    # TODO: Format prompt with .format_messages(...), time the call, and return (message, latency_seconds).
    raise NotImplementedError


def pretty_print(result: AIMessage, latency_s: float) -> None:
    """Emit the model response with any formatting that helps debugging."""

    # TODO: Include metadata such as model name, token usage, or latency if available.
    raise NotImplementedError


def load_context_from_file(path: Path) -> Iterable[str]:
    """Utility for CLI usage: load context snippets from a JSON or text file."""

    # TODO: Support newline-delimited text or JSON arrays of strings.
    raise NotImplementedError


def main() -> None:
    """Simple CLI hook for manual experimentation."""

    # TODO: Parse CLI args (question + context file), orchestrate the helpers above, and print the output.
    raise NotImplementedError


if __name__ == "__main__":
    main()
