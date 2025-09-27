"""Lightweight accessors for helpers defined in the numbered pipeline scripts.

The numbered filenames (e.g. ``02_query.py``) make direct imports awkward because
``from scripts import 02_query`` is not valid Python syntax. This module exposes the
key helpers via lazy attribute access so importing ``scripts`` does not require every
optional dependency to be installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

_EXPORT_MAP: Dict[str, tuple[str, str]] = {
    "ingest": ("scripts.00_ingest", "ingest"),
    "preview": ("scripts.00_ingest", "preview"),
    "create_retrieval_store": ("scripts.02_query", "create_retrieval_store"),
    "retrieve_contexts": ("scripts.02_query", "retrieve_contexts"),
    "compose_messages": ("scripts.02_query", "compose_messages"),
    "call_chat_model": ("scripts.02_query", "call_chat_model"),
    "clean_snippet": ("scripts.02_query", "clean_snippet"),
    "MissingAPIKeyError": ("scripts.02_query", "MissingAPIKeyError"),
    "LLMInvocationError": ("scripts.02_query", "LLMInvocationError"),
    "build_llm_client": ("scripts.04_llm_api", "build_llm_client"),
    "assemble_prompt": ("scripts.04_llm_api", "assemble_prompt"),
    "llm_call": ("scripts.04_llm_api", "call_llm"),
    "llm_pretty_print": ("scripts.04_llm_api", "pretty_print"),
    "load_context_from_file": ("scripts.04_llm_api", "load_context_from_file"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'scripts' has no attribute '{name}'")
    module_name, attribute = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attribute)


def __dir__() -> list[str]:
    return sorted(__all__)

