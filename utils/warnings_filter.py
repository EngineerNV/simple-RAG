"""Shared LangChain deprecation-warning suppression.

Every entry-point script used to carry an identical copy of this filter block;
they now call :func:`suppress_langchain_warnings` instead.
"""

import warnings


def suppress_langchain_warnings() -> None:
    """Install LangChain deprecation-warning filters (idempotent)."""
    try:  # Best-effort: some environments provide this warning class
        from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore

        warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    except ImportError:
        # Fallback to message-based filters if the class isn't importable
        warnings.filterwarnings(
            "ignore",
            message=r".*HuggingFaceEmbeddings.*was deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*manual persistence method is no longer supported.*",
        )
