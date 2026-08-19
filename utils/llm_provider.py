"""Shared LLM provider resolution and client construction.

Centralizes the auto-detect-provider-from-env-vars / build-a-chat-client logic
that used to be copy-pasted across ``scripts/02_query.py``, ``scripts/04_llm_api.py``,
and ``scripts/05_chat_cli.py``.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel


class UnsupportedProviderError(RuntimeError):
    """Raised when an unknown LLM provider identifier is supplied."""


class MissingProviderDependencyError(RuntimeError):
    """Raised when a provider-specific dependency is unavailable."""


def auto_detect_provider() -> tuple[str, str] | None:
    """Auto-detect provider based on which API key is set.

    Returns:
        (provider_name, api_key) tuple or None if no key found.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", os.environ["OPENAI_API_KEY"])
    if os.environ.get("GOOGLE_API_KEY"):
        return ("gemini", os.environ["GOOGLE_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("claude", os.environ["ANTHROPIC_API_KEY"])
    return None


def resolve_provider_and_key(explicit_key: str | None, provider_hint: str | None) -> tuple[str, str | None]:
    """Resolve the provider and API key from explicit args or environment.

    Args:
        explicit_key: Explicit --api-key override
        provider_hint: Explicit --provider override

    Returns:
        (provider, api_key) tuple
    """
    # If both explicit values provided, use them
    if explicit_key and provider_hint:
        return (provider_hint, explicit_key)

    # If only explicit key, need to detect provider
    if explicit_key:
        if provider_hint:
            return (provider_hint, explicit_key)
        # Try to auto-detect from env vars as fallback
        detected = auto_detect_provider()
        return (detected[0], explicit_key) if detected else ("openai", explicit_key)

    # Auto-detect from environment
    detected = auto_detect_provider()
    if detected:
        # If provider_hint given, respect it but use auto-detected key
        if provider_hint:
            return (provider_hint, detected[1])
        return detected

    # No key found anywhere
    return (provider_hint or "openai", None)


def build_chat_model(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """Instantiate the configured provider's chat client with the supplied key."""

    provider = provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-openai'. Install it with `pip install langchain-openai`."
            ) from exc
        init_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": api_key,
        }
        if base_url:
            init_kwargs["openai_api_base"] = base_url
        return ChatOpenAI(**init_kwargs)

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-google-genai'. Install it with `pip install langchain-google-genai`."
            ) from exc
        init_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "google_api_key": api_key,
        }
        return ChatGoogleGenerativeAI(**init_kwargs)  # type: ignore

    elif provider == "claude" or provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-anthropic'. Install it with `pip install langchain-anthropic`."
            ) from exc
        init_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "anthropic_api_key": api_key,
        }
        return ChatAnthropic(**init_kwargs)  # type: ignore

    else:
        raise UnsupportedProviderError(
            f"Unsupported provider '{provider}'. Supported: openai, gemini, claude (anthropic)."
        )
