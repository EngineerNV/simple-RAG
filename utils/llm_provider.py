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


PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


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

    # If only explicit key provided, determine provider
    if explicit_key:
        detected = auto_detect_provider()
        return (detected[0] if detected else "openai", explicit_key)

    # If explicit provider hint provided, check that provider's environment variable first
    if provider_hint:
        env_var = PROVIDER_ENV_KEYS.get(provider_hint.lower())
        if env_var and os.environ.get(env_var):
            return (provider_hint, os.environ[env_var])
        # Fall back to any auto-detected key
        detected = auto_detect_provider()
        if detected:
            return (provider_hint, detected[1])
        return (provider_hint, None)

    # Auto-detect from environment
    detected = auto_detect_provider()
    if detected:
        return detected

    # No key found anywhere
    return ("openai", None)


def is_openai_reasoning_model(model_name: str) -> bool:
    """True for OpenAI's reasoning-family models (o1/o3/o4, gpt-5+), which
    reject the legacy ``max_tokens``/non-default-``temperature`` params that
    the pinned langchain-openai==0.1.25 always sends -- see the workarounds
    in ``build_chat_model`` below.
    """
    name = model_name.lower()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


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
            "openai_api_key": api_key,
        }
        if is_openai_reasoning_model(model_name):
            # temperature=1 (the only value these models accept) and
            # max_completion_tokens via model_kwargs (the renamed limit
            # param) route around what the pinned client would otherwise
            # send. tiktoken_model_name="gpt-4" is a separate workaround:
            # ConversationSummaryBufferMemory's token counting raises
            # NotImplementedError for any model name it doesn't recognize,
            # so this points it at a recognized one -- an approximation
            # (different tokenizer), fine for a buffer-pruning trigger but
            # never for actual token billing.
            init_kwargs["temperature"] = 1
            init_kwargs["model_kwargs"] = {"max_completion_tokens": max_tokens}
            init_kwargs["tiktoken_model_name"] = "gpt-4"
        else:
            init_kwargs["temperature"] = temperature
            init_kwargs["max_tokens"] = max_tokens
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
