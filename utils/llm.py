"""Single home for LLM provider detection, model defaults, and client construction.

Previously three scripts carried near-identical copies of the provider/key
resolver and two carried the chat-model factory; they all import from here now.
"""

from __future__ import annotations

import os
from typing import Optional

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

# Per-provider default chat models. A provider auto-detected from the
# environment must never be paired with another provider's model name.
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-5",
}


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is requested without an API key."""


class LLMInvocationError(RuntimeError):
    """Raised when the underlying LLM client fails to produce a response."""


class UnsupportedProviderError(RuntimeError):
    """Raised when an unknown LLM provider identifier is supplied."""


class MissingProviderDependencyError(RuntimeError):
    """Raised when a provider-specific dependency is unavailable."""


def normalize_provider(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "anthropic":
        return "claude"
    return provider


def auto_detect_provider() -> tuple[str, str] | None:
    """Auto-detect provider based on which API key is set.

    Returns:
        (provider_name, api_key) tuple or None if no key found.
    """
    for provider, env_var in PROVIDER_ENV_VARS.items():
        if os.environ.get(env_var):
            return (provider, os.environ[env_var])
    return None


def resolve_provider_and_key(
    explicit_key: str | None, provider_hint: str | None
) -> tuple[str, str | None]:
    """Resolve the provider and API key from explicit args or environment.

    Args:
        explicit_key: Explicit --api-key override
        provider_hint: Explicit --provider override

    Returns:
        (provider, api_key) tuple; api_key is None when nothing was found.
    """
    provider_hint = normalize_provider(provider_hint) if provider_hint else None

    if explicit_key:
        if provider_hint:
            return (provider_hint, explicit_key)
        detected = auto_detect_provider()
        return (detected[0], explicit_key) if detected else ("openai", explicit_key)

    if provider_hint:
        env_var = PROVIDER_ENV_VARS.get(provider_hint)
        key = os.environ.get(env_var) if env_var else None
        if key:
            return (provider_hint, key)
        # Fall back to any available key so an explicit hint still works with
        # e.g. an OpenAI-compatible proxy key.
        detected = auto_detect_provider()
        return (provider_hint, detected[1] if detected else None)

    detected = auto_detect_provider()
    if detected:
        return detected
    return ("openai", None)


def resolve_model(provider: str, model_arg: str | None) -> str:
    """Return the explicit model when given, else the provider's default."""
    if model_arg:
        return model_arg
    provider = normalize_provider(provider)
    try:
        return DEFAULT_MODELS[provider]
    except KeyError:
        raise UnsupportedProviderError(
            f"Unsupported provider '{provider}'. Supported: {', '.join(DEFAULT_MODELS)} (anthropic is an alias for claude)."
        ) from None


def load_chat_model(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float | None = None,
    max_tokens: int = 2000,
    base_url: Optional[str] = None,
):
    """Instantiate the LangChain chat model for the given provider.

    ``temperature=None`` omits the parameter entirely — some models (e.g. the
    gpt-5 family) reject any non-default temperature.
    """
    provider = normalize_provider(provider)
    common_kwargs: dict = {"model": model_name, "max_tokens": max_tokens}
    if temperature is not None:
        common_kwargs["temperature"] = temperature

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-openai'. Install it with `pip install langchain-openai`."
            ) from exc
        init_kwargs = {**common_kwargs, "openai_api_key": api_key}
        if base_url:
            init_kwargs["openai_api_base"] = base_url
        return ChatOpenAI(**init_kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-google-genai'. Install it with `pip install langchain-google-genai`."
            ) from exc
        # Gemini's canonical token-cap field is max_output_tokens (the
        # max_tokens alias is not supported on every package version).
        gemini_kwargs = dict(common_kwargs)
        gemini_kwargs["max_output_tokens"] = gemini_kwargs.pop("max_tokens")
        return ChatGoogleGenerativeAI(**gemini_kwargs, google_api_key=api_key)

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingProviderDependencyError(
                "Missing optional dependency 'langchain-anthropic'. Install it with `pip install langchain-anthropic`."
            ) from exc
        return ChatAnthropic(**common_kwargs, anthropic_api_key=api_key)

    raise UnsupportedProviderError(
        f"Unsupported provider '{provider}'. Supported: openai, gemini, claude (anthropic)."
    )
