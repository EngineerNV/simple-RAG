from __future__ import annotations

import sys
import pytest
from utils.llm_provider import (
    MissingProviderDependencyError,
    UnsupportedProviderError,
    auto_detect_provider,
    build_chat_model,
    is_openai_reasoning_model,
    resolve_provider_and_key,
)


def test_auto_detect_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear environment variables
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert auto_detect_provider() is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    assert auto_detect_provider() == ("openai", "sk-openai-key")

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key")
    assert auto_detect_provider() == ("gemini", "goog-key")

    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
    assert auto_detect_provider() == ("claude", "sk-ant-key")


def test_resolve_provider_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Both explicit
    assert resolve_provider_and_key("custom-key", "gemini") == ("gemini", "custom-key")

    # Only explicit key, no env vars -> default to openai
    assert resolve_provider_and_key("custom-key", None) == ("openai", "custom-key")

    # Multiple env keys present: provider hint picks the specific key
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-goog-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-ant-key")
    assert resolve_provider_and_key(None, "gemini") == ("gemini", "env-goog-key")
    assert resolve_provider_and_key(None, "claude") == ("claude", "env-ant-key")
    assert resolve_provider_and_key(None, "openai") == ("openai", "env-openai-key")
    assert resolve_provider_and_key(None, None) == ("openai", "env-openai-key")

    # Provider hint when only another key exists (returns None for key to prevent key leakage)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert resolve_provider_and_key(None, "claude") == ("claude", None)

    # No keys present anywhere
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert resolve_provider_and_key(None, None) == ("openai", None)
    assert resolve_provider_and_key(None, "gemini") == ("gemini", None)


def test_is_openai_reasoning_model() -> None:
    assert is_openai_reasoning_model("o1-mini") is True
    assert is_openai_reasoning_model("o3-mini") is True
    assert is_openai_reasoning_model("o4") is True
    assert is_openai_reasoning_model("gpt-5") is True
    assert is_openai_reasoning_model("gpt-4o") is False
    assert is_openai_reasoning_model("gpt-3.5-turbo") is False


def test_unsupported_provider_raises_error() -> None:
    with pytest.raises(UnsupportedProviderError, match="Unsupported provider 'invalid_provider'"):
        build_chat_model("invalid_provider", "model", "key", 0.2, 100)


def test_missing_provider_dependency_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Block langchain_google_genai import by inserting None into sys.modules
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)

    with pytest.raises(
        MissingProviderDependencyError,
        match="Missing optional dependency 'langchain-google-genai'. Install it with `pip install langchain-google-genai`.",
    ):
        build_chat_model("gemini", "gemini-pro", "key", 0.2, 100)

    # Block langchain_anthropic import
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)

    with pytest.raises(
        MissingProviderDependencyError,
        match="Missing optional dependency 'langchain-anthropic'. Install it with `pip install langchain-anthropic`.",
    ):
        build_chat_model("claude", "claude-3-5-sonnet", "key", 0.2, 100)
