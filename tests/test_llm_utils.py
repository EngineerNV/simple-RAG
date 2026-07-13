from __future__ import annotations

import pytest

from utils import llm as llm_utils


ALL_KEYS = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)


def test_auto_detect_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert llm_utils.auto_detect_provider() == ("claude", "sk-ant")

    monkeypatch.setenv("GOOGLE_API_KEY", "sk-goog")
    assert llm_utils.auto_detect_provider() == ("gemini", "sk-goog")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert llm_utils.auto_detect_provider() == ("openai", "sk-oai")


def test_auto_detect_none_without_keys() -> None:
    assert llm_utils.auto_detect_provider() is None


def test_resolve_explicit_key_and_hint() -> None:
    assert llm_utils.resolve_provider_and_key("sk-x", "claude") == ("claude", "sk-x")


def test_resolve_explicit_key_only_prefers_detected_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-goog")
    assert llm_utils.resolve_provider_and_key("sk-x", None) == ("gemini", "sk-x")


def test_resolve_explicit_key_only_defaults_to_openai() -> None:
    assert llm_utils.resolve_provider_and_key("sk-x", None) == ("openai", "sk-x")


def test_resolve_hint_uses_own_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit provider hint must pick up ITS provider's key even when a
    # higher-precedence provider key is also set.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert llm_utils.resolve_provider_and_key(None, "claude") == ("claude", "sk-ant")


def test_resolve_hint_normalizes_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert llm_utils.resolve_provider_and_key(None, "anthropic") == ("claude", "sk-ant")


def test_resolve_nothing_found() -> None:
    assert llm_utils.resolve_provider_and_key(None, None) == ("openai", None)


def test_resolve_model_explicit_wins() -> None:
    assert llm_utils.resolve_model("openai", "gpt-custom") == "gpt-custom"


@pytest.mark.parametrize("provider", ["openai", "gemini", "claude"])
def test_resolve_model_per_provider_default(provider: str) -> None:
    assert llm_utils.resolve_model(provider, None) == llm_utils.DEFAULT_MODELS[provider]


def test_resolve_model_normalizes_anthropic() -> None:
    assert llm_utils.resolve_model("anthropic", None) == llm_utils.DEFAULT_MODELS["claude"]


def test_resolve_model_unknown_provider_raises() -> None:
    with pytest.raises(llm_utils.UnsupportedProviderError):
        llm_utils.resolve_model("mystery", None)


def test_load_chat_model_unknown_provider_raises() -> None:
    with pytest.raises(llm_utils.UnsupportedProviderError):
        llm_utils.load_chat_model("mystery", "model-x", "sk-x")


def test_load_chat_model_omits_temperature_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    llm_utils.load_chat_model("openai", "gpt-test", "sk-x", temperature=None)
    assert "temperature" not in captured
    assert "max_completion_tokens" not in captured

    captured.clear()
    llm_utils.load_chat_model("openai", "gpt-test", "sk-x", temperature=0.4)
    assert captured["temperature"] == 0.4


def test_load_chat_model_gemini_uses_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    import types

    fake_module = types.ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    llm_utils.load_chat_model("gemini", "gemini-test", "sk-g", max_tokens=1234)
    assert captured["max_output_tokens"] == 1234
    assert "max_tokens" not in captured
