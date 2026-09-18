"""Tests for ``src.utils.llm_clients`` (offline: no Groq calls)."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr
from src.utils.llm_clients import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_PRIMARY_MODEL,
    GROQ_API_KEY_ENV,
    LLMSettings,
    MissingAPIKeyError,
    build_chat_model,
    get_chat_models,
    get_llm,
    get_structured_llm,
)

from tests.conftest import FakeToolChatModel


class Answer(BaseModel):
    value: int


def make_factory(
    *, failing: set[str] = frozenset(), payload: dict | None = None
) -> tuple[dict[str, FakeToolChatModel], object]:
    """Return ``(registry, factory)``; ``registry`` lets tests inspect each fake model."""
    built: dict[str, FakeToolChatModel] = {}

    def factory(model: str, settings: LLMSettings) -> BaseChatModel:  # noqa: ARG001
        fake = FakeToolChatModel(payload=payload or {"value": 42}, name=model, fail=model in failing)
        built[model] = fake
        return fake

    return built, factory


class TestLLMSettings:
    def test_defaults(self):
        settings = LLMSettings()
        assert settings.primary_model == DEFAULT_PRIMARY_MODEL
        assert settings.fallback_models == DEFAULT_FALLBACK_MODELS
        assert settings.models == (DEFAULT_PRIMARY_MODEL, *DEFAULT_FALLBACK_MODELS)
        assert settings.temperature == 0.0

    def test_models_deduplicated(self):
        settings = LLMSettings(primary_model="a", fallback_models=("b", "a", "b"))
        assert settings.models == ("a", "b")

    def test_resolve_api_key_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv(GROQ_API_KEY_ENV, "env-key")
        assert LLMSettings(api_key=SecretStr("explicit")).resolve_api_key().get_secret_value() == "explicit"

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv(GROQ_API_KEY_ENV, "  env-key  ")
        assert LLMSettings().resolve_api_key().get_secret_value() == "env-key"

    def test_resolve_api_key_missing(self, monkeypatch):
        monkeypatch.delenv(GROQ_API_KEY_ENV, raising=False)
        with pytest.raises(MissingAPIKeyError, match=GROQ_API_KEY_ENV):
            LLMSettings().resolve_api_key()

    def test_settings_are_validated(self):
        with pytest.raises(ValueError):
            LLMSettings(temperature=3.0)
        with pytest.raises(ValueError):
            LLMSettings(primary_model="")


class TestBuildChatModel:
    def test_builds_chat_groq(self, monkeypatch):
        monkeypatch.setenv(GROQ_API_KEY_ENV, "gsk_test")
        settings = LLMSettings(temperature=0.3, max_tokens=256, timeout_seconds=12, max_retries=1)
        model = build_chat_model("llama-3.1-8b-instant", settings)
        assert isinstance(model, ChatGroq)
        assert model.model_name == "llama-3.1-8b-instant"
        assert model.temperature == 0.3
        assert model.max_tokens == 256
        assert model.max_retries == 1
        assert model.groq_api_key.get_secret_value() == "gsk_test"

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv(GROQ_API_KEY_ENV, raising=False)
        with pytest.raises(MissingAPIKeyError):
            build_chat_model("llama-3.1-8b-instant", LLMSettings())


class TestGetLLM:
    def test_get_chat_models_uses_factory_in_order(self):
        built, factory = make_factory()
        models = get_chat_models(LLMSettings(primary_model="p", fallback_models=("f1", "f2")), model_factory=factory)
        assert [m.name for m in models] == ["p", "f1", "f2"]
        assert set(built) == {"p", "f1", "f2"}

    def test_single_model_has_no_fallback_wrapper(self):
        _, factory = make_factory()
        llm = get_llm(LLMSettings(primary_model="only", fallback_models=()), model_factory=factory)
        assert isinstance(llm, FakeToolChatModel)

    def test_wraps_with_fallbacks(self):
        _, factory = make_factory()
        llm = get_llm(LLMSettings(), model_factory=factory)
        assert isinstance(llm, RunnableWithFallbacks)
        assert isinstance(llm.invoke("hello"), AIMessage)

    def test_primary_failure_falls_back(self):
        built, factory = make_factory(failing={"p"})
        llm = get_llm(LLMSettings(primary_model="p", fallback_models=("f",)), model_factory=factory)
        assert isinstance(llm.invoke("hello"), AIMessage)
        assert built["p"].calls == 1
        assert built["f"].calls == 1

    def test_all_models_failing_raises(self):
        built, factory = make_factory(failing={"p", "f"})
        llm = get_llm(LLMSettings(primary_model="p", fallback_models=("f",)), model_factory=factory)
        with pytest.raises(RuntimeError, match="is down"):
            llm.invoke("hello")
        assert built["p"].calls == 1 and built["f"].calls == 1


class TestGetStructuredLLM:
    def test_returns_schema_instances(self):
        _, factory = make_factory(payload={"value": 7})
        llm = get_structured_llm(Answer, LLMSettings(), model_factory=factory)
        result = llm.invoke("count")
        assert isinstance(result, Answer)
        assert result.value == 7

    def test_structured_fallback_on_primary_error(self):
        built, factory = make_factory(failing={"p"}, payload={"value": 1})
        llm = get_structured_llm(Answer, LLMSettings(primary_model="p", fallback_models=("f",)), model_factory=factory)
        assert llm.invoke("count") == Answer(value=1)
        assert built["p"].calls == 1 and built["f"].calls == 1

    def test_structured_fallback_on_invalid_output(self):
        """A primary that answers with the wrong shape must not surface a broken result."""
        built: dict[str, FakeToolChatModel] = {}

        def factory(model: str, settings: LLMSettings) -> BaseChatModel:  # noqa: ARG001
            payload = {"value": "not-a-number"} if model == "p" else {"value": 3}
            built[model] = FakeToolChatModel(payload=payload, name=model)
            return built[model]

        llm = get_structured_llm(Answer, LLMSettings(primary_model="p", fallback_models=("f",)), model_factory=factory)
        assert llm.invoke("count") == Answer(value=3)
        assert built["f"].calls == 1

    def test_batch_returns_schema_instances(self):
        _, factory = make_factory(payload={"value": 2})
        llm = get_structured_llm(Answer, LLMSettings(), model_factory=factory)
        assert llm.batch(["a", "b"]) == [Answer(value=2), Answer(value=2)]
