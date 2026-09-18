"""Groq LLM clients for CommunityLab.

Central place to build chat models so every layer (analysis, generators,
decisions) shares the same credentials, model names, retry policy and
fallback chain. Groq is used for its low-latency inference; the default
chain tries ``llama-3.3-70b-versatile`` first and falls back to the much
cheaper ``llama-3.1-8b-instant`` when the primary model errors out (rate
limit, timeout, model outage).

Typical use::

    llm = get_llm()                                  # chat model with fallback
    extractor = get_structured_llm(MySchema)         # returns MySchema instances

Set ``GROQ_API_KEY`` in the environment (or pass ``LLMSettings(api_key=...)``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)

GROQ_API_KEY_ENV = "GROQ_API_KEY"

DEFAULT_PRIMARY_MODEL = "llama-3.3-70b-versatile"
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = ("llama-3.1-8b-instant",)

StructuredOutputMethod = Literal["function_calling", "json_mode", "json_schema"]

ModelFactory = Callable[[str, "LLMSettings"], BaseChatModel]
"""Builds a chat model for a given model name; swappable for tests."""


class MissingAPIKeyError(RuntimeError):
    """Raised when no Groq API key is configured."""


class LLMSettings(BaseModel):
    """Connection and generation settings shared by all Groq clients.

    Attributes:
        api_key: Groq API key. Falls back to the ``GROQ_API_KEY`` env var.
        primary_model: Model tried first.
        fallback_models: Models tried in order when the previous one fails.
        temperature: Sampling temperature; ``0`` for deterministic extraction.
        max_tokens: Optional completion cap.
        timeout_seconds: Per-request timeout.
        max_retries: Retries per model before moving to the next fallback.
        structured_output_method: How structured outputs are requested from
            Groq (``function_calling`` works across all Llama models).
    """

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    primary_model: str = Field(default=DEFAULT_PRIMARY_MODEL, min_length=1)
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    structured_output_method: StructuredOutputMethod = "function_calling"

    @property
    def models(self) -> tuple[str, ...]:
        """Primary model followed by its fallbacks, duplicates removed."""
        ordered: list[str] = []
        for name in (self.primary_model, *self.fallback_models):
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)

    def resolve_api_key(self) -> SecretStr:
        """Return the configured key or read it from the environment.

        Raises:
            MissingAPIKeyError: If neither the setting nor the env var is set.
        """
        if self.api_key is not None and self.api_key.get_secret_value():
            return self.api_key
        env_value = os.environ.get(GROQ_API_KEY_ENV, "").strip()
        if not env_value:
            raise MissingAPIKeyError(
                f"No Groq API key found. Set the {GROQ_API_KEY_ENV} environment variable "
                "or pass LLMSettings(api_key=...)."
            )
        return SecretStr(env_value)


def build_chat_model(model: str, settings: LLMSettings) -> BaseChatModel:
    """Instantiate a :class:`ChatGroq` for ``model`` using ``settings``."""
    return ChatGroq(
        model=model,
        api_key=settings.resolve_api_key(),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _with_fallbacks(runnables: Sequence[Runnable], *, label: str) -> Runnable:
    """Chain ``runnables`` so each one is tried in turn until one succeeds."""
    if not runnables:
        raise ValueError("At least one model is required")
    primary, *fallbacks = runnables
    if not fallbacks:
        return primary
    logger.debug("Configured %s with %d fallback(s)", label, len(fallbacks))
    return primary.with_fallbacks(fallbacks)


def get_chat_models(
    settings: LLMSettings | None = None, *, model_factory: ModelFactory = build_chat_model
) -> list[BaseChatModel]:
    """Build one chat model per entry in ``settings.models`` (primary first)."""
    settings = settings or LLMSettings()
    return [model_factory(name, settings) for name in settings.models]


def get_llm(settings: LLMSettings | None = None, *, model_factory: ModelFactory = build_chat_model) -> Runnable:
    """Return the base chat model with fallback logic applied.

    The result accepts the usual LangChain inputs (a string, a list of
    messages or a prompt value) and returns an ``AIMessage``. When the
    primary model raises, the next model in ``settings.fallback_models`` is
    tried transparently.

    Args:
        settings: Connection/generation settings; defaults to :class:`LLMSettings`.
        model_factory: Function that builds a chat model from a model name.

    Returns:
        A runnable chat model (``ChatGroq`` or ``RunnableWithFallbacks``).
    """
    settings = settings or LLMSettings()
    return _with_fallbacks(get_chat_models(settings, model_factory=model_factory), label=settings.primary_model)


def get_structured_llm[SchemaT: BaseModel](
    schema: type[SchemaT],
    settings: LLMSettings | None = None,
    *,
    model_factory: ModelFactory = build_chat_model,
) -> Runnable:
    """Return a chat model with fallbacks that emits ``schema`` instances.

    Structured output is bound to every model *before* fallbacks are wired,
    so a parsing/validation failure on the primary model also triggers the
    fallback instead of surfacing a half-parsed result.

    Args:
        schema: Pydantic model describing the expected output.
        settings: Connection/generation settings; defaults to :class:`LLMSettings`.
        model_factory: Function that builds a chat model from a model name.

    Returns:
        A runnable whose ``invoke``/``batch`` return ``schema`` instances.
    """
    settings = settings or LLMSettings()
    structured = [
        model.with_structured_output(schema, method=settings.structured_output_method)
        for model in get_chat_models(settings, model_factory=model_factory)
    ]
    return _with_fallbacks(structured, label=f"{settings.primary_model}->{schema.__name__}")
