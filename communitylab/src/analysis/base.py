"""Shared plumbing for the analysis modules.

Each analysis module (sentiment, themes, relevance) is a thin
:class:`StructuredAnalyzer`: a prompt built from a system prompt plus the
message, piped into a Groq model constrained to a Pydantic schema. Keeping the
plumbing here means every analyzer gets the same input formatting, fallback
behaviour and batch API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from src.ingest.validator import CommunityMessage
from src.prompts.system_prompts import build_message_prompt
from src.utils.llm_clients import LLMSettings, get_structured_llm

logger = logging.getLogger(__name__)

type MessageInput = CommunityMessage | str
"""Analyzers accept a validated message or a bare string of text."""


class AnalysisError(RuntimeError):
    """Raised when an analyzer cannot produce a result for a message."""

    def __init__(self, analyzer: str, message_id: str | None, cause: BaseException) -> None:
        self.analyzer = analyzer
        self.message_id = message_id
        self.cause = cause
        super().__init__(f"{analyzer} failed for message {message_id or '<text>'}: {cause}")


def message_to_prompt_input(message: MessageInput) -> dict[str, Any]:
    """Flatten a message into the variables expected by ``MESSAGE_HUMAN_TEMPLATE``."""
    if isinstance(message, str):
        return {
            "source": "unknown",
            "channel": "unknown",
            "author": "unknown",
            "timestamp": "unknown",
            "reactions": 0,
            "is_reply": "no",
            "category": "none",
            "content": message,
        }
    return {
        "source": message.source,
        "channel": message.channel,
        "author": message.author,
        "timestamp": message.timestamp.isoformat(),
        "reactions": message.reactions,
        "is_reply": "yes" if message.is_reply else "no",
        "category": message.category or "none",
        "content": message.content,
    }


def message_id_of(message: MessageInput) -> str | None:
    """Return the id of a message, or ``None`` for bare text."""
    return message.message_id if isinstance(message, CommunityMessage) else None


class StructuredAnalyzer[ResultT: BaseModel]:
    """Prompt + structured Groq model returning ``schema`` instances.

    Args:
        name: Short identifier used in logs and errors (e.g. ``"sentiment"``).
        schema: Pydantic model the LLM output must satisfy.
        system_prompt: Static system prompt from :mod:`src.prompts.system_prompts`.
        llm: Optional pre-built structured runnable (must return ``schema``
            instances). When omitted, :func:`get_structured_llm` builds the
            default Groq chain with fallbacks.
        settings: LLM settings used when ``llm`` is not provided.
    """

    def __init__(
        self,
        name: str,
        schema: type[ResultT],
        system_prompt: str,
        *,
        llm: Runnable | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.prompt = build_message_prompt(system_prompt)
        self._llm: Runnable = llm if llm is not None else get_structured_llm(schema, settings)
        self.chain: Runnable = self.prompt | self._llm

    def _coerce(self, output: Any, message: MessageInput) -> ResultT:
        """Make sure the LLM output is an instance of ``schema``."""
        if isinstance(output, self.schema):
            return output
        try:
            return self.schema.model_validate(output)
        except Exception as exc:  # noqa: BLE001 - re-raised as AnalysisError
            raise AnalysisError(self.name, message_id_of(message), exc) from exc

    def analyze(self, message: MessageInput, *, config: RunnableConfig | None = None) -> ResultT:
        """Analyse a single message.

        Raises:
            AnalysisError: If the model call fails on every fallback or the
                output does not match ``schema``.
        """
        try:
            output = self.chain.invoke(message_to_prompt_input(message), config=config)
        except AnalysisError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as AnalysisError
            raise AnalysisError(self.name, message_id_of(message), exc) from exc
        return self._coerce(output, message)

    def analyze_many(
        self,
        messages: Iterable[MessageInput],
        *,
        max_concurrency: int | None = 4,
        config: RunnableConfig | None = None,
    ) -> list[ResultT | AnalysisError]:
        """Analyse several messages concurrently.

        Failures do not abort the batch: the corresponding slot holds an
        :class:`AnalysisError` so callers can decide how to handle it.
        """
        items: Sequence[MessageInput] = list(messages)
        if not items:
            return []
        run_config: RunnableConfig = dict(config or {})
        if max_concurrency is not None:
            run_config["max_concurrency"] = max_concurrency
        raw_outputs = self.chain.batch(
            [message_to_prompt_input(m) for m in items], config=run_config, return_exceptions=True
        )
        results: list[ResultT | AnalysisError] = []
        for message, output in zip(items, raw_outputs, strict=True):
            if isinstance(output, BaseException):
                results.append(AnalysisError(self.name, message_id_of(message), output))
                continue
            try:
                results.append(self._coerce(output, message))
            except AnalysisError as exc:
                results.append(exc)
        failures = sum(isinstance(r, AnalysisError) for r in results)
        if failures:
            logger.warning("%s: %d of %d message(s) failed", self.name, failures, len(results))
        return results
