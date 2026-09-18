"""Consolidation: merge sentiment, themes and relevance into one enriched record.

Architecture: ``CommunityMessage -> {Sentiment, Themes, Relevance} -> EnrichedMessage``.
The three analyzers run independently; a failure in one does not discard the
others' output unless ``strict=True`` is requested.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.base import AnalysisError, StructuredAnalyzer
from src.analysis.relevance import RelevanceAnalyzer, RelevanceResult
from src.analysis.sentiment import SentimentAnalyzer, SentimentResult
from src.analysis.themes import ThemesAnalyzer, ThemesResult
from src.ingest.validator import CommunityMessage
from src.utils.llm_clients import LLMSettings

logger = logging.getLogger(__name__)


class EnrichedMessage(BaseModel):
    """A validated message plus the outputs of the analysis layer.

    Attributes:
        message: The original validated message.
        sentiment: Sentiment output, ``None`` if that analyzer failed.
        themes: Topic output, ``None`` if that analyzer failed.
        relevance: Relevance output, ``None`` if that analyzer failed.
        errors: ``{analyzer_name: error_message}`` for failed analyzers.
        analyzed_at: UTC timestamp of the consolidation.
    """

    model_config = ConfigDict(frozen=True)

    message: CommunityMessage
    sentiment: SentimentResult | None = None
    themes: ThemesResult | None = None
    relevance: RelevanceResult | None = None
    errors: dict[str, str] = Field(default_factory=dict)
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def message_id(self) -> str:
        """Shortcut to the underlying message id."""
        return self.message.message_id

    @property
    def is_complete(self) -> bool:
        """All three analyzers produced output."""
        return not self.errors and None not in (self.sentiment, self.themes, self.relevance)

    def to_record(self) -> dict[str, Any]:
        """Flatten to a single dict, suitable for a DataFrame row or JSON.

        Analyzer fields are prefixed (``sentiment_label``, ``themes_topics``,
        ``relevance_score`` …) so they never collide with message columns.
        """
        record: dict[str, Any] = self.message.model_dump(mode="json")
        for prefix, result in (
            ("sentiment", self.sentiment),
            ("themes", self.themes),
            ("relevance", self.relevance),
        ):
            if result is None:
                continue
            for key, value in result.model_dump(mode="json").items():
                record[f"{prefix}_{key}"] = value
        record["analysis_errors"] = dict(self.errors)
        record["analyzed_at"] = self.analyzed_at.isoformat()
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EnrichedMessage:
        """Inverse of :meth:`to_record`; also accepts a plain ``model_dump`` (with a ``message`` key)."""
        if "message" in record:
            return cls.model_validate(record)

        def sub(prefix: str) -> dict[str, Any]:
            return {k[len(prefix) :]: v for k, v in record.items() if k.startswith(prefix) and v is not None}

        message_fields = {k: record[k] for k in CommunityMessage.model_fields if k in record}
        sentiment, themes, relevance = sub("sentiment_"), sub("themes_"), sub("relevance_")
        analyzed_at = record.get("analyzed_at")
        return cls(
            message=CommunityMessage.model_validate(message_fields),
            sentiment=SentimentResult.model_validate(sentiment) if sentiment else None,
            themes=ThemesResult.model_validate(themes) if themes else None,
            relevance=RelevanceResult.model_validate(relevance) if relevance else None,
            errors=dict(record.get("analysis_errors") or record.get("errors") or {}),
            analyzed_at=datetime.fromisoformat(analyzed_at) if isinstance(analyzed_at, str) else datetime.now(UTC),
        )


class Consolidator:
    """Run the three analyzers over messages and merge their outputs.

    Args:
        sentiment: Custom sentiment analyzer (defaults to the Groq-backed one).
        themes: Custom themes analyzer.
        relevance: Custom relevance analyzer.
        settings: LLM settings for the default analyzers.
        strict: Raise :class:`AnalysisError` instead of recording partial results.
    """

    def __init__(
        self,
        *,
        sentiment: StructuredAnalyzer[SentimentResult] | None = None,
        themes: StructuredAnalyzer[ThemesResult] | None = None,
        relevance: StructuredAnalyzer[RelevanceResult] | None = None,
        settings: LLMSettings | None = None,
        strict: bool = False,
    ) -> None:
        self.sentiment = sentiment if sentiment is not None else SentimentAnalyzer(settings=settings)
        self.themes = themes if themes is not None else ThemesAnalyzer(settings=settings)
        self.relevance = relevance if relevance is not None else RelevanceAnalyzer(settings=settings)
        self.strict = strict

    @classmethod
    def from_llms(
        cls,
        *,
        sentiment_llm: Runnable,
        themes_llm: Runnable,
        relevance_llm: Runnable,
        strict: bool = False,
    ) -> Consolidator:
        """Build a consolidator from pre-built structured runnables (handy for tests)."""
        return cls(
            sentiment=SentimentAnalyzer(llm=sentiment_llm),
            themes=ThemesAnalyzer(llm=themes_llm),
            relevance=RelevanceAnalyzer(llm=relevance_llm),
            strict=strict,
        )

    @property
    def analyzers(self) -> tuple[StructuredAnalyzer[Any], ...]:
        """The analyzers in execution order."""
        return (self.sentiment, self.themes, self.relevance)

    def _merge(
        self,
        message: CommunityMessage,
        outputs: Sequence[BaseModel | AnalysisError],
    ) -> EnrichedMessage:
        results: dict[str, BaseModel | None] = {}
        errors: dict[str, str] = {}
        for analyzer, output in zip(self.analyzers, outputs, strict=True):
            if isinstance(output, AnalysisError):
                if self.strict:
                    raise output
                logger.warning("%s", output)
                results[analyzer.name] = None
                errors[analyzer.name] = str(output.cause)
            else:
                results[analyzer.name] = output
        return EnrichedMessage(
            message=message,
            sentiment=results.get("sentiment"),
            themes=results.get("themes"),
            relevance=results.get("relevance"),
            errors=errors,
        )

    def enrich(self, message: CommunityMessage) -> EnrichedMessage:
        """Analyse one message with every analyzer and merge the outputs."""
        outputs: list[BaseModel | AnalysisError] = []
        for analyzer in self.analyzers:
            try:
                outputs.append(analyzer.analyze(message))
            except AnalysisError as exc:
                outputs.append(exc)
        return self._merge(message, outputs)

    def enrich_many(
        self,
        messages: Iterable[CommunityMessage],
        *,
        max_concurrency: int | None = 4,
    ) -> list[EnrichedMessage]:
        """Analyse a batch of messages, running each analyzer as a batched call."""
        items = list(messages)
        if not items:
            return []
        per_analyzer = [a.analyze_many(items, max_concurrency=max_concurrency) for a in self.analyzers]
        return [self._merge(message, [column[i] for column in per_analyzer]) for i, message in enumerate(items)]


def to_records(enriched: Iterable[EnrichedMessage]) -> list[dict[str, Any]]:
    """Flatten enriched messages into plain dicts."""
    return [item.to_record() for item in enriched]


def to_frame(enriched: Iterable[EnrichedMessage]) -> pd.DataFrame:
    """Flatten enriched messages into a DataFrame (one row per message)."""
    return pd.DataFrame.from_records(to_records(enriched))
