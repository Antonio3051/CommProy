"""Sentiment analysis: label + score for a community message."""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.analysis.base import MessageInput, StructuredAnalyzer
from src.prompts.system_prompts import SENTIMENT_SYSTEM_PROMPT
from src.utils.llm_clients import LLMSettings

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]


class SentimentResult(BaseModel):
    """Structured sentiment output for one message.

    Attributes:
        label: Discrete sentiment class.
        score: Polarity from ``-1.0`` (very negative) to ``1.0`` (very positive).
        confidence: Model certainty in ``[0, 1]``.
        emotions: Up to three lowercase emotion words.
        rationale: One-sentence evidence for the verdict.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    label: SentimentLabel = Field(description="One of positive, negative, neutral or mixed.")
    score: float = Field(ge=-1.0, le=1.0, description="-1.0 very negative … 1.0 very positive.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Certainty of the assessment.")
    emotions: list[str] = Field(default_factory=list, max_length=3, description="Detected emotions, lowercase.")
    rationale: str = Field(default="", max_length=300, description="Short justification.")

    @field_validator("emotions", mode="before")
    @classmethod
    def _normalize_emotions(cls, value: object) -> object:
        if isinstance(value, list):
            seen: list[str] = []
            for item in value:
                word = str(item).strip().lower()
                if word and word not in seen:
                    seen.append(word)
            return seen[:3]
        return value

    @model_validator(mode="after")
    def _score_matches_label(self) -> SentimentResult:
        """Reject outputs where the polarity sign contradicts the label."""
        if self.label == "positive" and self.score < 0:
            raise ValueError("positive label requires a non-negative score")
        if self.label == "negative" and self.score > 0:
            raise ValueError("negative label requires a non-positive score")
        return self

    @property
    def is_negative(self) -> bool:
        """Convenience flag for routing/escalation."""
        return self.label == "negative"


class SentimentAnalyzer(StructuredAnalyzer[SentimentResult]):
    """Extract :class:`SentimentResult` from community messages."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("sentiment", SentimentResult, SENTIMENT_SYSTEM_PROMPT, llm=llm, settings=settings)


def analyze_sentiment(
    message: MessageInput, *, llm: Runnable | None = None, settings: LLMSettings | None = None
) -> SentimentResult:
    """One-shot helper: build a :class:`SentimentAnalyzer` and analyse ``message``."""
    return SentimentAnalyzer(llm=llm, settings=settings).analyze(message)
