"""Relevance scoring: how much attention an interaction deserves."""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.analysis.base import MessageInput, StructuredAnalyzer
from src.prompts.system_prompts import CHANNELS, RELEVANCE_SYSTEM_PROMPT, Channel
from src.utils.llm_clients import LLMSettings

RelevanceTier = Literal["noise", "low", "medium", "high", "critical"]

_TIER_THRESHOLDS: tuple[tuple[float, RelevanceTier], ...] = (
    (0.85, "critical"),
    (0.6, "high"),
    (0.4, "medium"),
    (0.2, "low"),
    (0.0, "noise"),
)


def tier_for_score(score: float) -> RelevanceTier:
    """Map a ``[0, 1]`` relevance score onto its :data:`RelevanceTier`."""
    for threshold, tier in _TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "noise"


class RelevanceResult(BaseModel):
    """Structured relevance output for one message.

    Attributes:
        score: Importance from ``0.0`` (noise) to ``1.0`` (critical).
        tier: Discrete bucket derived from ``score``; recomputed if the model
            returns an inconsistent value.
        is_actionable: The team should do something about this message.
        requires_response: Nobody has answered/acknowledged it yet.
        suggested_channels: Where the message could be reused as content.
        reasons: Short justifications for the score.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    score: float = Field(ge=0.0, le=1.0, description="0.0 noise … 1.0 critical.")
    tier: RelevanceTier = Field(default="noise", description="Bucket matching the score.")
    is_actionable: bool = Field(default=False, description="Team action needed.")
    requires_response: bool = Field(default=False, description="Awaiting an answer.")
    suggested_channels: list[Channel] = Field(default_factory=list, description="Content reuse channels.")
    reasons: list[str] = Field(default_factory=list, max_length=3, description="Why this score.")

    @field_validator("suggested_channels", mode="before")
    @classmethod
    def _clean_channels(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            name = str(item).strip().lower()
            if name in CHANNELS and name not in cleaned:
                cleaned.append(name)
        return cleaned

    @field_validator("reasons", mode="before")
    @classmethod
    def _clean_reasons(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(r).strip() for r in value if str(r).strip()][:3]
        return value

    @model_validator(mode="after")
    def _align_tier(self) -> RelevanceResult:
        """The score is the source of truth; the tier is derived from it."""
        expected = tier_for_score(self.score)
        if self.tier != expected:
            object.__setattr__(self, "tier", expected)
        return self

    @property
    def is_noise(self) -> bool:
        """Whether the message can be safely ignored."""
        return self.tier == "noise"


class RelevanceAnalyzer(StructuredAnalyzer[RelevanceResult]):
    """Extract :class:`RelevanceResult` from community messages."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("relevance", RelevanceResult, RELEVANCE_SYSTEM_PROMPT, llm=llm, settings=settings)


def analyze_relevance(
    message: MessageInput, *, llm: Runnable | None = None, settings: LLMSettings | None = None
) -> RelevanceResult:
    """One-shot helper: build a :class:`RelevanceAnalyzer` and analyse ``message``."""
    return RelevanceAnalyzer(llm=llm, settings=settings).analyze(message)
