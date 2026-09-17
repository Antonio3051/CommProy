"""Topic extraction: categorise a message and pull out its themes."""

from __future__ import annotations

from typing import Literal, get_args

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.analysis.base import MessageInput, StructuredAnalyzer
from src.prompts.system_prompts import TOPIC_EXTRACTION_SYSTEM_PROMPT
from src.utils.llm_clients import LLMSettings

TopicCategory = Literal[
    "technical_question",
    "technical_answer",
    "success_story",
    "complaint",
    "feature_request",
    "announcement",
    "resource_share",
    "feedback",
    "community_chatter",
    "other",
]
"""Controlled taxonomy used for routing and reporting."""

TOPIC_CATEGORIES: tuple[str, ...] = get_args(TopicCategory)


def _dedupe(values: object, *, limit: int, lowercase: bool) -> object:
    if not isinstance(values, list):
        return values
    seen: list[str] = []
    for item in values:
        text = str(item).strip()
        if lowercase:
            text = text.lower()
        if text and text.lower() not in {s.lower() for s in seen}:
            seen.append(text)
    return seen[:limit]


class ThemesResult(BaseModel):
    """Structured topic output for one message.

    Attributes:
        primary_category: Main category from :data:`TOPIC_CATEGORIES`.
        topics: Short noun phrases describing what the message is about.
        technologies: Canonical tool/library/vendor names mentioned.
        keywords: Lowercase search terms.
        summary: Neutral one-sentence summary.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    primary_category: TopicCategory = Field(description="Main category of the message.")
    topics: list[str] = Field(default_factory=list, max_length=5, description="Topic phrases, most important first.")
    technologies: list[str] = Field(default_factory=list, max_length=10, description="Technologies mentioned.")
    keywords: list[str] = Field(default_factory=list, max_length=8, description="Lowercase search keywords.")
    summary: str = Field(default="", max_length=300, description="One-sentence summary.")

    @field_validator("primary_category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        return value.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(value, str) else value

    @field_validator("topics", mode="before")
    @classmethod
    def _clean_topics(cls, value: object) -> object:
        return _dedupe(value, limit=5, lowercase=False)

    @field_validator("technologies", mode="before")
    @classmethod
    def _clean_technologies(cls, value: object) -> object:
        return _dedupe(value, limit=10, lowercase=False)

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, value: object) -> object:
        return _dedupe(value, limit=8, lowercase=True)

    @property
    def is_question(self) -> bool:
        """Whether the message asks for help."""
        return self.primary_category == "technical_question"


class ThemesAnalyzer(StructuredAnalyzer[ThemesResult]):
    """Extract :class:`ThemesResult` from community messages."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("themes", ThemesResult, TOPIC_EXTRACTION_SYSTEM_PROMPT, llm=llm, settings=settings)


def analyze_themes(
    message: MessageInput, *, llm: Runnable | None = None, settings: LLMSettings | None = None
) -> ThemesResult:
    """One-shot helper: build a :class:`ThemesAnalyzer` and analyse ``message``."""
    return ThemesAnalyzer(llm=llm, settings=settings).analyze(message)
