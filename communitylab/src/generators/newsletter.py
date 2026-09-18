"""Weekly community-highlights newsletter section generator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.generators.base import ContentGenerator, GeneratedAsset
from src.utils.llm_clients import LLMSettings

# The prompt asks for 3-5 highlights; feeding many more messages than that
# only adds noise and tokens, so the selector caps the brief.
DEFAULT_MAX_MESSAGES = 12

# Ordering mirrors the prompt's priorities: problems first, wins second, resources third.
_CATEGORY_PRIORITY: dict[str, int] = {
    "complaint": 0,
    "technical_question": 0,
    "feature_request": 1,
    "success_story": 2,
    "announcement": 3,
    "resource_share": 3,
    "technical_answer": 4,
    "feedback": 4,
}


class NewsletterSection(BaseModel):
    """Structured output of the newsletter prompt."""

    model_config = ConfigDict(frozen=True)

    title: str
    intro: str
    highlights: list[str] = Field(default_factory=list, min_length=1)
    closing: str = ""

    def render(self) -> str:
        """Markdown section."""
        bullets = "\n".join(f"- {h}" for h in self.highlights)
        parts = [f"## {self.title.strip()}", "", self.intro.strip(), "", bullets]
        if self.closing.strip():
            parts += ["", self.closing.strip()]
        return "\n".join(parts)


def select_highlights(
    items: Iterable[EnrichedMessage], *, max_messages: int = DEFAULT_MAX_MESSAGES
) -> list[EnrichedMessage]:
    """Pick the messages worth mentioning: relevance ``medium`` or better, noise dropped.

    Sorted by the prompt's editorial priority, then relevance.
    """
    candidates = [i for i in items if i.relevance is not None and i.relevance.tier in ("medium", "high", "critical")]

    def key(item: EnrichedMessage) -> tuple[int, float]:
        category = item.themes.primary_category if item.themes is not None else "other"
        score = item.relevance.score if item.relevance is not None else 0.0
        return (_CATEGORY_PRIORITY.get(category, 5), -score)

    return sorted(candidates, key=key)[:max_messages]


def default_period_label(items: Sequence[EnrichedMessage]) -> str:
    """``"YYYY-MM-DD to YYYY-MM-DD"`` covering the messages' timestamps."""
    if not items:
        return date.today().isoformat()
    stamps = sorted(i.message.timestamp.date() for i in items)
    return stamps[0].isoformat() if stamps[0] == stamps[-1] else f"{stamps[0].isoformat()} to {stamps[-1].isoformat()}"


class NewsletterGenerator(ContentGenerator[NewsletterSection]):
    """Summarises a period's highlights into one newsletter section."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("newsletter", NewsletterSection, llm=llm, settings=settings)

    def generate(
        self,
        items: Sequence[EnrichedMessage],
        *,
        period_label: str | None = None,
        community_name: str = "the community",
        language: str = "en",
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ) -> GeneratedAsset:
        """Draft the section from the most relevant messages of the period.

        Raises:
            ValueError: If no message passes the relevance filter.
        """
        selected = select_highlights(items, max_messages=max_messages)
        if not selected:
            raise ValueError("No messages with medium or higher relevance to build a newsletter section")
        period = period_label or default_period_label(selected)
        context = (
            f"Weekly highlights of {community_name} for the period {period}. "
            f"{len(selected)} messages selected out of {len(items)} analysed."
        )
        section = self.invoke(selected, context=context, language=language)
        return GeneratedAsset(
            channel="newsletter",
            title=section.title.strip() or f"Community highlights {period}",
            content=section,
            markdown=section.render(),
            language=language,
            source_message_ids=tuple(i.message.message_id for i in selected),
            metadata={"period": period, "community": community_name, "messages_analysed": len(items)},
        )
