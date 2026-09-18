"""FAQ entry generator for recurring technical doubts (Decision Engine rule R2)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.analysis.consolidator import EnrichedMessage
from src.decisions.recurring_topics import RecurringTopic
from src.generators.base import ContentGenerator, GeneratedAsset, GenerationError
from src.utils.llm_clients import LLMSettings

logger = logging.getLogger(__name__)


class FAQEntry(BaseModel):
    """Structured output of the FAQ prompt."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    tags: list[str] = Field(default_factory=list, max_length=5)
    answer_is_complete: bool = True

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: object) -> object:
        if isinstance(value, list):
            seen: dict[str, None] = dict.fromkeys(str(t).strip().lower().lstrip("#") for t in value if str(t).strip())
            return list(seen)[:5]
        return value

    def render(self) -> str:
        """Markdown Q&A block; incomplete answers carry a visible note."""
        parts = [f"### {self.question.strip()}", "", self.answer.strip()]
        if not self.answer_is_complete:
            parts += ["", "> **Note:** this answer is partial — the source thread did not contain a full solution."]
        if self.tags:
            parts += ["", "Tags: " + ", ".join(f"`{t}`" for t in self.tags)]
        return "\n".join(parts)


def thread_messages(
    topic: RecurringTopic,
    items: Iterable[EnrichedMessage],
    *,
    include_replies: bool = True,
) -> list[EnrichedMessage]:
    """Messages backing a recurring topic, plus any replies in the same threads.

    Replies are what usually contain the *answer*; the topic itself only
    tracks the questions.
    """
    by_id = {i.message.message_id: i for i in items}
    wanted = [by_id[mid] for mid in topic.message_ids if mid in by_id]
    if include_replies:
        thread_ids = {i.message.thread_id or i.message.message_id for i in wanted}
        for item in items:
            if item.message.thread_id in thread_ids and item not in wanted:
                wanted.append(item)
    return sorted(wanted, key=lambda i: i.message.timestamp)


class FAQGenerator(ContentGenerator[FAQEntry]):
    """Turns a recurring doubt (and its thread) into one FAQ entry."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("faq", FAQEntry, llm=llm, settings=settings)

    def generate(
        self,
        items: Sequence[EnrichedMessage],
        *,
        topic: str | None = None,
        language: str = "en",
    ) -> GeneratedAsset:
        """Draft an entry from a set of question/answer messages."""
        if not items:
            raise ValueError("FAQGenerator.generate needs at least one message")
        label = topic or (items[0].themes.topics[0] if items[0].themes is not None and items[0].themes.topics else "")
        context = (
            f"Recurring doubt: {label or 'see messages'}. "
            f"{len(items)} message(s) from the community about it; later messages may contain the answer."
        )
        entry = self.invoke(items, context=context, language=language)
        return GeneratedAsset(
            channel="faq",
            title=entry.question.strip() or f"FAQ — {label}",
            content=entry,
            markdown=entry.render(),
            language=language,
            source_message_ids=tuple(i.message.message_id for i in items),
            metadata={"topic": label, "answer_is_complete": entry.answer_is_complete},
        )

    def generate_for_topic(
        self,
        topic: RecurringTopic,
        items: Sequence[EnrichedMessage],
        *,
        language: str = "en",
    ) -> GeneratedAsset:
        """Draft an entry for one R2 :class:`RecurringTopic`."""
        thread = thread_messages(topic, items)
        if not thread:
            raise ValueError(f"None of the topic's messages {topic.message_ids} are present in the batch")
        asset = self.generate(thread, topic=topic.topic, language=language)
        metadata = {
            **asset.metadata,
            "topic_key": topic.topic_key,
            "occurrences": topic.occurrences,
            "distinct_authors": topic.distinct_authors,
            "recommended_action": topic.action,
        }
        return asset.model_copy(update={"metadata": metadata})

    def generate_many(
        self,
        topics: Iterable[RecurringTopic],
        items: Sequence[EnrichedMessage],
        *,
        language: str = "en",
        only_faq_actions: bool = True,
        raise_on_error: bool = False,
    ) -> list[GeneratedAsset]:
        """One entry per recurring topic whose R2 action asks for a FAQ."""
        assets: list[GeneratedAsset] = []
        for topic in topics:
            if only_faq_actions and topic.action not in ("faq", "faq_and_mentorship"):
                continue
            try:
                assets.append(self.generate_for_topic(topic, items, language=language))
            except (GenerationError, ValueError) as exc:
                if raise_on_error:
                    raise
                logger.warning("Skipping topic %s: %s", topic.topic_key, exc)
        return assets
