"""LinkedIn post generator for community success stories."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.analysis.consolidator import EnrichedMessage
from src.generators.base import ContentGenerator, GeneratedAsset, GenerationError
from src.utils.llm_clients import LLMSettings

logger = logging.getLogger(__name__)


class LinkedInPost(BaseModel):
    """Structured output of the LinkedIn prompt."""

    model_config = ConfigDict(frozen=True)

    hook: str = Field(description="First line of the post.")
    body: str = Field(description="Full post text including hook and call to action.")
    hashtags: list[str] = Field(default_factory=list, max_length=3, description="Hashtags without '#'.")

    @field_validator("hashtags", mode="before")
    @classmethod
    def _strip_hashes(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(tag).lstrip("#").strip() for tag in value if str(tag).strip("# ")][:3]
        return value

    def render(self) -> str:
        """Post text ready to paste, hashtags on the last line."""
        tags = " ".join(f"#{t}" for t in self.hashtags)
        return f"{self.body.rstrip()}\n\n{tags}".rstrip() if tags else self.body.rstrip()


def select_success_stories(items: Iterable[EnrichedMessage]) -> list[EnrichedMessage]:
    """Messages worth a LinkedIn post: success stories or explicitly suggested for LinkedIn.

    Negative messages are never selected even if tagged as success stories.
    Results are sorted by relevance (highest first).
    """
    picked: list[EnrichedMessage] = []
    for item in items:
        if item.sentiment is not None and item.sentiment.label == "negative":
            continue
        is_story = item.themes is not None and item.themes.primary_category == "success_story"
        suggested = item.relevance is not None and "linkedin" in item.relevance.suggested_channels
        if is_story or suggested:
            picked.append(item)
    return sorted(picked, key=lambda i: -(i.relevance.score if i.relevance is not None else 0.0))


class LinkedInGenerator(ContentGenerator[LinkedInPost]):
    """Turns one success story into a LinkedIn post."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("linkedin", LinkedInPost, llm=llm, settings=settings)

    def generate(self, item: EnrichedMessage, *, language: str = "en", context: str = "") -> GeneratedAsset:
        """Draft a post for a single message."""
        extra = context or "Celebrate this member's achievement and extract one lesson for the audience."
        post = self.invoke([item], context=extra, language=language)
        title = post.hook.strip() or f"LinkedIn post — {item.message.author}"
        return GeneratedAsset(
            channel="linkedin",
            title=title,
            content=post,
            markdown=post.render(),
            language=language,
            source_message_ids=(item.message.message_id,),
            metadata={
                "author": item.message.author,
                "source": item.message.source,
                "topics": list(item.themes.topics) if item.themes is not None else [],
            },
        )

    def generate_many(
        self,
        items: Sequence[EnrichedMessage],
        *,
        language: str = "en",
        limit: int | None = None,
        raise_on_error: bool = False,
    ) -> list[GeneratedAsset]:
        """One post per selected success story (see :func:`select_success_stories`)."""
        assets: list[GeneratedAsset] = []
        for item in select_success_stories(items)[:limit]:
            try:
                assets.append(self.generate(item, language=language))
            except GenerationError as exc:
                if raise_on_error:
                    raise
                logger.warning("Skipping message %s: %s", item.message.message_id, exc)
        return assets
