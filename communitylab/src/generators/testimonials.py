"""Testimonial extraction from positive community messages."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.generators.base import ContentGenerator, GeneratedAsset, GenerationError
from src.utils.llm_clients import LLMSettings

logger = logging.getLogger(__name__)

# A testimonial must be genuinely positive; "mixed" or lukewarm messages make
# weak marketing copy, so we require a clearly positive score.
MIN_SENTIMENT_SCORE = 0.3


class Testimonial(BaseModel):
    """Structured output of the testimonial prompt."""

    model_config = ConfigDict(frozen=True)

    quote: str = Field(min_length=1)
    context: str = ""
    attribution: str = "Community member"
    needs_consent: bool = False

    def render(self) -> str:
        """Markdown blockquote with attribution; flags consent when needed."""
        parts = [f"> {self.quote.strip()}", f"> — *{self.attribution.strip() or 'Community member'}*"]
        if self.context.strip():
            parts += ["", self.context.strip()]
        if self.needs_consent:
            parts += ["", "**Consent required before publishing** (contains personal or employer details)."]
        return "\n".join(parts)


def select_testimonial_candidates(
    items: Iterable[EnrichedMessage],
    *,
    min_score: float = MIN_SENTIMENT_SCORE,
) -> list[EnrichedMessage]:
    """Positive success stories, or messages the relevance analyzer suggested for testimonials."""
    picked: list[EnrichedMessage] = []
    for item in items:
        score = item.sentiment.score if item.sentiment is not None else 0.0
        if score < min_score:
            continue
        is_story = item.themes is not None and item.themes.primary_category == "success_story"
        suggested = item.relevance is not None and "testimonial" in item.relevance.suggested_channels
        if is_story or suggested:
            picked.append(item)
    return sorted(picked, key=lambda i: -(i.sentiment.score if i.sentiment is not None else 0.0))


class TestimonialGenerator(ContentGenerator[Testimonial]):
    """Extracts a clean, quotable testimonial from one message."""

    def __init__(self, *, llm: Runnable | None = None, settings: LLMSettings | None = None) -> None:
        super().__init__("testimonial", Testimonial, llm=llm, settings=settings)

    def generate(self, item: EnrichedMessage, *, language: str = "en", allow_names: bool = False) -> GeneratedAsset:
        """Format one message as a testimonial."""
        naming = (
            "The member's first name may be used in the attribution."
            if allow_names
            else "Do not use the member's name; attribute to their role/track or 'Community member'."
        )
        testimonial = self.invoke([item], context=naming, language=language)
        return GeneratedAsset(
            channel="testimonial",
            title=f"Testimonial — {testimonial.attribution.strip() or item.message.author}",
            content=testimonial,
            markdown=testimonial.render(),
            language=language,
            source_message_ids=(item.message.message_id,),
            metadata={
                "author": item.message.author,
                "needs_consent": testimonial.needs_consent,
                "sentiment_score": item.sentiment.score if item.sentiment is not None else None,
            },
        )

    def generate_many(
        self,
        items: Sequence[EnrichedMessage],
        *,
        language: str = "en",
        limit: int | None = None,
        allow_names: bool = False,
        raise_on_error: bool = False,
    ) -> list[GeneratedAsset]:
        """One testimonial per candidate (see :func:`select_testimonial_candidates`)."""
        assets: list[GeneratedAsset] = []
        for item in select_testimonial_candidates(items)[:limit]:
            try:
                assets.append(self.generate(item, language=language, allow_names=allow_names))
            except GenerationError as exc:
                if raise_on_error:
                    raise
                logger.warning("Skipping message %s: %s", item.message.message_id, exc)
        return assets
