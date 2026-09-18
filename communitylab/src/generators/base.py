"""Shared plumbing for the content generators.

Every generator is a :class:`ContentGenerator`: a Phase 2 copywriting prompt
(:mod:`src.prompts.system_prompts`) plus a *brief* built from enriched
messages, piped into a Groq model constrained to a Pydantic schema. The
structured output is wrapped in a :class:`GeneratedAsset`, which knows how
to render itself as Markdown/JSON and how to name its file so the OCI
storage layer can persist it without knowing about channels.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.prompts.system_prompts import Channel, get_copywriting_prompt
from src.utils.llm_clients import LLMSettings, get_structured_llm

logger = logging.getLogger(__name__)

BRIEF_HUMAN_TEMPLATE: Final[str] = """\
Write in this language: {language}

Context for this piece:
{context}

Source community messages (with AI analysis):
{messages}
"""
"""Human turn shared by all generators; variables come from :func:`build_brief`."""


class GenerationError(RuntimeError):
    """Raised when a generator cannot produce an asset."""

    def __init__(self, generator: str, cause: BaseException) -> None:
        self.generator = generator
        self.cause = cause
        super().__init__(f"{generator} generation failed: {cause}")


class GeneratedAsset(BaseModel):
    """A finished piece of content plus the metadata needed to store and trace it.

    Attributes:
        channel: Copywriting channel that produced it.
        title: Short human title (also used to build the file name).
        content: The structured LLM output (``LinkedInPost``, ``FAQEntry``, ...).
        markdown: Rendered Markdown version of ``content``.
        language: Language the copy was requested in.
        source_message_ids: Messages the asset was built from.
        generated_at: Creation timestamp (UTC).
        metadata: Free-form extras (topic key, period, tier...).
    """

    model_config = ConfigDict(frozen=True)

    channel: Channel
    title: str
    content: BaseModel
    markdown: str
    language: str = "en"
    source_message_ids: tuple[str, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def slug(self) -> str:
        """File-system friendly identifier derived from the title."""
        return slugify(self.title) or self.channel

    def filename(self, extension: str = "md") -> str:
        """``<channel>/<date>-<slug>.<ext>``, the object name used in storage."""
        stamp = self.generated_at.strftime("%Y%m%d-%H%M%S")
        return f"{self.channel}/{stamp}-{self.slug}.{extension.lstrip('.')}"

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable dict with the structured content unpacked."""
        return {
            "channel": self.channel,
            "title": self.title,
            "language": self.language,
            "generated_at": self.generated_at.isoformat(),
            "source_message_ids": list(self.source_message_ids),
            "metadata": self.metadata,
            "content": self.content.model_dump(mode="json"),
            "markdown": self.markdown,
        }


def slugify(text: str, max_length: int = 60) -> str:
    """ASCII-only, lowercase, hyphen-separated slug."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    return slug[:max_length].rstrip("-")


def format_message_block(item: EnrichedMessage) -> str:
    """One source message rendered for the LLM brief, including its analysis."""
    base = item.message
    lines = [
        f"- id: {base.message_id} | {base.source} #{base.channel} | author: {base.author} | "
        f"{base.timestamp.date().isoformat()} | reactions: {base.reactions}"
    ]
    if item.sentiment is not None:
        lines.append(f"  sentiment: {item.sentiment.label} ({item.sentiment.score:+.2f})")
    if item.themes is not None:
        topics = ", ".join(item.themes.topics) or "-"
        tech = ", ".join(item.themes.technologies) or "-"
        lines.append(f"  category: {item.themes.primary_category} | topics: {topics} | technologies: {tech}")
    if item.relevance is not None:
        lines.append(f"  relevance: {item.relevance.tier} ({item.relevance.score:.2f})")
    lines.append('  text: """' + " ".join(base.content.split()) + '"""')
    return "\n".join(lines)


def build_brief(
    items: Iterable[EnrichedMessage],
    *,
    context: str,
    language: str = "en",
) -> dict[str, str]:
    """Variables for :data:`BRIEF_HUMAN_TEMPLATE`."""
    blocks = [format_message_block(i) for i in items]
    return {
        "language": language,
        "context": context.strip() or "(none)",
        "messages": "\n".join(blocks) if blocks else "(no messages)",
    }


def build_brief_prompt(system_prompt: str, human_template: str = BRIEF_HUMAN_TEMPLATE) -> ChatPromptTemplate:
    """System prompt (brace-safe) + templated brief."""
    return ChatPromptTemplate.from_messages([SystemMessage(content=system_prompt), ("human", human_template)])


class ContentGenerator[ResultT: BaseModel]:
    """Prompt + structured Groq model returning ``schema`` instances.

    Args:
        channel: Copywriting channel; selects the Phase 2 system prompt.
        schema: Pydantic model the LLM output must satisfy.
        llm: Optional pre-built structured runnable (tests inject fakes here).
        settings: LLM settings used when ``llm`` is not provided.
    """

    def __init__(
        self,
        channel: Channel,
        schema: type[ResultT],
        *,
        llm: Runnable | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self.channel: Channel = channel
        self.schema = schema
        self.prompt = build_brief_prompt(get_copywriting_prompt(channel))
        self._llm: Runnable = llm if llm is not None else get_structured_llm(schema, settings)
        self.chain: Runnable = self.prompt | self._llm

    def _coerce(self, output: Any) -> ResultT:
        if isinstance(output, self.schema):
            return output
        try:
            return self.schema.model_validate(output)
        except Exception as exc:  # noqa: BLE001 - re-raised as GenerationError
            raise GenerationError(self.channel, exc) from exc

    def invoke(
        self,
        items: Sequence[EnrichedMessage],
        *,
        context: str,
        language: str = "en",
        config: RunnableConfig | None = None,
    ) -> ResultT:
        """Run the chain on a brief built from ``items``.

        Raises:
            GenerationError: If the model call fails on every fallback or the
                output does not match ``schema``.
        """
        brief = build_brief(items, context=context, language=language)
        try:
            output = self.chain.invoke(brief, config=config)
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as GenerationError
            raise GenerationError(self.channel, exc) from exc
        result = self._coerce(output)
        logger.debug("%s generated from %d message(s)", self.channel, len(items))
        return result
