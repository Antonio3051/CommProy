"""Rule R2 — identify doubts that keep coming back.

When the same doubt appears in ``min_occurrences`` (default 3) or more
messages it is no longer an individual problem but a gap in the material or
the onboarding. The recommended action depends on *who* is asking:

    many different members  -> "faq"          (document it once for everyone)
    one member, repeatedly  -> "mentorship"   (that person needs a human)
    very frequent + many    -> "faq_and_mentorship" (systemic gap: do both)

Topics come from the themes analyzer (``ThemesResult.topics`` first, then
``keywords``) and are normalised so "Virtual Environments" and
"virtual environments" count as the same doubt.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.decisions.config import RecurringTopicsConfig
from src.decisions.member_risk import member_key_of

RecurringAction = Literal["faq", "mentorship", "faq_and_mentorship"]

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_topic(text: str) -> str:
    """Canonical key for a topic: ASCII-folded, lowercase, single-spaced words.

    >>> normalize_topic("  Entornos Virtuales (venv) ")
    'entornos virtuales venv'
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_WORD.sub(" ", folded.lower()).strip()


class RecurringTopic(BaseModel):
    """A doubt that appeared ``occurrences`` times across the batch.

    Attributes:
        topic_key: Normalised topic used for grouping.
        topic: Most common human-readable spelling of the topic.
        occurrences: Number of distinct messages mentioning it.
        distinct_authors: Number of different members asking.
        action: Recommended follow-up (``faq`` / ``mentorship`` / both).
        technologies: Technologies mentioned alongside the topic.
        message_ids: Messages to feed into the FAQ generator.
        authors: Members involved (for mentorship assignment).
        sample_questions: Up to three representative message excerpts.
        rationale: Why this action was chosen.
    """

    model_config = ConfigDict(frozen=True)

    topic_key: str
    topic: str
    occurrences: int = Field(ge=1)
    distinct_authors: int = Field(ge=1)
    action: RecurringAction
    technologies: tuple[str, ...] = ()
    message_ids: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    sample_questions: tuple[str, ...] = ()
    rationale: str = ""


def _is_doubt(item: EnrichedMessage, config: RecurringTopicsConfig) -> bool:
    """A message counts as a doubt when categorised as one, or when it asks a question."""
    if item.themes is not None:
        return item.themes.primary_category in config.doubt_categories
    # Fallback when the themes analyzer failed: rely on the ingestion category
    # or a trailing question mark.
    category = (item.message.category or "").lower()
    return category in config.doubt_categories or item.message.content.rstrip().endswith("?")


def _topic_labels(item: EnrichedMessage) -> list[str]:
    """Topic phrases for a message: analyzer topics, else keywords, else category."""
    if item.themes is None:
        return [item.message.category] if item.message.category else []
    if item.themes.topics:
        return list(item.themes.topics)
    if item.themes.keywords:
        return list(item.themes.keywords)
    return [item.themes.primary_category]


def _decide_action(
    occurrences: int, distinct_authors: int, config: RecurringTopicsConfig
) -> tuple[RecurringAction, str]:
    # Systemic gap: lots of messages from several people -> both actions.
    if occurrences >= config.occurrences_for_both_actions and distinct_authors >= config.min_authors_for_faq:
        return (
            "faq_and_mentorship",
            f"{occurrences} messages from {distinct_authors} members: document it AND offer a mentoring session",
        )
    # Widespread doubt: several people -> a FAQ entry solves it for everyone.
    if distinct_authors >= config.min_authors_for_faq:
        return "faq", f"{distinct_authors} different members asked about it: add it to the FAQ"
    # One person asking again and again -> they are stuck; pair them with a mentor.
    return "mentorship", f"a single member asked {occurrences} times: assign a mentor"


def _excerpt(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def detect_recurring_topics(
    enriched: Iterable[EnrichedMessage],
    *,
    config: RecurringTopicsConfig | None = None,
) -> list[RecurringTopic]:
    """Group doubts by topic and return those at or above ``min_occurrences``.

    Each message contributes once per distinct topic it mentions, so a
    message about "venv" and "pip" counts towards both topics but never
    twice towards the same one.

    Returns:
        Topics sorted by descending occurrences, then distinct authors, then name.
    """
    config = config or RecurringTopicsConfig()
    buckets: dict[str, list[EnrichedMessage]] = defaultdict(list)
    spellings: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for item in enriched:
        if not _is_doubt(item, config):
            continue
        seen_keys: set[str] = set()
        for label in _topic_labels(item):
            key = normalize_topic(label)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            buckets[key].append(item)
            spellings[key][label.strip()] += 1

    results: list[RecurringTopic] = []
    for key, items in buckets.items():
        if len(items) < config.min_occurrences:
            continue
        authors = list(dict.fromkeys(member_key_of(i) for i in items))
        author_names = list(dict.fromkeys(i.message.author for i in items))
        action, rationale = _decide_action(len(items), len(authors), config)
        technologies = list(dict.fromkeys(t for i in items if i.themes is not None for t in i.themes.technologies))
        topic = max(spellings[key].items(), key=lambda kv: kv[1])[0]
        results.append(
            RecurringTopic(
                topic_key=key,
                topic=topic,
                occurrences=len(items),
                distinct_authors=len(authors),
                action=action,
                technologies=tuple(technologies),
                message_ids=tuple(i.message.message_id for i in items),
                authors=tuple(author_names),
                sample_questions=tuple(_excerpt(i.message.content) for i in items[:3]),
                rationale=rationale,
            )
        )

    return sorted(results, key=lambda t: (-t.occurrences, -t.distinct_authors, t.topic_key))
