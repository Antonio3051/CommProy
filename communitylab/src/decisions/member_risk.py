"""Rule R1 — detect members at risk of disengaging or churning.

A member is evaluated on three independent signals:

1. **Negative sentiment** — their messages are, on average, clearly negative
   or they have posted several negative messages.
2. **Frustration** — they use frustration vocabulary ("giving up", "no
   funciona", …) or the sentiment analyzer detected frustration/anger.
3. **Inactivity** — they have not posted for ``inactivity_days``.

Each signal contributes a weighted amount to a 0..1 ``risk_score``; the number
of signals decides the discrete ``level``:

    0 signals -> "none"     (not reported)
    1 signal  -> "watch"    (keep an eye, no action yet)
    2 signals -> "at_risk"  (proactive outreach recommended)
    3 signals -> "high"     (immediate personal contact)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.decisions.config import MemberRiskConfig

RiskSignal = Literal["negative_sentiment", "frustration", "inactivity"]
RiskLevel = Literal["none", "watch", "at_risk", "high"]

# Signal count -> level. Kept as a lookup table so the mapping is explicit
# and easy to change (e.g. if the business decides 2 signals is already
# "high" for a premium cohort).
_LEVEL_BY_SIGNAL_COUNT: dict[int, RiskLevel] = {0: "none", 1: "watch", 2: "at_risk", 3: "high"}


class MemberRisk(BaseModel):
    """Risk assessment for one community member.

    Attributes:
        member_key: Stable identifier (``author_id`` when known, else ``author``).
        author: Display name.
        signals: Which of the three signals fired.
        risk_score: Weighted 0..1 score (see :class:`MemberRiskConfig`).
        level: Discrete level derived from the number of signals.
        evidence: Human-readable justification per signal.
        message_count: Messages by this member in the analysed batch.
        negative_count: Messages labelled ``negative``.
        mean_sentiment: Average sentiment score (``None`` if no sentiment available).
        last_active: Timestamp of the member's most recent message.
        days_inactive: Days between ``last_active`` and the reference time.
        message_ids: Messages that triggered a signal (for drill-down).
    """

    model_config = ConfigDict(frozen=True)

    member_key: str
    author: str
    signals: tuple[RiskSignal, ...] = ()
    risk_score: float = Field(ge=0.0, le=1.0)
    level: RiskLevel
    evidence: tuple[str, ...] = ()
    message_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    mean_sentiment: float | None = None
    last_active: datetime
    days_inactive: float = Field(ge=0.0)
    message_ids: tuple[str, ...] = ()

    @property
    def is_at_risk(self) -> bool:
        """``True`` for ``at_risk`` and ``high`` (two or more signals)."""
        return self.level in ("at_risk", "high")


def member_key_of(message: EnrichedMessage) -> str:
    """Group messages by ``author_id`` when available, else by display name."""
    base = message.message
    return base.author_id or base.author


def _group_by_member(enriched: Iterable[EnrichedMessage]) -> dict[str, list[EnrichedMessage]]:
    groups: dict[str, list[EnrichedMessage]] = defaultdict(list)
    for item in enriched:
        groups[member_key_of(item)].append(item)
    return groups


def _matches_frustration(item: EnrichedMessage, config: MemberRiskConfig) -> str | None:
    """Return the keyword/emotion that matched, or ``None``."""
    content = item.message.content.lower()
    for keyword in config.frustration_keywords:
        if keyword in content:
            return f'keyword "{keyword}"'
    if item.sentiment is not None:
        for emotion in item.sentiment.emotions:
            if emotion in config.frustration_emotions:
                return f'emotion "{emotion}"'
    return None


def assess_member(
    messages: list[EnrichedMessage],
    *,
    reference_time: datetime,
    config: MemberRiskConfig,
) -> MemberRisk:
    """Evaluate the three risk signals for one member's messages."""
    if not messages:
        raise ValueError("assess_member requires at least one message")
    messages = sorted(messages, key=lambda m: m.message.timestamp)
    latest = messages[-1].message
    signals: list[RiskSignal] = []
    evidence: list[str] = []
    flagged_ids: list[str] = []

    # ---- Signal 1: negative sentiment -------------------------------------
    # Only messages where the sentiment analyzer succeeded are considered;
    # messages without a sentiment neither help nor hurt the member.
    scores = [m.sentiment.score for m in messages if m.sentiment is not None]
    negatives = [m for m in messages if m.sentiment is not None and m.sentiment.label == "negative"]
    mean_sentiment = sum(scores) / len(scores) if scores else None
    if (mean_sentiment is not None and mean_sentiment <= config.negative_mean_score) or len(
        negatives
    ) >= config.min_negative_messages:
        signals.append("negative_sentiment")
        mean_text = f"{mean_sentiment:+.2f}" if mean_sentiment is not None else "n/a"
        evidence.append(
            f"{len(negatives)} negative message(s), mean sentiment {mean_text} "
            f"(threshold {config.negative_mean_score:+.2f} or >= {config.min_negative_messages} negatives)"
        )
        flagged_ids.extend(m.message.message_id for m in negatives)

    # ---- Signal 2: frustration vocabulary / emotions ------------------------
    for item in messages:
        match = _matches_frustration(item, config)
        if match is not None:
            signals.append("frustration")
            evidence.append(f"frustration {match} in message {item.message.message_id}")
            flagged_ids.append(item.message.message_id)
            break  # one match is enough; don't inflate the evidence list

    # ---- Signal 3: inactivity -----------------------------------------------
    # Measured from the member's last message to `reference_time` (normally
    # "now"). Members with fewer than `min_messages_for_inactivity` messages
    # are skipped because we cannot tell "inactive" from "never joined".
    days_inactive = max(0.0, (reference_time - latest.timestamp).total_seconds() / 86_400)
    if len(messages) >= config.min_messages_for_inactivity and days_inactive >= config.inactivity_days:
        signals.append("inactivity")
        evidence.append(
            f"last message {days_inactive:.0f} day(s) ago on {latest.timestamp.date().isoformat()} "
            f"(threshold {config.inactivity_days} days)"
        )

    # ---- Score & level ------------------------------------------------------
    weights: dict[RiskSignal, float] = {
        "negative_sentiment": config.weight_negative_sentiment,
        "frustration": config.weight_frustration,
        "inactivity": config.weight_inactivity,
    }
    risk_score = min(1.0, sum(weights[s] for s in signals))
    level = _LEVEL_BY_SIGNAL_COUNT[len(signals)]

    return MemberRisk(
        member_key=member_key_of(messages[-1]),
        author=latest.author,
        signals=tuple(signals),
        risk_score=round(risk_score, 3),
        level=level,
        evidence=tuple(evidence),
        message_count=len(messages),
        negative_count=len(negatives),
        mean_sentiment=round(mean_sentiment, 3) if mean_sentiment is not None else None,
        last_active=latest.timestamp,
        days_inactive=round(days_inactive, 2),
        message_ids=tuple(dict.fromkeys(flagged_ids)),
    )


def detect_member_risk(
    enriched: Iterable[EnrichedMessage],
    *,
    reference_time: datetime | None = None,
    config: MemberRiskConfig | None = None,
    include_watch: bool = True,
) -> list[MemberRisk]:
    """Assess every member in ``enriched`` and return those with at least one signal.

    Args:
        enriched: Consolidated analysis output (any order).
        reference_time: "Now" for the inactivity signal; defaults to the current UTC time.
        config: Thresholds; defaults to :class:`MemberRiskConfig`.
        include_watch: Also return members with a single signal (``watch``).

    Returns:
        Members sorted by descending ``risk_score`` (ties broken by name).
    """
    config = config or MemberRiskConfig()
    reference_time = reference_time or datetime.now(UTC)
    results = [
        assess_member(messages, reference_time=reference_time, config=config)
        for messages in _group_by_member(enriched).values()
    ]
    minimum = 1 if include_watch else 2
    flagged = [r for r in results if len(r.signals) >= minimum]
    return sorted(flagged, key=lambda r: (-r.risk_score, r.author.lower()))
