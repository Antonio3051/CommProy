"""Rule R3 — route each interaction to an escalation level.

Levels, from lowest to highest urgency:

    INFO     nothing to do; the message is context or noise.
    MEDIO    somebody should look at it within days (open question, mild complaint).
    ALTO     needs a human within a day (high-value, negative & actionable, member at risk).
    CRÍTICO  drop everything (critical relevance, very negative public complaint,
             high-risk member venting, or several ALTO triggers at once).

The routing combines two inputs: the *relevance* of the message (from the
relevance analyzer) and the *risk* around it (sentiment of the message and
the R1 risk level of its author). The rules are evaluated top-down; the first
matching level wins.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.decisions.config import EscalationConfig
from src.decisions.member_risk import MemberRisk, RiskLevel


class EscalationLevel(StrEnum):
    """Ordered escalation levels (``INFO < MEDIO < ALTO < CRITICO``)."""

    INFO = "INFO"
    MEDIO = "MEDIO"
    ALTO = "ALTO"
    CRITICO = "CRÍTICO"

    @property
    def rank(self) -> int:
        """Numeric urgency, useful for sorting and comparisons."""
        return _RANK[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EscalationLevel):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, EscalationLevel):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, EscalationLevel):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, EscalationLevel):
            return NotImplemented
        return self.rank >= other.rank


_RANK: dict[EscalationLevel, int] = {
    EscalationLevel.INFO: 0,
    EscalationLevel.MEDIO: 1,
    EscalationLevel.ALTO: 2,
    EscalationLevel.CRITICO: 3,
}

LEVELS_BY_URGENCY: tuple[EscalationLevel, ...] = (
    EscalationLevel.CRITICO,
    EscalationLevel.ALTO,
    EscalationLevel.MEDIO,
    EscalationLevel.INFO,
)


class Escalation(BaseModel):
    """Routing decision for one message.

    Attributes:
        message_id: The routed message.
        author: Display name of the author.
        member_key: Stable member identifier (matches :class:`MemberRisk.member_key`).
        level: Assigned escalation level.
        reasons: Rules that fired, in evaluation order.
        relevance_score: Copied from the analysis (``None`` if unavailable).
        sentiment_score: Copied from the analysis (``None`` if unavailable).
        member_risk_level: R1 level of the author (``none`` when not flagged).
        requires_response: Whether the message is still waiting for an answer.
        sla_hours: Response deadline implied by the level (``None`` for INFO).
        excerpt: Short preview of the content for notifications.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    author: str
    member_key: str
    level: EscalationLevel
    reasons: tuple[str, ...] = ()
    relevance_score: float | None = None
    sentiment_score: float | None = None
    member_risk_level: RiskLevel = "none"
    requires_response: bool = False
    sla_hours: int | None = Field(default=None, ge=0)
    excerpt: str = ""

    @property
    def is_urgent(self) -> bool:
        """``ALTO`` or ``CRÍTICO``."""
        return self.level >= EscalationLevel.ALTO


def sla_for_level(level: EscalationLevel, config: EscalationConfig) -> int | None:
    """Response SLA in hours for ``level`` (``None`` when no response is expected)."""
    return {
        EscalationLevel.CRITICO: config.sla_hours_critico,
        EscalationLevel.ALTO: config.sla_hours_alto,
        EscalationLevel.MEDIO: config.sla_hours_medio,
        EscalationLevel.INFO: None,
    }[level]


def _excerpt(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def route(
    item: EnrichedMessage,
    member_risk: MemberRisk | None = None,
    *,
    config: EscalationConfig | None = None,
) -> Escalation:
    """Assign an escalation level to one enriched message.

    Args:
        item: Consolidated analysis for the message.
        member_risk: R1 assessment of the author, if any.
        config: Thresholds; defaults to :class:`EscalationConfig`.
    """
    config = config or EscalationConfig()
    base = item.message
    relevance = item.relevance.score if item.relevance is not None else None
    sentiment = item.sentiment.score if item.sentiment is not None else None
    category = item.themes.primary_category if item.themes is not None else (base.category or "")
    is_actionable = item.relevance.is_actionable if item.relevance is not None else False
    requires_response = item.relevance.requires_response if item.relevance is not None else False
    risk_level: RiskLevel = member_risk.level if member_risk is not None else "none"

    # Missing analyses are treated conservatively: an unknown relevance is
    # scored as 0 (never escalates on its own) and unknown sentiment as 0
    # (neutral). We still route the message so nothing silently disappears.
    rel = relevance if relevance is not None else 0.0
    sent = sentiment if sentiment is not None else 0.0

    critical: list[str] = []
    high: list[str] = []
    medium: list[str] = []

    # ---------------------------------------------------------------- CRÍTICO
    # C1. The relevance analyzer itself flagged the message as critical
    #     (score >= 0.85: outages, blockers affecting many, public incidents).
    if rel >= config.relevance_critico:
        critical.append(f"relevance {rel:.2f} >= {config.relevance_critico} (critical tier)")
    # C2. Very negative AND at least high relevance: an angry, visible complaint
    #     that the community will read. Silence here costs reputation.
    if sent <= config.sentiment_very_negative and rel >= config.relevance_alto:
        critical.append(f"very negative sentiment {sent:+.2f} on a high-relevance message")
    # C3. A member already at high risk (all three R1 signals) posting a
    #     negative message is about to leave; treat as an incident.
    if risk_level == "high" and sent <= config.sentiment_negative:
        critical.append("high-risk member (3 R1 signals) posting negatively")

    # ------------------------------------------------------------------- ALTO
    # A1. High relevance tier (0.6..0.85): detailed questions, success stories
    #     worth amplifying, bug reports.
    if rel >= config.relevance_alto:
        high.append(f"relevance {rel:.2f} >= {config.relevance_alto} (high tier)")
    # A2. Negative and actionable: the team can do something about it now.
    if sent <= config.sentiment_negative and is_actionable:
        high.append(f"negative sentiment {sent:+.2f} and actionable")
    # A3. The author is at risk (2+ R1 signals): every interaction with them
    #     is an opportunity to retain them.
    if risk_level in ("at_risk", "high"):
        high.append(f"author flagged {risk_level} by R1")
    # A4. Complaints are always at least ALTO when they are still unanswered.
    if category == "complaint" and requires_response:
        high.append("unanswered complaint")

    # ------------------------------------------------------------------ MEDIO
    # M1. Medium relevance tier (0.4..0.6).
    if rel >= config.relevance_medio:
        medium.append(f"relevance {rel:.2f} >= {config.relevance_medio} (medium tier)")
    # M2. Anything still waiting for an answer deserves a look this week.
    if requires_response:
        medium.append("requires a response")
    # M3. Mildly negative messages and feature requests are worth tracking.
    if sent <= config.sentiment_negative:
        medium.append(f"negative sentiment {sent:+.2f}")
    if category in ("complaint", "feature_request"):
        medium.append(f"category {category}")
    # M4. A member on watch (1 R1 signal) gets a slightly higher floor.
    if risk_level == "watch":
        medium.append("author on R1 watch list")

    # ------------------------------------------------------------- Resolution
    # First matching level wins. Two independent ALTO triggers escalate to
    # CRÍTICO: several moderate red flags at once are as serious as one big one.
    if critical:
        level, reasons = EscalationLevel.CRITICO, critical
    elif len(high) >= 2:
        level, reasons = EscalationLevel.CRITICO, [f"multiple ALTO triggers: {'; '.join(high)}"]
    elif high:
        level, reasons = EscalationLevel.ALTO, high
    elif medium:
        level, reasons = EscalationLevel.MEDIO, medium
    else:
        level, reasons = EscalationLevel.INFO, ["no relevance or risk triggers"]

    return Escalation(
        message_id=base.message_id,
        author=base.author,
        member_key=base.author_id or base.author,
        level=level,
        reasons=tuple(reasons),
        relevance_score=relevance,
        sentiment_score=sentiment,
        member_risk_level=risk_level,
        requires_response=requires_response,
        sla_hours=sla_for_level(level, config),
        excerpt=_excerpt(base.content),
    )


def route_all(
    enriched: list[EnrichedMessage],
    member_risks: list[MemberRisk] | None = None,
    *,
    config: EscalationConfig | None = None,
) -> list[Escalation]:
    """Route every message, sorted by urgency (CRÍTICO first) then relevance."""
    risk_by_member = {r.member_key: r for r in member_risks or []}
    routed = [
        route(item, risk_by_member.get(item.message.author_id or item.message.author), config=config)
        for item in enriched
    ]
    return sorted(routed, key=lambda e: (-e.level.rank, -(e.relevance_score or 0.0), e.message_id))
