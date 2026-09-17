"""Builders for alert messages, one style per escalation level.

Each level maps to a delivery *target* and a tone:

    INFO     -> daily digest (batched, nobody is pinged)
    MEDIO    -> team channel (visible, no mention)
    ALTO     -> team channel with @mentors mention, 24h SLA
    CRÍTICO  -> direct message to the community lead + channel mention, 2h SLA

Notifications are plain :class:`Notification` objects; the transport (Slack,
Discord bot, e-mail, n8n webhook) lives in later layers and only needs
``title``/``body``/``target``. ``body`` is Markdown-compatible text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.decisions.escalation import Escalation, EscalationLevel
from src.decisions.member_risk import MemberRisk
from src.decisions.recurring_topics import RecurringTopic

NotificationTarget = Literal["digest", "channel", "channel_mention", "direct_message"]
NotificationKind = Literal["escalation", "member_risk", "recurring_topic", "summary"]


class LevelStyle(BaseModel):
    """Presentation rules for one escalation level."""

    model_config = ConfigDict(frozen=True)

    emoji: str
    target: NotificationTarget
    audience: str
    urgency_text: str


# Presentation per level. `audience` is a human hint that later layers can
# translate to real Slack/Discord handles.
LEVEL_STYLES: dict[EscalationLevel, LevelStyle] = {
    EscalationLevel.INFO: LevelStyle(
        emoji="ℹ️", target="digest", audience="community team (daily digest)", urgency_text="no action required"
    ),
    EscalationLevel.MEDIO: LevelStyle(
        emoji="🟡", target="channel", audience="#community-ops", urgency_text="review this week"
    ),
    EscalationLevel.ALTO: LevelStyle(
        emoji="🟠", target="channel_mention", audience="@mentors in #community-ops", urgency_text="respond within 24h"
    ),
    EscalationLevel.CRITICO: LevelStyle(
        emoji="🔴",
        target="direct_message",
        audience="community lead (DM) + @mentors",
        urgency_text="IMMEDIATE action required",
    ),
}


class Notification(BaseModel):
    """A ready-to-send alert.

    Attributes:
        kind: What triggered it (escalation, member risk, recurring topic, summary).
        level: Escalation level driving tone and target.
        title: One-line headline (safe for Slack/Discord titles).
        body: Markdown body.
        target: Delivery target class.
        audience: Human-readable recipient hint.
        message_ids: Related messages for deep links.
        member_key: Related member, if any.
        due_by: Deadline derived from the SLA, if any.
        created_at: When the notification was built.
    """

    model_config = ConfigDict(frozen=True)

    kind: NotificationKind
    level: EscalationLevel
    title: str
    body: str
    target: NotificationTarget
    audience: str
    message_ids: tuple[str, ...] = ()
    member_key: str | None = None
    due_by: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _style(level: EscalationLevel) -> LevelStyle:
    return LEVEL_STYLES[level]


def _due(sla_hours: int | None, now: datetime) -> datetime | None:
    return None if sla_hours is None else now + timedelta(hours=sla_hours)


def build_escalation_alert(escalation: Escalation, *, now: datetime | None = None) -> Notification:
    """Alert for a single routed message; tone/target follow its level."""
    now = now or datetime.now(UTC)
    style = _style(escalation.level)
    sla = f" — {style.urgency_text}"
    if escalation.sla_hours is not None:
        sla += f" (SLA {escalation.sla_hours}h)"
    lines = [
        f"*Author:* {escalation.author}",
        f'*Message:* "{escalation.excerpt}"',
        "*Why:* " + "; ".join(escalation.reasons),
    ]
    metrics: list[str] = []
    if escalation.relevance_score is not None:
        metrics.append(f"relevance {escalation.relevance_score:.2f}")
    if escalation.sentiment_score is not None:
        metrics.append(f"sentiment {escalation.sentiment_score:+.2f}")
    if escalation.member_risk_level != "none":
        metrics.append(f"member risk {escalation.member_risk_level}")
    if metrics:
        lines.append("*Signals:* " + ", ".join(metrics))
    if escalation.requires_response:
        lines.append("*Status:* still unanswered")
    return Notification(
        kind="escalation",
        level=escalation.level,
        title=f"{style.emoji} [{escalation.level.value}] {escalation.author}: {escalation.excerpt[:60]}{sla}",
        body="\n".join(lines),
        target=style.target,
        audience=style.audience,
        message_ids=(escalation.message_id,),
        member_key=escalation.member_key,
        due_by=_due(escalation.sla_hours, now),
        created_at=now,
    )


# R1 level -> escalation level used for the alert's tone/target.
# "watch" is informational; two signals warrant a mentor; three is critical.
_RISK_TO_LEVEL: dict[str, EscalationLevel] = {
    "watch": EscalationLevel.INFO,
    "at_risk": EscalationLevel.ALTO,
    "high": EscalationLevel.CRITICO,
}


def build_member_risk_alert(risk: MemberRisk, *, now: datetime | None = None) -> Notification:
    """Alert about a member showing churn signals (Rule R1)."""
    now = now or datetime.now(UTC)
    level = _RISK_TO_LEVEL.get(risk.level, EscalationLevel.INFO)
    style = _style(level)
    action = {
        EscalationLevel.INFO: "Keep an eye on their next interactions.",
        EscalationLevel.ALTO: "Reach out personally within 24h and offer help or a mentor.",
        EscalationLevel.CRITICO: "Contact them today; consider a 1:1 call with the community lead.",
    }.get(level, "Monitor.")
    lines = [
        f"*Member:* {risk.author} (`{risk.member_key}`)",
        f"*Signals ({len(risk.signals)}/3):* " + ", ".join(risk.signals),
        f"*Risk score:* {risk.risk_score:.2f} — level *{risk.level}*",
        "*Evidence:*",
        *[f"  • {line}" for line in risk.evidence],
        f"*Last active:* {risk.last_active.date().isoformat()} ({risk.days_inactive:.0f} days ago)",
        f"*Action:* {action}",
    ]
    return Notification(
        kind="member_risk",
        level=level,
        title=f"{style.emoji} [{level.value}] Member at risk: {risk.author} ({len(risk.signals)}/3 signals)",
        body="\n".join(lines),
        target=style.target,
        audience=style.audience,
        message_ids=risk.message_ids,
        member_key=risk.member_key,
        created_at=now,
    )


def build_recurring_topic_alert(topic: RecurringTopic, *, now: datetime | None = None) -> Notification:
    """Alert about a doubt that keeps recurring (Rule R2). Always MEDIO: important, not urgent."""
    now = now or datetime.now(UTC)
    level = EscalationLevel.MEDIO
    style = _style(level)
    action_text = {
        "faq": "Write a FAQ entry (feed the linked messages to the FAQ generator).",
        "mentorship": "Assign a mentor to the member asking repeatedly.",
        "faq_and_mentorship": "Write a FAQ entry AND schedule a mentoring session on this topic.",
    }[topic.action]
    lines = [
        f"*Topic:* {topic.topic}",
        f"*Occurrences:* {topic.occurrences} message(s) from {topic.distinct_authors} member(s): "
        + ", ".join(topic.authors),
    ]
    if topic.technologies:
        lines.append("*Technologies:* " + ", ".join(topic.technologies))
    lines.append("*Examples:*")
    lines.extend(f'  • "{q}"' for q in topic.sample_questions)
    lines.append(f"*Why:* {topic.rationale}")
    lines.append(f"*Action:* {action_text}")
    return Notification(
        kind="recurring_topic",
        level=level,
        title=f"{style.emoji} [{level.value}] Recurring doubt ×{topic.occurrences}: {topic.topic} → {topic.action}",
        body="\n".join(lines),
        target=style.target,
        audience=style.audience,
        message_ids=topic.message_ids,
        created_at=now,
    )


def build_summary_notification(
    summary: str,
    *,
    level: EscalationLevel,
    message_ids: Sequence[str] = (),
    now: datetime | None = None,
) -> Notification:
    """Wrap the executive summary; its level is the highest level found in the batch."""
    now = now or datetime.now(UTC)
    style = _style(level)
    return Notification(
        kind="summary",
        level=level,
        title=f"{style.emoji} CommunityLab executive summary — highest level {level.value}",
        body=summary,
        target=style.target,
        audience=style.audience,
        message_ids=tuple(message_ids),
        created_at=now,
    )


def build_notifications(
    escalations: Iterable[Escalation],
    member_risks: Iterable[MemberRisk] = (),
    recurring_topics: Iterable[RecurringTopic] = (),
    *,
    min_level: EscalationLevel = EscalationLevel.MEDIO,
    now: datetime | None = None,
) -> list[Notification]:
    """Build alerts for everything at or above ``min_level``.

    INFO escalations are excluded by default because they belong in the
    digest, not in individual alerts. Member-risk alerts for ``watch`` members
    are likewise INFO and filtered out unless ``min_level`` is INFO.
    Results are sorted by urgency (CRÍTICO first).
    """
    now = now or datetime.now(UTC)
    notifications = [build_escalation_alert(e, now=now) for e in escalations]
    notifications += [build_member_risk_alert(r, now=now) for r in member_risks]
    notifications += [build_recurring_topic_alert(t, now=now) for t in recurring_topics]
    kept = [n for n in notifications if n.level >= min_level]
    return sorted(kept, key=lambda n: (-n.level.rank, n.kind, n.title))
