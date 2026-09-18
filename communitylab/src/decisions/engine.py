"""Decision Engine entry point.

Applies the three business rules to a batch of consolidated messages and
turns the results into a prioritised action list plus an executive summary:

    R1  Member at risk        -> member_risk.detect_member_risk
    R2  Recurring doubts      -> recurring_topics.detect_recurring_topics
    R3  Escalation routing    -> escalation.route_all (uses R1 output)

Typical usage::

    report = DecisionEngine().run(enriched_messages)
    print(report.executive_summary)
    for action in report.actions:
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.consolidator import EnrichedMessage
from src.decisions.config import DecisionConfig
from src.decisions.escalation import LEVELS_BY_URGENCY, Escalation, EscalationLevel, route_all
from src.decisions.member_risk import MemberRisk, detect_member_risk
from src.decisions.notifications import Notification, build_notifications, build_summary_notification
from src.decisions.recurring_topics import RecurringTopic, detect_recurring_topics

logger = logging.getLogger(__name__)

Rule = Literal["R1", "R2", "R3"]
ActionKind = Literal["respond", "escalate", "reach_out", "create_faq", "assign_mentor", "monitor", "amplify"]

RULE_DESCRIPTIONS: dict[Rule, str] = {
    "R1": "Member at risk: negative sentiment, frustration or inactivity (>= 2 of 3 signals)",
    "R2": "Recurring doubt: the same question appears >= 3 times -> FAQ / mentorship",
    "R3": "Escalation: route each interaction to INFO / MEDIO / ALTO / CRÍTICO",
}


class Action(BaseModel):
    """One concrete thing the team should do.

    Attributes:
        rule: Which rule produced it.
        kind: Type of action.
        level: Escalation level (drives ordering and SLA).
        title: Short imperative sentence.
        detail: Supporting context.
        owner: Suggested owner role.
        due_hours: Hours until the action is overdue (``None`` = no deadline).
        message_ids: Related messages.
        member_key: Related member, if any.
    """

    model_config = ConfigDict(frozen=True)

    rule: Rule
    kind: ActionKind
    level: EscalationLevel
    title: str
    detail: str = ""
    owner: str = "community team"
    due_hours: int | None = Field(default=None, ge=0)
    message_ids: tuple[str, ...] = ()
    member_key: str | None = None


class DecisionReport(BaseModel):
    """Everything the Decision Engine produced for one batch."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    reference_time: datetime
    message_count: int = Field(ge=0)
    escalations: tuple[Escalation, ...] = ()
    member_risks: tuple[MemberRisk, ...] = ()
    recurring_topics: tuple[RecurringTopic, ...] = ()
    actions: tuple[Action, ...] = ()
    notifications: tuple[Notification, ...] = ()
    level_counts: dict[str, int] = Field(default_factory=dict)
    highest_level: EscalationLevel = EscalationLevel.INFO
    executive_summary: str = ""

    def escalations_at(self, level: EscalationLevel) -> tuple[Escalation, ...]:
        """Escalations with exactly ``level``."""
        return tuple(e for e in self.escalations if e.level == level)

    @property
    def urgent_actions(self) -> tuple[Action, ...]:
        """Actions at ALTO or CRÍTICO."""
        return tuple(a for a in self.actions if a.level >= EscalationLevel.ALTO)


class DecisionEngine:
    """Coordinates R1-R3 and produces a :class:`DecisionReport`.

    Args:
        config: Thresholds for all submodules.
        notify_from: Lowest level that gets an individual notification.
    """

    def __init__(
        self, config: DecisionConfig | None = None, *, notify_from: EscalationLevel = EscalationLevel.MEDIO
    ) -> None:
        self.config = config or DecisionConfig()
        self.notify_from = notify_from

    # ------------------------------------------------------------------ rules
    def apply_r1(self, enriched: list[EnrichedMessage], *, reference_time: datetime) -> list[MemberRisk]:
        """R1 — members with at least one risk signal (``watch`` included for context)."""
        return detect_member_risk(enriched, reference_time=reference_time, config=self.config.member_risk)

    def apply_r2(self, enriched: list[EnrichedMessage]) -> list[RecurringTopic]:
        """R2 — doubts occurring at least ``min_occurrences`` times."""
        return detect_recurring_topics(enriched, config=self.config.recurring_topics)

    def apply_r3(self, enriched: list[EnrichedMessage], member_risks: list[MemberRisk]) -> list[Escalation]:
        """R3 — route every message, taking the author's R1 level into account."""
        return route_all(enriched, member_risks, config=self.config.escalation)

    # ---------------------------------------------------------------- actions
    def _actions_from_risks(self, risks: list[MemberRisk]) -> list[Action]:
        actions: list[Action] = []
        for risk in risks:
            # Only 2+ signals produce an action; a single signal is context for
            # R3 and the summary but does not justify contacting the member.
            if not risk.is_at_risk:
                continue
            critical = risk.level == "high"
            actions.append(
                Action(
                    rule="R1",
                    kind="reach_out",
                    level=EscalationLevel.CRITICO if critical else EscalationLevel.ALTO,
                    title=f"Reach out to {risk.author} ({len(risk.signals)}/3 risk signals)",
                    detail="; ".join(risk.evidence),
                    owner="community lead" if critical else "mentor",
                    due_hours=self.config.escalation.sla_hours_critico
                    if critical
                    else self.config.escalation.sla_hours_alto,
                    message_ids=risk.message_ids,
                    member_key=risk.member_key,
                )
            )
        return actions

    def _actions_from_topics(self, topics: list[RecurringTopic]) -> list[Action]:
        actions: list[Action] = []
        for topic in topics:
            if topic.action in ("faq", "faq_and_mentorship"):
                actions.append(
                    Action(
                        rule="R2",
                        kind="create_faq",
                        level=EscalationLevel.MEDIO,
                        title=f'Create FAQ entry for "{topic.topic}" (asked {topic.occurrences}×)',
                        detail=topic.rationale,
                        owner="content team",
                        due_hours=self.config.escalation.sla_hours_medio,
                        message_ids=topic.message_ids,
                    )
                )
            if topic.action in ("mentorship", "faq_and_mentorship"):
                actions.append(
                    Action(
                        rule="R2",
                        kind="assign_mentor",
                        level=EscalationLevel.MEDIO,
                        title=f'Assign a mentor for "{topic.topic}" ({", ".join(topic.authors)})',
                        detail=topic.rationale,
                        owner="mentor coordinator",
                        due_hours=self.config.escalation.sla_hours_medio,
                        message_ids=topic.message_ids,
                    )
                )
        return actions

    def _actions_from_escalations(self, escalations: list[Escalation]) -> list[Action]:
        actions: list[Action] = []
        for esc in escalations:
            if esc.level == EscalationLevel.INFO:
                continue  # INFO lives in the digest only
            if esc.level == EscalationLevel.CRITICO:
                kind: ActionKind = "escalate"
                owner = "community lead"
            elif esc.requires_response or (esc.sentiment_score is not None and esc.sentiment_score < 0):
                kind, owner = "respond", "mentor"
            elif esc.sentiment_score is not None and esc.sentiment_score > 0 and esc.level == EscalationLevel.ALTO:
                # High-relevance positive messages (success stories) are content
                # opportunities rather than problems.
                kind, owner = "amplify", "content team"
            else:
                kind, owner = "monitor", "community team"
            verb = kind.replace("_", " ").capitalize()
            actions.append(
                Action(
                    rule="R3",
                    kind=kind,
                    level=esc.level,
                    title=f"[{esc.level.value}] {verb}: {esc.author} — {esc.excerpt[:80]}",
                    detail="; ".join(esc.reasons),
                    owner=owner,
                    due_hours=esc.sla_hours,
                    message_ids=(esc.message_id,),
                    member_key=esc.member_key,
                )
            )
        return actions

    # ---------------------------------------------------------------- summary
    @staticmethod
    def _executive_summary(
        *,
        message_count: int,
        level_counts: dict[str, int],
        escalations: list[Escalation],
        risks: list[MemberRisk],
        topics: list[RecurringTopic],
        actions: list[Action],
        reference_time: datetime,
    ) -> str:
        lines = [
            f"CommunityLab — Decision Engine summary ({reference_time.strftime('%Y-%m-%d %H:%M UTC')})",
            f"Analysed {message_count} message(s). Escalation levels: "
            + ", ".join(f"{lvl.value} {level_counts.get(lvl.value, 0)}" for lvl in LEVELS_BY_URGENCY)
            + ".",
            "",
        ]
        urgent = [e for e in escalations if e.is_urgent]
        lines.append(f"R3 · Urgent interactions: {len(urgent)}")
        for esc in urgent[:5]:
            sla = f" (SLA {esc.sla_hours}h)" if esc.sla_hours is not None else ""
            lines.append(f'  - [{esc.level.value}] {esc.author}: "{esc.excerpt[:90]}"{sla}')
        if len(urgent) > 5:
            lines.append(f"  - … and {len(urgent) - 5} more")

        at_risk = [r for r in risks if r.is_at_risk]
        watch = [r for r in risks if not r.is_at_risk]
        lines.append("")
        lines.append(f"R1 · Members at risk: {len(at_risk)} (plus {len(watch)} on watch)")
        for risk in at_risk[:5]:
            lines.append(f"  - {risk.author}: {risk.level} — {', '.join(risk.signals)} (score {risk.risk_score:.2f})")

        lines.append("")
        lines.append(f"R2 · Recurring doubts: {len(topics)}")
        for topic in topics[:5]:
            lines.append(
                f'  - "{topic.topic}" ×{topic.occurrences} from {topic.distinct_authors} member(s) → {topic.action}'
            )

        lines.append("")
        lines.append(f"Required actions: {len(actions)}")
        for action in actions[:10]:
            due = f" · due in {action.due_hours}h" if action.due_hours is not None else ""
            lines.append(f"  {action.level.value:<8} {action.owner:<18} {action.title}{due}")
        if len(actions) > 10:
            lines.append(f"  … and {len(actions) - 10} more")
        if not actions:
            lines.append("  Nothing urgent. Community is healthy.")
        return "\n".join(lines)

    # -------------------------------------------------------------------- run
    def run(self, enriched: Iterable[EnrichedMessage], *, reference_time: datetime | None = None) -> DecisionReport:
        """Apply R1-R3 to ``enriched`` and build the report.

        Args:
            enriched: Consolidated analysis output for a batch of messages.
            reference_time: "Now" for inactivity and SLA computations (default: current UTC time).
        """
        items = list(enriched)
        reference_time = reference_time or datetime.now(UTC)

        member_risks = self.apply_r1(items, reference_time=reference_time)
        recurring = self.apply_r2(items)
        escalations = self.apply_r3(items, member_risks)

        level_counts = {lvl.value: 0 for lvl in LEVELS_BY_URGENCY}
        for esc in escalations:
            level_counts[esc.level.value] += 1
        highest = max((e.level for e in escalations), default=EscalationLevel.INFO)

        # Order: R3 critical items first, then R1 outreach, then the rest by level.
        actions = (
            self._actions_from_escalations(escalations)
            + self._actions_from_risks(member_risks)
            + self._actions_from_topics(recurring)
        )
        actions.sort(key=lambda a: (-a.level.rank, a.due_hours if a.due_hours is not None else 10**6, a.rule, a.title))

        summary = self._executive_summary(
            message_count=len(items),
            level_counts=level_counts,
            escalations=escalations,
            risks=member_risks,
            topics=recurring,
            actions=actions,
            reference_time=reference_time,
        )
        notifications = build_notifications(
            escalations, member_risks, recurring, min_level=self.notify_from, now=reference_time
        )
        notifications.append(
            build_summary_notification(
                summary,
                level=highest,
                message_ids=[e.message_id for e in escalations if e.is_urgent],
                now=reference_time,
            )
        )
        logger.info(
            "Decision engine: %d messages, %d actions, %d at-risk members, %d recurring topics, highest level %s",
            len(items),
            len(actions),
            sum(r.is_at_risk for r in member_risks),
            len(recurring),
            highest.value,
        )
        return DecisionReport(
            generated_at=datetime.now(UTC),
            reference_time=reference_time,
            message_count=len(items),
            escalations=tuple(escalations),
            member_risks=tuple(member_risks),
            recurring_topics=tuple(recurring),
            actions=tuple(actions),
            notifications=tuple(notifications),
            level_counts=level_counts,
            highest_level=highest,
            executive_summary=summary,
        )


def decide(
    enriched: Iterable[EnrichedMessage], *, config: DecisionConfig | None = None, reference_time: datetime | None = None
) -> DecisionReport:
    """One-shot helper: ``DecisionEngine(config).run(enriched)``."""
    return DecisionEngine(config).run(enriched, reference_time=reference_time)
