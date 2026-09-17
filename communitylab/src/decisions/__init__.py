"""Decision Engine: rules R1-R3 over consolidated analysis output."""

from src.decisions.config import DecisionConfig, EscalationConfig, MemberRiskConfig, RecurringTopicsConfig
from src.decisions.engine import RULE_DESCRIPTIONS, Action, ActionKind, DecisionEngine, DecisionReport, Rule, decide
from src.decisions.escalation import LEVELS_BY_URGENCY, Escalation, EscalationLevel, route, route_all, sla_for_level
from src.decisions.member_risk import (
    MemberRisk,
    RiskLevel,
    RiskSignal,
    assess_member,
    detect_member_risk,
    member_key_of,
)
from src.decisions.notifications import (
    LEVEL_STYLES,
    Notification,
    NotificationKind,
    NotificationTarget,
    build_escalation_alert,
    build_member_risk_alert,
    build_notifications,
    build_recurring_topic_alert,
    build_summary_notification,
)
from src.decisions.recurring_topics import RecurringAction, RecurringTopic, detect_recurring_topics, normalize_topic

__all__ = [
    "LEVELS_BY_URGENCY",
    "LEVEL_STYLES",
    "RULE_DESCRIPTIONS",
    "Action",
    "ActionKind",
    "DecisionConfig",
    "DecisionEngine",
    "DecisionReport",
    "Escalation",
    "EscalationConfig",
    "EscalationLevel",
    "MemberRisk",
    "MemberRiskConfig",
    "Notification",
    "NotificationKind",
    "NotificationTarget",
    "RecurringAction",
    "RecurringTopic",
    "RecurringTopicsConfig",
    "RiskLevel",
    "RiskSignal",
    "Rule",
    "assess_member",
    "build_escalation_alert",
    "build_member_risk_alert",
    "build_notifications",
    "build_recurring_topic_alert",
    "build_summary_notification",
    "decide",
    "detect_member_risk",
    "detect_recurring_topics",
    "member_key_of",
    "normalize_topic",
    "route",
    "route_all",
    "sla_for_level",
]
