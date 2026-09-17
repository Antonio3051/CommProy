"""Tunable thresholds for the Decision Engine.

All business thresholds live here so they can be reviewed (and overridden)
in one place instead of being scattered as magic numbers. Every value is
documented with the reasoning behind the default.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Frustration vocabulary (English + Spanish, lowercase). Matched as substrings
# of the lowercased message content, so "frustrat" also catches
# "frustrated"/"frustrating"/"frustración". Keep entries short and specific:
# a false positive here only adds ONE of the three risk signals, but too many
# generic words ("problem", "error") would flag every technical question.
DEFAULT_FRUSTRATION_KEYWORDS: tuple[str, ...] = (
    # English
    "frustrat",
    "giving up",
    "give up",
    "i quit",
    "waste of time",
    "wasted",
    "doesn't work",
    "does not work",
    "not working",
    "still broken",
    "nobody answer",
    "no one answer",
    "no response",
    "stuck for",
    "fed up",
    "useless",
    "disappointed",
    "cancel my",
    "refund",
    # Spanish
    "frustra",
    "me rindo",
    "rendirme",
    "abandon",
    "dejar el curso",
    "no funciona",
    "sigue sin funcionar",
    "nadie responde",
    "nadie me ayuda",
    "pérdida de tiempo",
    "perdiendo el tiempo",
    "harto",
    "harta",
    "no entiendo nada",
    "decepcion",
    "reembolso",
)

# Emotions emitted by the sentiment analyzer that indicate frustration even
# when the member does not use one of the keywords above.
DEFAULT_FRUSTRATION_EMOTIONS: tuple[str, ...] = ("frustration", "anger", "disappointment", "despair", "annoyance")


class MemberRiskConfig(BaseModel):
    """Thresholds for Rule R1 (member at risk)."""

    model_config = ConfigDict(frozen=True)

    # Signal 1 — negative sentiment. A member is "negative" when the average
    # sentiment score of their messages is at or below this value (scores go
    # from -1 to 1; -0.3 is clearly below "neutral noise" of ±0.2) OR when they
    # have at least `min_negative_messages` messages labelled negative.
    negative_mean_score: float = Field(default=-0.3, ge=-1.0, le=0.0)
    min_negative_messages: int = Field(default=2, ge=1)

    # Signal 2 — frustration vocabulary / emotions. One match is enough: a
    # member who writes "I'm about to give up" once deserves attention.
    frustration_keywords: tuple[str, ...] = DEFAULT_FRUSTRATION_KEYWORDS
    frustration_emotions: tuple[str, ...] = DEFAULT_FRUSTRATION_EMOTIONS

    # Signal 3 — inactivity. 14 days without a message is roughly two weekly
    # cohort cycles; shorter windows flag members who are simply busy.
    inactivity_days: int = Field(default=14, ge=1)
    # A member must have been active at least once before we call them
    # "inactive"; one-off visitors are not churn.
    min_messages_for_inactivity: int = Field(default=1, ge=1)

    # Weights used to turn signals into a 0..1 risk score (they sum to 1).
    # Negative sentiment and frustration are stronger churn predictors than
    # inactivity alone, hence the asymmetric weights.
    weight_negative_sentiment: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_frustration: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_inactivity: float = Field(default=0.25, ge=0.0, le=1.0)


class RecurringTopicsConfig(BaseModel):
    """Thresholds for Rule R2 (recurring doubts)."""

    model_config = ConfigDict(frozen=True)

    # A doubt is "recurring" when it appears in at least this many distinct
    # messages. Three is the classic "once is chance, twice is coincidence,
    # three times is a pattern" threshold requested by the business.
    min_occurrences: int = Field(default=3, ge=2)
    # When at least this many DIFFERENT members ask the same thing, the doubt
    # is widespread and worth a FAQ entry. Below it (one person asking three
    # times) the right response is a mentor, not documentation.
    min_authors_for_faq: int = Field(default=2, ge=1)
    # Above this many occurrences the topic is a systemic gap: do both.
    occurrences_for_both_actions: int = Field(default=5, ge=2)
    # Message categories treated as "doubts". Complaints are included because
    # a repeated complaint about the same thing is a documentation/mentoring
    # gap as much as a repeated question.
    doubt_categories: tuple[str, ...] = ("technical_question", "complaint")


class EscalationConfig(BaseModel):
    """Thresholds for Rule R3 (escalation routing)."""

    model_config = ConfigDict(frozen=True)

    # Relevance score (0..1) boundaries. They mirror the tiers produced by the
    # relevance analyzer (medium >= 0.4, high >= 0.6, critical >= 0.85) so the
    # routing is consistent with what the LLM was asked to score.
    relevance_medio: float = Field(default=0.4, ge=0.0, le=1.0)
    relevance_alto: float = Field(default=0.6, ge=0.0, le=1.0)
    relevance_critico: float = Field(default=0.85, ge=0.0, le=1.0)

    # Sentiment score (-1..1) boundaries. -0.3 is "clearly negative";
    # -0.7 is "very negative" (anger, threats to leave, public complaints).
    sentiment_negative: float = Field(default=-0.3, ge=-1.0, le=0.0)
    sentiment_very_negative: float = Field(default=-0.7, ge=-1.0, le=0.0)

    # Response SLAs per level, in hours. Used by notifications and the
    # executive summary to state *when* something must happen.
    sla_hours_critico: int = Field(default=2, ge=0)
    sla_hours_alto: int = Field(default=24, ge=0)
    sla_hours_medio: int = Field(default=72, ge=0)


class DecisionConfig(BaseModel):
    """Aggregate configuration for the whole Decision Engine."""

    model_config = ConfigDict(frozen=True)

    member_risk: MemberRiskConfig = MemberRiskConfig()
    recurring_topics: RecurringTopicsConfig = RecurringTopicsConfig()
    escalation: EscalationConfig = EscalationConfig()
