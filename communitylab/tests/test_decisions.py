"""Tests for the Decision Engine (src/decisions)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from src.analysis import EnrichedMessage, RelevanceResult, SentimentResult, ThemesResult
from src.decisions import (
    DecisionConfig,
    DecisionEngine,
    EscalationConfig,
    EscalationLevel,
    MemberRiskConfig,
    RecurringTopicsConfig,
    assess_member,
    build_escalation_alert,
    build_member_risk_alert,
    build_notifications,
    build_recurring_topic_alert,
    decide,
    detect_member_risk,
    detect_recurring_topics,
    normalize_topic,
    route,
    route_all,
    sla_for_level,
)
from src.ingest.validator import CommunityMessage

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def enriched(
    message_id: str,
    author: str,
    content: str = "hello",
    *,
    days_ago: float = 0,
    label: str = "neutral",
    score: float = 0.0,
    emotions: list[str] | None = None,
    category: str = "community_chatter",
    topics: list[str] | None = None,
    keywords: list[str] | None = None,
    relevance: float | None = 0.2,
    requires_response: bool = False,
    is_actionable: bool = False,
    author_id: str | None = None,
    with_sentiment: bool = True,
    with_themes: bool = True,
) -> EnrichedMessage:
    message = CommunityMessage(
        message_id=message_id,
        source="discord",
        channel="help",
        author=author,
        author_id=author_id or author,
        content=content,
        timestamp=NOW - timedelta(days=days_ago),
    )
    return EnrichedMessage(
        message=message,
        sentiment=SentimentResult(label=label, score=score, emotions=emotions or []) if with_sentiment else None,
        themes=ThemesResult(primary_category=category, topics=topics or [], keywords=keywords or [])
        if with_themes
        else None,
        relevance=RelevanceResult(score=relevance, requires_response=requires_response, is_actionable=is_actionable)
        if relevance is not None
        else None,
    )


# --------------------------------------------------------------------- config


def test_default_config_is_frozen_and_documented() -> None:
    config = DecisionConfig()
    assert config.recurring_topics.min_occurrences == 3
    assert config.escalation.sla_hours_critico < config.escalation.sla_hours_alto < config.escalation.sla_hours_medio
    with pytest.raises(ValidationError):
        config.escalation.relevance_alto = 0.1  # type: ignore[misc]


def test_config_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        MemberRiskConfig(negative_mean_score=0.5)
    with pytest.raises(ValueError):
        RecurringTopicsConfig(min_occurrences=1)


# ------------------------------------------------------------------ R1: risk


class TestMemberRisk:
    def test_no_signals_not_reported(self) -> None:
        items = [enriched("1", "ana", "Great course!", label="positive", score=0.8)]
        assert detect_member_risk(items, reference_time=NOW) == []

    def test_negative_sentiment_by_mean(self) -> None:
        risk = assess_member(
            [enriched("1", "bob", "meh", label="negative", score=-0.5)], reference_time=NOW, config=MemberRiskConfig()
        )
        assert risk.signals == ("negative_sentiment",)
        assert risk.level == "watch"
        assert risk.risk_score == pytest.approx(0.4)
        assert risk.mean_sentiment == -0.5
        assert risk.message_ids == ("1",)

    def test_negative_sentiment_by_count_even_if_mean_is_mild(self) -> None:
        items = [
            enriched("1", "bob", label="negative", score=-0.2),
            enriched("2", "bob", label="negative", score=-0.2),
            enriched("3", "bob", label="positive", score=0.9),
        ]
        (risk,) = detect_member_risk(items, reference_time=NOW)
        assert "negative_sentiment" in risk.signals
        assert risk.negative_count == 2

    @pytest.mark.parametrize("content", ["I'm about to give up", "esto NO FUNCIONA", "me rindo con esto"])
    def test_frustration_keyword(self, content: str) -> None:
        (risk,) = detect_member_risk([enriched("1", "bob", content)], reference_time=NOW)
        assert risk.signals == ("frustration",)
        assert "keyword" in risk.evidence[0]

    def test_frustration_emotion_from_analyzer(self) -> None:
        (risk,) = detect_member_risk([enriched("1", "bob", "hmm", emotions=["frustration"])], reference_time=NOW)
        assert risk.signals == ("frustration",)
        assert 'emotion "frustration"' in risk.evidence[0]

    def test_inactivity(self) -> None:
        (risk,) = detect_member_risk([enriched("1", "bob", days_ago=30)], reference_time=NOW)
        assert risk.signals == ("inactivity",)
        assert risk.days_inactive == 30
        assert detect_member_risk([enriched("1", "bob", days_ago=13.9)], reference_time=NOW) == []

    def test_three_signals_is_high(self) -> None:
        items = [
            enriched("1", "bob", "I'm frustrated", days_ago=20, label="negative", score=-0.8),
            enriched("2", "bob", "still broken", days_ago=15, label="negative", score=-0.6),
        ]
        (risk,) = detect_member_risk(items, reference_time=NOW)
        assert risk.signals == ("negative_sentiment", "frustration", "inactivity")
        assert risk.level == "high"
        assert risk.risk_score == 1.0
        assert risk.is_at_risk
        assert risk.last_active == NOW - timedelta(days=15)

    def test_groups_by_author_id_and_sorts_by_score(self) -> None:
        items = [
            enriched("1", "Bob", "give up", author_id="u1"),
            enriched("2", "bob", "x", author_id="u1", label="negative", score=-0.9),
            enriched("3", "cid", "x", days_ago=40),
        ]
        risks = detect_member_risk(items, reference_time=NOW)
        assert [r.member_key for r in risks] == ["u1", "cid"]
        assert risks[0].message_count == 2

    def test_include_watch_false_filters_single_signal(self) -> None:
        items = [enriched("1", "bob", days_ago=40), enriched("2", "cid", "give up", label="negative", score=-0.9)]
        assert [r.author for r in detect_member_risk(items, reference_time=NOW, include_watch=False)] == ["cid"]

    def test_missing_sentiment_is_ignored(self) -> None:
        (risk,) = detect_member_risk([enriched("1", "bob", "give up", with_sentiment=False)], reference_time=NOW)
        assert risk.mean_sentiment is None
        assert risk.signals == ("frustration",)

    def test_custom_thresholds(self) -> None:
        config = MemberRiskConfig(inactivity_days=3)
        (risk,) = detect_member_risk([enriched("1", "bob", days_ago=5)], reference_time=NOW, config=config)
        assert risk.signals == ("inactivity",)

    def test_assess_member_requires_messages(self) -> None:
        with pytest.raises(ValueError):
            assess_member([], reference_time=NOW, config=MemberRiskConfig())


# ---------------------------------------------------------------- R2: topics


class TestRecurringTopics:
    def test_normalize_topic(self) -> None:
        assert normalize_topic("  Entornos Virtuales (venv) ") == "entornos virtuales venv"
        assert normalize_topic("Virtual-Environments") == normalize_topic("virtual environments")

    def test_below_threshold_not_flagged(self) -> None:
        items = [enriched(str(i), f"u{i}", "?", category="technical_question", topics=["venv"]) for i in range(2)]
        assert detect_recurring_topics(items) == []

    def test_three_authors_is_faq(self) -> None:
        items = [
            enriched("1", "ana", "How do I use venv?", category="technical_question", topics=["Virtual environments"]),
            enriched("2", "bob", "venv?", category="technical_question", topics=["virtual environments"]),
            enriched("3", "cid", "venv!", category="technical_question", topics=["Virtual environments", "pip"]),
        ]
        (topic,) = detect_recurring_topics(items)
        assert topic.topic == "Virtual environments"
        assert topic.occurrences == 3
        assert topic.distinct_authors == 3
        assert topic.action == "faq"
        assert topic.message_ids == ("1", "2", "3")
        assert topic.sample_questions[0] == "How do I use venv?"

    def test_single_author_is_mentorship(self) -> None:
        items = [enriched(str(i), "bob", "?", category="technical_question", topics=["venv"]) for i in range(3)]
        (topic,) = detect_recurring_topics(items)
        assert topic.action == "mentorship"
        assert topic.authors == ("bob",)

    def test_many_occurrences_and_authors_is_both(self) -> None:
        items = [enriched(str(i), f"u{i % 2}", "?", category="technical_question", topics=["venv"]) for i in range(5)]
        (topic,) = detect_recurring_topics(items)
        assert topic.action == "faq_and_mentorship"

    def test_non_doubt_categories_ignored(self) -> None:
        items = [enriched(str(i), f"u{i}", "yay", category="success_story", topics=["venv"]) for i in range(4)]
        assert detect_recurring_topics(items) == []

    def test_message_counts_once_per_topic(self) -> None:
        items = [
            enriched("1", "ana", "?", category="technical_question", topics=["venv", "VENV", "Venv"]),
            enriched("2", "bob", "?", category="technical_question", topics=["venv"]),
        ]
        assert detect_recurring_topics(items) == []

    def test_falls_back_to_keywords_and_question_mark(self) -> None:
        items = [
            enriched("1", "ana", "?", category="technical_question", keywords=["docker"]),
            enriched("2", "bob", "?", category="technical_question", keywords=["Docker"]),
            enriched("3", "cid", "docker?", with_themes=False),
        ]
        # Third message has no themes -> falls back to ingestion category (None) -> no topic label.
        assert detect_recurring_topics(items) == []
        items[2] = enriched("3", "cid", "?", category="technical_question", keywords=["docker"])
        (topic,) = detect_recurring_topics(items)
        assert topic.topic_key == "docker"

    def test_sorted_by_occurrences(self) -> None:
        items = [enriched(str(i), f"u{i}", "?", category="technical_question", topics=["a"]) for i in range(3)]
        items += [enriched(f"b{i}", f"u{i}", "?", category="technical_question", topics=["b"]) for i in range(4)]
        assert [t.topic_key for t in detect_recurring_topics(items)] == ["b", "a"]


# ------------------------------------------------------------ R3: escalation


class TestEscalation:
    def test_levels_are_ordered(self) -> None:
        assert EscalationLevel.INFO < EscalationLevel.MEDIO < EscalationLevel.ALTO < EscalationLevel.CRITICO
        assert EscalationLevel.CRITICO.value == "CRÍTICO"
        assert max([EscalationLevel.MEDIO, EscalationLevel.ALTO]) is EscalationLevel.ALTO

    def test_sla_per_level(self) -> None:
        config = EscalationConfig()
        assert sla_for_level(EscalationLevel.CRITICO, config) == 2
        assert sla_for_level(EscalationLevel.ALTO, config) == 24
        assert sla_for_level(EscalationLevel.MEDIO, config) == 72
        assert sla_for_level(EscalationLevel.INFO, config) is None

    def test_noise_is_info(self) -> None:
        esc = route(enriched("1", "ana", "lol", relevance=0.05))
        assert esc.level is EscalationLevel.INFO
        assert esc.sla_hours is None
        assert not esc.is_urgent

    def test_medium_relevance_is_medio(self) -> None:
        esc = route(enriched("1", "ana", "?", relevance=0.5))
        assert esc.level is EscalationLevel.MEDIO
        assert esc.sla_hours == 72

    def test_requires_response_lifts_to_medio(self) -> None:
        esc = route(enriched("1", "ana", "?", relevance=0.1, requires_response=True))
        assert esc.level is EscalationLevel.MEDIO
        assert "requires a response" in esc.reasons

    def test_high_relevance_is_alto(self) -> None:
        esc = route(enriched("1", "dan", "shipped!", label="positive", score=0.9, relevance=0.7))
        assert esc.level is EscalationLevel.ALTO
        assert esc.sla_hours == 24

    def test_critical_relevance_is_critico(self) -> None:
        esc = route(enriched("1", "ana", "prod is down", relevance=0.9))
        assert esc.level is EscalationLevel.CRITICO
        assert esc.sla_hours == 2

    def test_very_negative_high_relevance_is_critico(self) -> None:
        esc = route(enriched("1", "bob", "awful", label="negative", score=-0.8, relevance=0.65))
        assert esc.level is EscalationLevel.CRITICO
        assert any("very negative" in r for r in esc.reasons)

    def test_two_alto_triggers_become_critico(self) -> None:
        # Negative+actionable (A2) and unanswered complaint (A4) with medium relevance.
        esc = route(
            enriched(
                "1",
                "bob",
                "this is bad",
                label="negative",
                score=-0.5,
                category="complaint",
                relevance=0.5,
                requires_response=True,
                is_actionable=True,
            )
        )
        assert esc.level is EscalationLevel.CRITICO
        assert esc.reasons[0].startswith("multiple ALTO triggers")

    def test_member_risk_raises_level(self) -> None:
        items = [
            enriched("1", "bob", "I give up", days_ago=20, label="negative", score=-0.8),
            enriched("2", "bob", "hi", days_ago=15, relevance=0.1),
        ]
        risks = detect_member_risk(items, reference_time=NOW)
        assert risks[0].level == "high"
        plain = route(items[1])
        with_risk = route(items[1], risks[0])
        assert plain.level is EscalationLevel.INFO
        assert with_risk.level is EscalationLevel.ALTO
        assert with_risk.member_risk_level == "high"

    def test_watch_member_gets_medio_floor(self) -> None:
        (risk,) = detect_member_risk([enriched("1", "bob", days_ago=40, relevance=0.05)], reference_time=NOW)
        esc = route(enriched("1", "bob", days_ago=40, relevance=0.05), risk)
        assert esc.level is EscalationLevel.MEDIO

    def test_missing_analyses_still_route(self) -> None:
        esc = route(enriched("1", "ana", "?", relevance=None, with_sentiment=False, with_themes=False))
        assert esc.level is EscalationLevel.INFO
        assert esc.relevance_score is None and esc.sentiment_score is None

    def test_custom_thresholds(self) -> None:
        esc = route(enriched("1", "ana", "?", relevance=0.5), config=EscalationConfig(relevance_critico=0.5))
        assert esc.level is EscalationLevel.CRITICO

    def test_route_all_sorted_by_urgency(self) -> None:
        items = [
            enriched("low", "a", relevance=0.05),
            enriched("crit", "b", relevance=0.95),
            enriched("mid", "c", relevance=0.5),
        ]
        assert [e.message_id for e in route_all(items)] == ["crit", "mid", "low"]


# ---------------------------------------------------------------- notifications


class TestNotifications:
    def test_escalation_alert_per_level(self) -> None:
        crit = build_escalation_alert(
            route(enriched("1", "bob", "down", relevance=0.95, requires_response=True)), now=NOW
        )
        assert crit.level is EscalationLevel.CRITICO
        assert crit.target == "direct_message"
        assert crit.title.startswith("🔴 [CRÍTICO] bob")
        assert "SLA 2h" in crit.title
        assert crit.due_by == NOW + timedelta(hours=2)
        assert "*Status:* still unanswered" in crit.body
        assert crit.message_ids == ("1",)

        info = build_escalation_alert(route(enriched("2", "ana", "lol", relevance=0.05)), now=NOW)
        assert info.target == "digest"
        assert info.due_by is None

    def test_member_risk_alert_levels(self) -> None:
        (watch,) = detect_member_risk([enriched("1", "bob", days_ago=40)], reference_time=NOW)
        assert build_member_risk_alert(watch, now=NOW).level is EscalationLevel.INFO
        (high,) = detect_member_risk(
            [enriched("1", "bob", "give up", days_ago=40, label="negative", score=-0.9)], reference_time=NOW
        )
        alert = build_member_risk_alert(high, now=NOW)
        assert alert.level is EscalationLevel.CRITICO
        assert "3/3" in alert.title
        assert alert.member_key == "bob"
        assert "threshold 14 days" in alert.body

    def test_recurring_topic_alert(self) -> None:
        items = [enriched(str(i), f"u{i}", "venv?", category="technical_question", topics=["venv"]) for i in range(3)]
        (topic,) = detect_recurring_topics(items)
        alert = build_recurring_topic_alert(topic, now=NOW)
        assert alert.level is EscalationLevel.MEDIO
        assert "×3" in alert.title and "faq" in alert.title
        assert "*Action:* Write a FAQ entry" in alert.body
        assert alert.message_ids == ("0", "1", "2")

    def test_build_notifications_filters_and_sorts(self) -> None:
        escalations = route_all([enriched("a", "x", relevance=0.05), enriched("b", "y", relevance=0.95)])
        (watch,) = detect_member_risk([enriched("c", "z", days_ago=40)], reference_time=NOW)
        default = build_notifications(escalations, [watch], now=NOW)
        assert [n.level for n in default] == [EscalationLevel.CRITICO]
        everything = build_notifications(escalations, [watch], min_level=EscalationLevel.INFO, now=NOW)
        assert len(everything) == 3
        assert everything[0].level is EscalationLevel.CRITICO


# --------------------------------------------------------------------- engine


@pytest.fixture
def batch() -> list[EnrichedMessage]:
    return [
        enriched(
            "1",
            "ana",
            "How do I fix ModuleNotFoundError with venv?",
            days_ago=1,
            category="technical_question",
            topics=["Virtual environments"],
            relevance=0.5,
            requires_response=True,
        ),
        enriched(
            "2",
            "bob",
            "venv again not working, about to give up",
            days_ago=20,
            label="negative",
            score=-0.8,
            category="complaint",
            topics=["virtual environments"],
            relevance=0.7,
            requires_response=True,
            is_actionable=True,
        ),
        enriched(
            "3",
            "cid",
            "Virtual Environments confuse me",
            days_ago=2,
            category="technical_question",
            topics=["Virtual Environments"],
            relevance=0.45,
            requires_response=True,
        ),
        enriched(
            "4",
            "dan",
            "Shipped my first pipeline!",
            label="positive",
            score=0.9,
            category="success_story",
            relevance=0.7,
        ),
        enriched("5", "eve", "lol", relevance=0.05),
        enriched(
            "6",
            "bob",
            "still broken, nobody answers",
            days_ago=21,
            label="negative",
            score=-0.7,
            category="complaint",
            topics=["venv"],
            relevance=0.9,
            requires_response=True,
            is_actionable=True,
        ),
    ]


class TestDecisionEngine:
    def test_run_applies_all_rules(self, batch: list[EnrichedMessage]) -> None:
        report = DecisionEngine().run(batch, reference_time=NOW)
        assert report.message_count == 6
        assert report.level_counts == {"CRÍTICO": 2, "ALTO": 1, "MEDIO": 2, "INFO": 1}
        assert report.highest_level is EscalationLevel.CRITICO
        assert [r.author for r in report.member_risks] == ["bob"]
        assert report.member_risks[0].level == "high"
        assert [t.action for t in report.recurring_topics] == ["faq"]
        assert len(report.escalations_at(EscalationLevel.CRITICO)) == 2

    def test_actions_are_prioritised(self, batch: list[EnrichedMessage]) -> None:
        report = DecisionEngine().run(batch, reference_time=NOW)
        levels = [a.level.rank for a in report.actions]
        assert levels == sorted(levels, reverse=True)
        kinds = {a.kind for a in report.actions}
        assert {"escalate", "reach_out", "amplify", "create_faq", "respond"} <= kinds
        reach_out = next(a for a in report.actions if a.kind == "reach_out")
        assert reach_out.rule == "R1" and reach_out.member_key == "bob" and reach_out.due_hours == 2
        assert all(a.level >= EscalationLevel.ALTO for a in report.urgent_actions)
        assert all(a.level is not EscalationLevel.INFO for a in report.actions)

    def test_executive_summary_contents(self, batch: list[EnrichedMessage]) -> None:
        summary = DecisionEngine().run(batch, reference_time=NOW).executive_summary
        assert "Analysed 6 message(s)" in summary
        assert "CRÍTICO 2, ALTO 1, MEDIO 2, INFO 1" in summary
        assert "R1 · Members at risk: 1" in summary
        assert "R2 · Recurring doubts: 1" in summary and "Virtual environments" in summary
        assert "Required actions:" in summary

    def test_notifications_include_summary_last(self, batch: list[EnrichedMessage]) -> None:
        report = DecisionEngine().run(batch, reference_time=NOW)
        assert report.notifications[-1].kind == "summary"
        assert report.notifications[-1].level is EscalationLevel.CRITICO
        assert all(n.level >= EscalationLevel.MEDIO for n in report.notifications)
        kinds = {n.kind for n in report.notifications}
        assert kinds == {"escalation", "member_risk", "recurring_topic", "summary"}

    def test_empty_batch(self) -> None:
        report = decide([], reference_time=NOW)
        assert report.message_count == 0
        assert report.actions == ()
        assert report.highest_level is EscalationLevel.INFO
        assert "Community is healthy" in report.executive_summary
        assert [n.kind for n in report.notifications] == ["summary"]

    def test_custom_config_changes_outcome(self, batch: list[EnrichedMessage]) -> None:
        strict = DecisionConfig(recurring_topics=RecurringTopicsConfig(min_occurrences=4))
        assert decide(batch, config=strict, reference_time=NOW).recurring_topics == ()

    def test_report_serialises(self, batch: list[EnrichedMessage]) -> None:
        report = DecisionEngine().run(batch, reference_time=NOW)
        payload = report.model_dump(mode="json")
        assert payload["highest_level"] == "CRÍTICO"
        assert payload["escalations"][0]["level"] == "CRÍTICO"
