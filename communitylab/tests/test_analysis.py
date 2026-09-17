"""Tests for the analysis layer (sentiment, themes, relevance, consolidator).

All LLM calls are served by ``FakeToolChatModel`` or ``RunnableLambda`` so the
suite runs offline and deterministically.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError
from src.analysis import (
    AnalysisError,
    Consolidator,
    EnrichedMessage,
    RelevanceAnalyzer,
    RelevanceResult,
    SentimentAnalyzer,
    SentimentResult,
    ThemesAnalyzer,
    ThemesResult,
    analyze_relevance,
    analyze_sentiment,
    analyze_themes,
    message_to_prompt_input,
    tier_for_score,
    to_frame,
    to_records,
)
from src.ingest import load_json, normalize, validate
from src.ingest.validator import CommunityMessage
from src.utils.llm_clients import LLMSettings

from tests.conftest import FakeToolChatModel

SENTIMENT_PAYLOAD: dict[str, Any] = {
    "label": "negative",
    "score": -0.6,
    "confidence": 0.8,
    "emotions": ["Frustration", "confusion"],
    "rationale": "Repeated import error despite installing.",
}
THEMES_PAYLOAD: dict[str, Any] = {
    "primary_category": "technical_question",
    "topics": ["Python environment setup", "Dependency installation"],
    "technologies": ["LangGraph", "pip"],
    "keywords": ["modulenotfounderror", "pip install"],
    "summary": "Member cannot import langgraph after installing it.",
}
RELEVANCE_PAYLOAD: dict[str, Any] = {
    "score": 0.7,
    "tier": "high",
    "is_actionable": True,
    "requires_response": True,
    "suggested_channels": ["faq"],
    "reasons": ["Unanswered technical question", "Common onboarding blocker"],
}


def fake_llm(payload: dict[str, Any]) -> RunnableLambda:
    """A structured runnable that ignores its input and returns ``payload``."""
    return RunnableLambda(lambda _: dict(payload))


def failing_llm(message: str = "boom") -> RunnableLambda:
    def _raise(_: Any) -> Any:
        raise RuntimeError(message)

    return RunnableLambda(_raise)


# --------------------------------------------------------------------------- #
# Base plumbing
# --------------------------------------------------------------------------- #
class TestMessageToPromptInput:
    def test_from_community_message(self, community_message: CommunityMessage):
        data = message_to_prompt_input(community_message)
        assert data == {
            "source": "discord",
            "channel": "help-python",
            "author": "alice",
            "timestamp": "2026-09-10T10:00:00+00:00",
            "reactions": 2,
            "is_reply": "no",
            "category": "none",
            "content": community_message.content,
        }

    def test_reply_and_category(self, community_message: CommunityMessage):
        reply = community_message.model_copy(update={"thread_id": "t-1", "category": "question"})
        data = message_to_prompt_input(reply)
        assert data["is_reply"] == "yes"
        assert data["category"] == "question"

    def test_from_plain_text(self):
        data = message_to_prompt_input("just text")
        assert data["content"] == "just text"
        assert data["source"] == "unknown"


class TestStructuredAnalyzer:
    def test_prompt_receives_message_fields(self, community_message: CommunityMessage):
        captured: dict[str, Any] = {}

        def spy(prompt_value: Any) -> dict[str, Any]:
            captured["text"] = prompt_value.to_string()
            return SENTIMENT_PAYLOAD

        SentimentAnalyzer(llm=RunnableLambda(spy)).analyze(community_message)
        assert "System:" in captured["text"]
        assert community_message.content in captured["text"]
        assert "Source: discord" in captured["text"]

    def test_uses_groq_structured_llm_by_default(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy")
        analyzer = SentimentAnalyzer(settings=LLMSettings(primary_model="p", fallback_models=("f",)))
        assert analyzer.name == "sentiment"
        assert analyzer.schema is SentimentResult

    def test_llm_error_is_wrapped(self, community_message: CommunityMessage):
        with pytest.raises(AnalysisError, match="sentiment failed for message m-1: boom"):
            SentimentAnalyzer(llm=failing_llm()).analyze(community_message)

    def test_invalid_output_is_wrapped(self, community_message: CommunityMessage):
        with pytest.raises(AnalysisError) as exc_info:
            SentimentAnalyzer(llm=fake_llm({"label": "happy", "score": 5})).analyze(community_message)
        assert isinstance(exc_info.value.cause, ValidationError)

    def test_accepts_model_instances(self, community_message: CommunityMessage):
        expected = SentimentResult(**SENTIMENT_PAYLOAD)
        result = SentimentAnalyzer(llm=RunnableLambda(lambda _: expected)).analyze(community_message)
        assert result is expected

    def test_analyze_many_keeps_order_and_errors(self, community_message: CommunityMessage):
        other = community_message.model_copy(update={"message_id": "m-2", "content": "Thanks, solved it!"})

        def by_message(prompt_value: Any) -> dict[str, Any]:
            if "solved" in prompt_value.to_string():
                raise RuntimeError("no result")
            return SENTIMENT_PAYLOAD

        results = SentimentAnalyzer(llm=RunnableLambda(by_message)).analyze_many([community_message, other])
        assert isinstance(results[0], SentimentResult)
        assert isinstance(results[1], AnalysisError)
        assert results[1].message_id == "m-2"

    def test_analyze_many_empty(self):
        assert SentimentAnalyzer(llm=fake_llm(SENTIMENT_PAYLOAD)).analyze_many([]) == []

    def test_end_to_end_with_fake_chat_model_and_fallback(self, community_message: CommunityMessage):
        """Prompt -> structured Groq-style model -> Pydantic, with the primary failing."""
        built: dict[str, FakeToolChatModel] = {}

        def factory(model: str, settings: LLMSettings) -> FakeToolChatModel:  # noqa: ARG001
            built[model] = FakeToolChatModel(payload=SENTIMENT_PAYLOAD, name=model, fail=model == "p")
            return built[model]

        from src.utils.llm_clients import get_structured_llm

        llm = get_structured_llm(
            SentimentResult, LLMSettings(primary_model="p", fallback_models=("f",)), model_factory=factory
        )
        result = SentimentAnalyzer(llm=llm).analyze(community_message)
        assert result.label == "negative"
        assert built["p"].calls == 1 and built["f"].calls == 1


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
class TestSentiment:
    def test_analyze_sentiment(self, community_message: CommunityMessage):
        result = analyze_sentiment(community_message, llm=fake_llm(SENTIMENT_PAYLOAD))
        assert result.label == "negative"
        assert result.score == -0.6
        assert result.emotions == ["frustration", "confusion"]
        assert result.is_negative

    def test_emotions_deduped_and_capped(self):
        result = SentimentResult(label="neutral", score=0.0, emotions=["Calm", "calm", "curious", "bored", "x"])
        assert result.emotions == ["calm", "curious", "bored"]

    @pytest.mark.parametrize("label,score", [("positive", -0.2), ("negative", 0.3)])
    def test_sign_must_match_label(self, label: str, score: float):
        with pytest.raises(ValidationError):
            SentimentResult(label=label, score=score)

    def test_mixed_allows_any_sign(self):
        assert SentimentResult(label="mixed", score=-0.1).label == "mixed"

    @pytest.mark.parametrize("bad", [{"label": "angry", "score": 0}, {"label": "neutral", "score": 1.5}])
    def test_rejects_out_of_range(self, bad: dict[str, Any]):
        with pytest.raises(ValidationError):
            SentimentResult(**bad)

    def test_result_is_frozen(self):
        result = SentimentResult(label="positive", score=0.9)
        with pytest.raises(ValidationError):
            result.score = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
class TestThemes:
    def test_analyze_themes(self, community_message: CommunityMessage):
        result = analyze_themes(community_message, llm=fake_llm(THEMES_PAYLOAD))
        assert result.primary_category == "technical_question"
        assert result.technologies == ["LangGraph", "pip"]
        assert result.is_question

    def test_category_is_normalised(self):
        result = ThemesResult(primary_category=" Success Story ")
        assert result.primary_category == "success_story"

    def test_unknown_category_rejected(self):
        with pytest.raises(ValidationError):
            ThemesResult(primary_category="rant")

    def test_lists_cleaned(self):
        result = ThemesResult(
            primary_category="other",
            topics=["A", "a", " B ", "", "C", "D", "E", "F"],
            keywords=["OCI", "oci", "Lab"],
            technologies=["Groq", "groq"],
        )
        assert result.topics == ["A", "B", "C", "D", "E"]
        assert result.keywords == ["oci", "lab"]
        assert result.technologies == ["Groq"]

    def test_analyzer_metadata(self):
        analyzer = ThemesAnalyzer(llm=fake_llm(THEMES_PAYLOAD))
        assert analyzer.name == "themes"
        assert analyzer.schema is ThemesResult


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
class TestRelevance:
    def test_analyze_relevance(self, community_message: CommunityMessage):
        result = analyze_relevance(community_message, llm=fake_llm(RELEVANCE_PAYLOAD))
        assert result.tier == "high"
        assert result.is_actionable and result.requires_response
        assert result.suggested_channels == ["faq"]
        assert not result.is_noise

    @pytest.mark.parametrize(
        "score,tier",
        [
            (0.0, "noise"),
            (0.19, "noise"),
            (0.2, "low"),
            (0.4, "medium"),
            (0.6, "high"),
            (0.85, "critical"),
            (1.0, "critical"),
        ],
    )
    def test_tier_for_score(self, score: float, tier: str):
        assert tier_for_score(score) == tier

    def test_inconsistent_tier_is_corrected(self):
        result = RelevanceResult(score=0.05, tier="critical")
        assert result.tier == "noise"
        assert result.is_noise

    def test_unknown_channels_dropped(self):
        result = RelevanceResult(score=0.5, suggested_channels=["LinkedIn", "tiktok", "linkedin", "faq"])
        assert result.suggested_channels == ["linkedin", "faq"]

    def test_reasons_capped(self):
        result = RelevanceResult(score=0.5, reasons=["a", " ", "b", "c", "d"])
        assert result.reasons == ["a", "b", "c"]

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            RelevanceResult(score=1.2)

    def test_analyzer_metadata(self):
        analyzer = RelevanceAnalyzer(llm=fake_llm(RELEVANCE_PAYLOAD))
        assert analyzer.name == "relevance"


# --------------------------------------------------------------------------- #
# Consolidator
# --------------------------------------------------------------------------- #
@pytest.fixture
def consolidator() -> Consolidator:
    return Consolidator.from_llms(
        sentiment_llm=fake_llm(SENTIMENT_PAYLOAD),
        themes_llm=fake_llm(THEMES_PAYLOAD),
        relevance_llm=fake_llm(RELEVANCE_PAYLOAD),
    )


class TestConsolidator:
    def test_enrich_merges_all_outputs(self, consolidator: Consolidator, community_message: CommunityMessage):
        enriched = consolidator.enrich(community_message)
        assert isinstance(enriched, EnrichedMessage)
        assert enriched.message is community_message
        assert enriched.message_id == "m-1"
        assert enriched.sentiment == SentimentResult(**SENTIMENT_PAYLOAD)
        assert enriched.themes == ThemesResult(**THEMES_PAYLOAD)
        assert enriched.relevance == RelevanceResult(**RELEVANCE_PAYLOAD)
        assert enriched.errors == {}
        assert enriched.is_complete
        assert enriched.analyzed_at.tzinfo is not None

    def test_partial_failure_is_recorded(self, community_message: CommunityMessage):
        consolidator = Consolidator.from_llms(
            sentiment_llm=failing_llm("groq down"),
            themes_llm=fake_llm(THEMES_PAYLOAD),
            relevance_llm=fake_llm(RELEVANCE_PAYLOAD),
        )
        enriched = consolidator.enrich(community_message)
        assert enriched.sentiment is None
        assert enriched.themes is not None and enriched.relevance is not None
        assert enriched.errors == {"sentiment": "groq down"}
        assert not enriched.is_complete

    def test_strict_mode_raises(self, community_message: CommunityMessage):
        consolidator = Consolidator.from_llms(
            sentiment_llm=fake_llm(SENTIMENT_PAYLOAD),
            themes_llm=failing_llm(),
            relevance_llm=fake_llm(RELEVANCE_PAYLOAD),
            strict=True,
        )
        with pytest.raises(AnalysisError, match="themes failed"):
            consolidator.enrich(community_message)

    def test_enrich_many_preserves_order(self, consolidator: Consolidator, sample_path):
        messages = validate(normalize(load_json(sample_path))).valid
        enriched = consolidator.enrich_many(messages)
        assert [e.message_id for e in enriched] == [m.message_id for m in messages]
        assert all(e.is_complete for e in enriched)

    def test_enrich_many_isolates_failures(self, community_message: CommunityMessage):
        other = community_message.model_copy(update={"message_id": "m-2", "content": "NOISE_MARKER"})

        def flaky(prompt_value: Any) -> dict[str, Any]:
            if "NOISE_MARKER" in prompt_value.to_string():
                raise RuntimeError("noise")
            return RELEVANCE_PAYLOAD

        consolidator = Consolidator.from_llms(
            sentiment_llm=fake_llm(SENTIMENT_PAYLOAD),
            themes_llm=fake_llm(THEMES_PAYLOAD),
            relevance_llm=RunnableLambda(flaky),
        )
        first, second = consolidator.enrich_many([community_message, other])
        assert first.is_complete
        assert second.relevance is None and second.errors == {"relevance": "noise"}

    def test_enrich_many_empty(self, consolidator: Consolidator):
        assert consolidator.enrich_many([]) == []

    def test_custom_analyzers_are_used(self, community_message: CommunityMessage):
        consolidator = Consolidator(
            sentiment=SentimentAnalyzer(llm=fake_llm(SENTIMENT_PAYLOAD)),
            themes=ThemesAnalyzer(llm=fake_llm(THEMES_PAYLOAD)),
            relevance=RelevanceAnalyzer(llm=fake_llm(RELEVANCE_PAYLOAD)),
        )
        assert consolidator.enrich(community_message).is_complete

    def test_to_record_flattens_with_prefixes(self, consolidator: Consolidator, community_message: CommunityMessage):
        record = consolidator.enrich(community_message).to_record()
        assert record["message_id"] == "m-1"
        assert record["sentiment_label"] == "negative"
        assert record["themes_primary_category"] == "technical_question"
        assert record["relevance_tier"] == "high"
        assert record["analysis_errors"] == {}
        assert isinstance(record["analyzed_at"], str)
        # The raw ``sentiment`` column from ingestion is kept separately from the analysis output.
        assert record["sentiment"] is None

    def test_to_records_and_frame(self, consolidator: Consolidator, sample_path):
        messages = validate(normalize(load_json(sample_path))).valid
        enriched = consolidator.enrich_many(messages)
        records = to_records(enriched)
        assert len(records) == len(messages)
        frame = to_frame(enriched)
        assert list(frame["message_id"]) == [m.message_id for m in messages]
        assert {"sentiment_score", "themes_topics", "relevance_score", "analysis_errors"} <= set(frame.columns)

    def test_enriched_message_serialises(self, consolidator: Consolidator, community_message: CommunityMessage):
        enriched = consolidator.enrich(community_message)
        dumped = enriched.model_dump(mode="json")
        assert dumped["sentiment"]["label"] == "negative"
        assert EnrichedMessage.model_validate(dumped).sentiment == enriched.sentiment
