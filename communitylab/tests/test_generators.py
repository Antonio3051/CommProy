"""Tests for the content generators (src/generators)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError
from src.analysis import EnrichedMessage, RelevanceResult, SentimentResult, ThemesResult
from src.decisions import RecurringTopic
from src.generators import (
    BRIEF_HUMAN_TEMPLATE,
    FAQEntry,
    FAQGenerator,
    GeneratedAsset,
    GenerationError,
    LinkedInGenerator,
    LinkedInPost,
    NewsletterGenerator,
    NewsletterSection,
    build_brief,
    format_message_block,
    select_highlights,
    select_success_stories,
    select_testimonial_candidates,
    slugify,
    thread_messages,
)
from src.generators import testimonials as tm
from src.generators.newsletter import default_period_label
from src.ingest.validator import CommunityMessage
from src.prompts.system_prompts import get_copywriting_prompt
from src.utils.llm_clients import LLMSettings, get_structured_llm

from tests.conftest import FakeToolChatModel

T0 = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)


def enriched(
    message_id: str,
    content: str,
    *,
    author: str = "alice",
    score: float = 0.0,
    label: str = "neutral",
    category: str = "community_chatter",
    topics: list[str] | None = None,
    relevance: float = 0.5,
    channels: list[str] | None = None,
    thread_id: str | None = None,
    hours: float = 0,
    with_analysis: bool = True,
) -> EnrichedMessage:
    message = CommunityMessage(
        message_id=message_id,
        source="discord",
        channel="general",
        author=author,
        author_id=author,
        content=content,
        timestamp=T0 + timedelta(hours=hours),
        thread_id=thread_id,
    )
    if not with_analysis:
        return EnrichedMessage(message=message)
    return EnrichedMessage(
        message=message,
        sentiment=SentimentResult(label=label, score=score),
        themes=ThemesResult(primary_category=category, topics=topics or []),
        relevance=RelevanceResult(score=relevance, suggested_channels=channels or []),
    )


def fake_llm(payload: dict[str, Any]) -> RunnableLambda:
    return RunnableLambda(lambda _: dict(payload))


def failing_llm(message: str = "boom") -> RunnableLambda:
    def _raise(_: Any) -> Any:
        raise RuntimeError(message)

    return RunnableLambda(_raise)


def recording_llm(payload: dict[str, Any], seen: list[Any]) -> RunnableLambda:
    def _run(prompt_value: Any) -> dict[str, Any]:
        seen.append(prompt_value)
        return dict(payload)

    return RunnableLambda(_run)


LINKEDIN_PAYLOAD = {
    "hook": "Six weeks ago I had never deployed anything.",
    "body": "Six weeks ago I had never deployed anything.\n\nToday my pipeline runs in prod.\n\nWhat helped you ship?",
    "hashtags": ["#DataEngineering", "Python", "#Community", "#TooMany"],
}
NEWSLETTER_PAYLOAD = {
    "title": "This week in the community",
    "intro": "Busy week: two unanswered questions and a big win.",
    "highlights": ["Docker networking confusion in #help", "Priya shipped her first pipeline"],
    "closing": "Reply and tell us what you are building.",
}
FAQ_PAYLOAD = {
    "question": "How do I fix ModuleNotFoundError for langgraph after pip install?",
    "answer": "Make sure you install into the same interpreter: run `python -m pip install langgraph`.",
    "tags": ["#Python", "pip", "Python", "venv"],
    "answer_is_complete": True,
}
TESTIMONIAL_PAYLOAD = {
    "quote": "The mentors here got me from zero to a deployed pipeline in six weeks.",
    "context": "Shared after completing the data-engineering track.",
    "attribution": "Data engineering track member",
    "needs_consent": False,
}


# --------------------------------------------------------------------------- #
# Base plumbing
# --------------------------------------------------------------------------- #
class TestBase:
    def test_slugify(self):
        assert slugify("  ¡Hola, Mundo! Café & Python 3.12 ") == "hola-mundo-cafe-python-3-12"
        assert slugify("x" * 100, max_length=10) == "x" * 10
        assert slugify("!!!") == ""

    def test_format_message_block_includes_analysis(self):
        item = enriched("m1", "Docker fails", score=-0.6, label="negative", category="complaint", topics=["docker"])
        block = format_message_block(item)
        assert "id: m1" in block and "alice" in block and "Docker fails" in block
        assert "negative" in block and "complaint" in block and "docker" in block

    def test_format_message_block_without_analysis(self):
        block = format_message_block(enriched("m1", "hello", with_analysis=False))
        assert "hello" in block and "sentiment" not in block.lower()

    def test_build_brief_variables(self):
        variables = build_brief([enriched("m1", "a"), enriched("m2", "b")], context="ctx", language="es")
        assert set(variables) == {"language", "context", "messages"}
        assert variables["language"] == "es" and variables["context"] == "ctx"
        assert "id: m1" in variables["messages"] and "id: m2" in variables["messages"]
        rendered = BRIEF_HUMAN_TEMPLATE.format(**variables)
        assert "Write in this language: es" in rendered

    def test_generator_uses_phase2_prompt_and_brief(self):
        seen: list[Any] = []
        LinkedInGenerator(llm=recording_llm(LINKEDIN_PAYLOAD, seen)).generate(
            enriched("m1", "Shipped it!", score=0.9, label="positive", category="success_story"), language="es"
        )
        messages = seen[0].to_messages()
        assert messages[0].content == get_copywriting_prompt("linkedin")
        assert "Write in this language: es" in messages[1].content
        assert "Shipped it!" in messages[1].content

    def test_generation_error_wraps_llm_failure(self):
        with pytest.raises(GenerationError, match="boom"):
            LinkedInGenerator(llm=failing_llm()).generate(enriched("m1", "x"))

    def test_generation_error_wraps_invalid_output(self):
        with pytest.raises(GenerationError):
            NewsletterGenerator(llm=fake_llm({"title": "t", "intro": "i", "highlights": []})).generate(
                [enriched("m1", "x", relevance=0.7)]
            )

    def test_works_with_real_structured_output_pipeline(self):
        fake = FakeToolChatModel(payload=LINKEDIN_PAYLOAD)
        llm = get_structured_llm(LinkedInPost, LLMSettings(api_key="k"), model_factory=lambda _n, _s: fake)
        asset = LinkedInGenerator(llm=llm).generate(enriched("m1", "Shipped!", score=0.8, label="positive"))
        assert isinstance(asset.content, LinkedInPost)
        assert fake.calls == 1

    def test_generated_asset_filename_and_record(self):
        asset = GeneratedAsset(
            channel="faq",
            title="How do I fix it?",
            content=FAQEntry(question="q", answer="a"),
            markdown="### q",
            source_message_ids=("m1",),
            generated_at=datetime(2026, 9, 17, 8, 30, 5, tzinfo=UTC),
        )
        assert asset.slug == "how-do-i-fix-it"
        assert asset.filename() == "faq/20260917-083005-how-do-i-fix-it.md"
        assert asset.filename(".json") == "faq/20260917-083005-how-do-i-fix-it.json"
        record = asset.to_record()
        assert record["content"] == {"question": "q", "answer": "a", "tags": [], "answer_is_complete": True}
        assert record["source_message_ids"] == ["m1"] and record["generated_at"].startswith("2026-09-17")

    def test_generated_asset_slug_falls_back_to_channel(self):
        asset = GeneratedAsset(channel="faq", title="???", content=FAQEntry(question="q", answer="a"), markdown="")
        assert asset.slug == "faq"


# --------------------------------------------------------------------------- #
# LinkedIn
# --------------------------------------------------------------------------- #
class TestLinkedIn:
    def test_post_validation_normalises_hashtags(self):
        post = LinkedInPost(**LINKEDIN_PAYLOAD)
        assert post.hashtags == ["DataEngineering", "Python", "Community"]
        assert post.render().endswith("#DataEngineering #Python #Community")

    def test_render_without_hashtags(self):
        post = LinkedInPost(hook="h", body="body text  ", hashtags=[])
        assert post.render() == "body text"

    def test_select_success_stories(self):
        items = [
            enriched("story", "win", score=0.8, label="positive", category="success_story", relevance=0.6),
            enriched("suggested", "nice", score=0.2, category="feedback", relevance=0.9, channels=["linkedin"]),
            enriched("negative", "ugh", score=-0.7, label="negative", category="success_story"),
            enriched("question", "how?", category="technical_question"),
            enriched("raw", "no analysis", with_analysis=False),
        ]
        assert [i.message_id for i in select_success_stories(items)] == ["suggested", "story"]

    def test_generate_builds_asset(self):
        item = enriched("m1", "Shipped!", score=0.9, label="positive", category="success_story", topics=["prod"])
        asset = LinkedInGenerator(llm=fake_llm(LINKEDIN_PAYLOAD)).generate(item, language="es")
        assert asset.channel == "linkedin"
        assert asset.title == LINKEDIN_PAYLOAD["hook"]
        assert asset.language == "es"
        assert asset.source_message_ids == ("m1",)
        assert asset.markdown == asset.content.render()
        assert asset.metadata["author"] == "alice" and asset.metadata["topics"] == ["prod"]

    def test_generate_many_filters_limits_and_skips_failures(self, caplog: pytest.LogCaptureFixture):
        items = [
            enriched(f"s{i}", "win", score=0.9, label="positive", category="success_story", relevance=0.5 + i / 10)
            for i in range(3)
        ] + [enriched("q", "how?", category="technical_question")]
        gen = LinkedInGenerator(llm=fake_llm(LINKEDIN_PAYLOAD))
        assets = gen.generate_many(items, limit=2)
        assert [a.source_message_ids[0] for a in assets] == ["s2", "s1"]

        failing = LinkedInGenerator(llm=failing_llm("down"))
        with caplog.at_level("WARNING"):
            assert failing.generate_many(items) == []
        assert "down" in caplog.text
        with pytest.raises(GenerationError):
            failing.generate_many(items, raise_on_error=True)


# --------------------------------------------------------------------------- #
# Newsletter
# --------------------------------------------------------------------------- #
class TestNewsletter:
    def test_section_requires_highlights(self):
        with pytest.raises(ValidationError):
            NewsletterSection(title="t", intro="i", highlights=[])

    def test_render_markdown(self):
        md = NewsletterSection(**NEWSLETTER_PAYLOAD).render()
        assert md.startswith("## This week in the community\n\nBusy week")
        assert "- Docker networking confusion in #help\n- Priya shipped" in md
        assert md.endswith("Reply and tell us what you are building.")

    def test_select_highlights_orders_by_editorial_priority(self):
        items = [
            enriched("noise", "lol", relevance=0.05),
            enriched("low", "meh", relevance=0.2),
            enriched("win", "shipped", category="success_story", relevance=0.9),
            enriched("q", "how do I...", category="technical_question", relevance=0.5),
            enriched("complaint", "broken again", category="complaint", relevance=0.95),
            enriched("share", "great article", category="resource_share", relevance=0.45),
        ]
        picked = [i.message_id for i in select_highlights(items)]
        assert picked == ["complaint", "q", "win", "share"]
        assert [i.message_id for i in select_highlights(items, max_messages=2)] == ["complaint", "q"]

    def test_default_period_label(self):
        items = [enriched("a", "x", hours=0), enriched("b", "y", hours=72)]
        assert default_period_label(items) == "2026-09-14 to 2026-09-17"
        assert default_period_label([]) == date.today().isoformat()

    def test_generate_section(self):
        seen: list[Any] = []
        items = [
            enriched("noise", "lol", relevance=0.05),
            enriched("q", "how do I run docker?", category="technical_question", relevance=0.7, hours=1),
            enriched("win", "shipped!", category="success_story", relevance=0.5, hours=30),
        ]
        gen = NewsletterGenerator(llm=recording_llm(NEWSLETTER_PAYLOAD, seen))
        asset = gen.generate(items, community_name="DataLab")
        assert asset.channel == "newsletter"
        assert asset.title == "This week in the community"
        assert asset.source_message_ids == ("q", "win")
        assert asset.metadata["community"] == "DataLab"
        assert asset.metadata["period"] == "2026-09-14 to 2026-09-15"
        assert asset.metadata["messages_analysed"] == 3
        human = seen[0].to_messages()[1].content
        assert "DataLab" in human and "id: noise" not in human and "id: q" in human

    def test_generate_with_nothing_relevant_raises(self):
        with pytest.raises(ValueError, match="No messages"):
            NewsletterGenerator(llm=fake_llm(NEWSLETTER_PAYLOAD)).generate([enriched("n", "lol", relevance=0.05)])


# --------------------------------------------------------------------------- #
# FAQ
# --------------------------------------------------------------------------- #
def topic(*message_ids: str, action: str = "faq") -> RecurringTopic:
    return RecurringTopic(
        topic_key="langgraph install",
        topic="LangGraph install",
        occurrences=len(message_ids),
        distinct_authors=len(message_ids),
        action=action,
        message_ids=message_ids,
        authors=tuple(f"u{i}" for i in range(len(message_ids))),
    )


class TestFAQ:
    def test_entry_normalises_tags_and_renders(self):
        entry = FAQEntry(**FAQ_PAYLOAD)
        assert entry.tags == ["python", "pip", "venv"]
        md = entry.render()
        assert md.startswith("### How do I fix ModuleNotFoundError")
        assert "`python -m pip install langgraph`" in md
        assert "Tags: `python`, `pip`, `venv`" in md
        assert "partial" not in md

    def test_incomplete_answer_is_flagged(self):
        md = FAQEntry(question="q", answer="a", answer_is_complete=False).render()
        assert "**Note:** this answer is partial" in md

    def test_thread_messages_pulls_replies_in_order(self):
        items = [
            enriched("q2", "same error here", author="bob", hours=5),
            enriched("a1", "use python -m pip", author="mentor", thread_id="q1", hours=1),
            enriched("q1", "ModuleNotFoundError langgraph", hours=0),
            enriched("other", "unrelated", hours=2),
            enriched("missing-reply", "reply to unknown", thread_id="zzz", hours=3),
        ]
        picked = thread_messages(topic("q1", "q2", "ghost"), items)
        assert [i.message_id for i in picked] == ["q1", "a1", "q2"]
        assert [i.message_id for i in thread_messages(topic("q1"), items, include_replies=False)] == ["q1"]

    def test_generate_for_topic(self):
        items = [
            enriched("q1", "ModuleNotFoundError langgraph", category="technical_question", topics=["langgraph"]),
            enriched("a1", "use python -m pip", author="mentor", thread_id="q1", hours=1),
            enriched("q2", "same here", author="bob", category="technical_question", hours=2),
        ]
        gen = FAQGenerator(llm=fake_llm(FAQ_PAYLOAD))
        asset = gen.generate_for_topic(topic("q1", "q2", action="faq_and_mentorship"), items)
        assert asset.channel == "faq"
        assert asset.title == FAQ_PAYLOAD["question"]
        assert asset.source_message_ids == ("q1", "a1", "q2")
        assert asset.metadata["topic"] == "LangGraph install"
        assert asset.metadata["occurrences"] == 2
        assert asset.metadata["recommended_action"] == "faq_and_mentorship"
        assert asset.metadata["answer_is_complete"] is True

    def test_generate_for_topic_without_messages_raises(self):
        with pytest.raises(ValueError, match="present in the batch"):
            FAQGenerator(llm=fake_llm(FAQ_PAYLOAD)).generate_for_topic(topic("nope"), [enriched("m", "x")])

    def test_generate_requires_messages_and_infers_topic(self):
        gen = FAQGenerator(llm=fake_llm(FAQ_PAYLOAD))
        with pytest.raises(ValueError):
            gen.generate([])
        asset = gen.generate([enriched("q1", "x", topics=["docker networking"])])
        assert asset.metadata["topic"] == "docker networking"

    def test_generate_many_filters_by_action(self, caplog: pytest.LogCaptureFixture):
        items = [enriched("q1", "x", category="technical_question"), enriched("q2", "y", author="bob")]
        topics = [
            topic("q1", "q2", action="faq"),
            topic("q1", action="mentorship"),
            topic("ghost", action="faq_and_mentorship"),
        ]
        gen = FAQGenerator(llm=fake_llm(FAQ_PAYLOAD))
        with caplog.at_level("WARNING"):
            assets = gen.generate_many(topics, items)
        assert len(assets) == 1 and assets[0].source_message_ids == ("q1", "q2")
        assert "Skipping topic" in caplog.text
        assert len(gen.generate_many(topics[:2], items, only_faq_actions=False)) == 2
        with pytest.raises(ValueError):
            gen.generate_many(topics, items, raise_on_error=True)


# --------------------------------------------------------------------------- #
# Testimonials
# --------------------------------------------------------------------------- #
class TestTestimonials:
    def test_render_with_consent_flag(self):
        md = tm.Testimonial(**TESTIMONIAL_PAYLOAD).render()
        assert md.startswith("> The mentors here")
        assert "> — *Data engineering track member*" in md
        assert "Consent required" not in md
        flagged = tm.Testimonial(quote="I work at ACME", attribution="", needs_consent=True).render()
        assert "> — *Community member*" in flagged and "**Consent required before publishing**" in flagged

    def test_quote_required(self):
        with pytest.raises(ValidationError):
            tm.Testimonial(quote="")

    def test_select_candidates(self):
        items = [
            enriched("story", "great", score=0.6, label="positive", category="success_story"),
            enriched("best", "amazing", score=0.95, label="positive", channels=["testimonial"]),
            enriched("lukewarm", "ok I guess", score=0.1, category="success_story"),
            enriched("negative", "hated it", score=-0.8, label="negative", channels=["testimonial"]),
            enriched("question", "how?", score=0.7, label="positive", category="technical_question"),
            enriched("raw", "no analysis", with_analysis=False),
        ]
        assert [i.message_id for i in select_testimonial_candidates(items)] == ["best", "story"]
        assert [i.message_id for i in select_testimonial_candidates(items, min_score=0.0)] == [
            "best",
            "story",
            "lukewarm",
        ]

    def test_generate_asset_and_naming_instruction(self):
        seen: list[Any] = []
        item = enriched("m1", "Loved the mentors", score=0.9, label="positive", category="success_story")
        gen = tm.TestimonialGenerator(llm=recording_llm(TESTIMONIAL_PAYLOAD, seen))
        asset = gen.generate(item)
        assert asset.channel == "testimonial"
        assert asset.title == "Testimonial — Data engineering track member"
        assert asset.source_message_ids == ("m1",)
        assert asset.metadata == {"author": "alice", "needs_consent": False, "sentiment_score": 0.9}
        assert "Do not use the member's name" in seen[0].to_messages()[1].content
        gen.generate(item, allow_names=True)
        assert "first name may be used" in seen[1].to_messages()[1].content

    def test_generate_many(self, caplog: pytest.LogCaptureFixture):
        items = [
            enriched("a", "great", score=0.6, label="positive", category="success_story"),
            enriched("b", "amazing", score=0.9, label="positive", category="success_story"),
            enriched("c", "meh", score=0.0),
        ]
        assets = tm.TestimonialGenerator(llm=fake_llm(TESTIMONIAL_PAYLOAD)).generate_many(items, limit=1)
        assert [a.source_message_ids[0] for a in assets] == ["b"]
        with caplog.at_level("WARNING"):
            assert tm.TestimonialGenerator(llm=failing_llm("nope")).generate_many(items) == []
        assert "nope" in caplog.text
