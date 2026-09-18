"""Tests for ``src.prompts.system_prompts``."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from src.prompts.system_prompts import (
    ANALYSIS_PROMPTS,
    CHANNELS,
    COPYWRITING_PROMPTS,
    MESSAGE_HUMAN_TEMPLATE,
    RELEVANCE_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    TOPIC_EXTRACTION_SYSTEM_PROMPT,
    build_message_prompt,
    get_copywriting_prompt,
)


def test_channels_match_registry():
    assert set(CHANNELS) == {"linkedin", "newsletter", "faq", "testimonial"}
    assert set(COPYWRITING_PROMPTS) == set(CHANNELS)


def test_analysis_registry_complete():
    assert set(ANALYSIS_PROMPTS) == {"sentiment", "themes", "relevance"}


@pytest.mark.parametrize("prompt", [*ANALYSIS_PROMPTS.values(), *COPYWRITING_PROMPTS.values()])
def test_prompts_are_structured(prompt: str):
    """Every prompt states a role, a task and rules, and is non-trivial."""
    assert "Role:" in prompt
    assert "Task:" in prompt
    assert "Rules:" in prompt
    assert len(prompt) > 300


def test_analysis_prompts_name_their_output_fields():
    for field in ("label", "score", "confidence", "emotions", "rationale"):
        assert field in SENTIMENT_SYSTEM_PROMPT
    for field in ("primary_category", "topics", "technologies", "keywords", "summary"):
        assert field in TOPIC_EXTRACTION_SYSTEM_PROMPT
    for field in ("score", "tier", "is_actionable", "requires_response", "suggested_channels", "reasons"):
        assert field in RELEVANCE_SYSTEM_PROMPT


def test_prompts_defend_against_injection():
    for prompt in ANALYSIS_PROMPTS.values():
        assert "data" in prompt.lower() and "instruction" in prompt.lower()


def test_get_copywriting_prompt():
    assert get_copywriting_prompt("linkedin") == COPYWRITING_PROMPTS["linkedin"]
    with pytest.raises(KeyError, match="Unknown channel 'tiktok'"):
        get_copywriting_prompt("tiktok")  # type: ignore[arg-type]


def test_build_message_prompt_renders_all_fields():
    prompt = build_message_prompt("System text with {braces} that must survive.")
    assert set(prompt.input_variables) == {
        "source",
        "channel",
        "author",
        "timestamp",
        "reactions",
        "is_reply",
        "category",
        "content",
    }
    messages = prompt.invoke(
        {
            "source": "discord",
            "channel": "help",
            "author": "alice",
            "timestamp": "2026-09-10T10:00:00+00:00",
            "reactions": 3,
            "is_reply": "no",
            "category": "none",
            "content": "Hello {world}",
        }
    ).to_messages()
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "System text with {braces} that must survive."
    assert isinstance(messages[1], HumanMessage)
    assert "Hello {world}" in messages[1].content
    assert "Source: discord" in messages[1].content
    assert "Reactions: 3" in messages[1].content


def test_human_template_lists_every_variable():
    for var in ("source", "channel", "author", "timestamp", "reactions", "is_reply", "category", "content"):
        assert f"{{{var}}}" in MESSAGE_HUMAN_TEMPLATE
