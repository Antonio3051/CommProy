"""Deterministic, keyword-driven stand-ins for the Groq structured LLMs.

Used by the Streamlit *demo mode* and by tests so the whole pipeline can run
without an API key. Each :class:`MockStructuredLLM` is bound to one output
schema and derives a plausible instance from the prompt text alone::

    components = mock_components(storage=AssetStorage(StorageSettings(force_local=True)))
    compile_pipeline(components).invoke(initial_state("json", "data/sample/messages.json"))
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from src.analysis import Consolidator, RelevanceResult, SentimentResult, ThemesResult
from src.analysis.relevance import tier_for_score
from src.decisions import DecisionEngine
from src.generators import (
    FAQEntry,
    FAQGenerator,
    LinkedInGenerator,
    LinkedInPost,
    NewsletterGenerator,
    NewsletterSection,
    Testimonial,
    TestimonialGenerator,
)
from src.oci import AssetStorage
from src.orchestration.router import Generators, PipelineComponents

POSITIVE_WORDS = frozenset(
    "thanks thank shout-out shoutout huge great awesome love landed hired offer works solved finally 🎉 amazing".split()
)
NEGATIVE_WORDS = frozenset(
    "frustrated frustrating broken stuck error failing fails annoying waste giving up unusable bug crash 😕".split()
)
TECH_WORDS = ("python", "langgraph", "langchain", "groq", "oci", "pandas", "pydantic", "docker", "venv", "pip", "sql")


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'🎉😕-]+", text.lower())


def _last_message_text(prompt_text: str) -> str:
    """The human/message part of a rendered prompt (everything after the last ``Human:``)."""
    _, _, tail = prompt_text.rpartition("Human:")
    return tail or prompt_text


def _content_lines(prompt_text: str) -> list[str]:
    """Message bodies quoted in a generator brief (``text: \"\"\"…\"\"\"``), or the whole text."""
    lines = [m.group(1).strip() for m in re.finditer(r'text: """(.*?)"""', prompt_text, flags=re.DOTALL)]
    return lines or [_last_message_text(prompt_text).strip()]


def mock_sentiment(text: str) -> dict[str, Any]:
    words = _words(text)
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    score = max(-1.0, min(1.0, 0.3 * (pos - neg)))
    if pos and neg:
        label = "mixed"
    elif score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": score, "confidence": 0.6, "rationale": f"{pos} positive / {neg} negative cue(s)."}


def mock_themes(text: str) -> dict[str, Any]:
    lower = text.lower()
    words = _words(text)
    if any(w in NEGATIVE_WORDS for w in words) and ("weeks" in lower or "still" in lower or "broken" in lower):
        category = "complaint"
    elif "error" in lower or "?" in text or "how do" in lower or "any idea" in lower:
        category = "technical_question"
    elif any(w in POSITIVE_WORDS for w in words):
        category = "success_story"
    elif "@" in text and ("try" in lower or "run" in lower or "because" in lower):
        category = "technical_answer"
    else:
        category = "community_chatter"
    technologies = [t for t in TECH_WORDS if t in lower]
    topics = [f"{technologies[0]} setup"] if technologies else ([category.replace("_", " ")])
    if "modulenotfounderror" in lower or "no module named" in lower:
        topics = ["missing module in venv"]
    keywords = sorted({w for w in words if len(w) > 5})[:5]
    summary = text.strip().split("\n")[0][:120]
    return {
        "primary_category": category,
        "topics": topics,
        "technologies": technologies,
        "keywords": keywords,
        "summary": summary,
    }


def mock_relevance(text: str) -> dict[str, Any]:
    themes = mock_themes(text)
    base = {
        "technical_question": 0.55,
        "technical_answer": 0.6,
        "success_story": 0.75,
        "complaint": 0.7,
        "community_chatter": 0.1,
    }.get(themes["primary_category"], 0.4)
    score = min(1.0, base + 0.02 * len(themes["technologies"]))
    channels = {"success_story": ["linkedin", "testimonial"], "technical_answer": ["faq"], "complaint": ["faq"]}.get(
        themes["primary_category"], []
    )
    return {
        "score": score,
        "tier": tier_for_score(score),
        "is_actionable": score >= 0.4,
        "requires_response": themes["primary_category"] in ("technical_question", "complaint"),
        "suggested_channels": channels,
        "reasons": [f"{themes['primary_category']} with {len(themes['technologies'])} technology mention(s)"],
    }


def mock_linkedin(text: str) -> dict[str, Any]:
    first = _content_lines(text)[0]
    return {
        "hook": f"A community win worth sharing: {first[:80].rstrip('.!')}.",
        "body": (
            f"{first}\n\nBehind every milestone like this there are hours of practice, questions asked in public "
            "and people who answered them.\n\nWhat was the moment things clicked for you? Tell us below."
        ),
        "hashtags": ["#Community", "#Learning", "#DataEngineering"],
    }


def mock_newsletter(text: str) -> dict[str, Any]:
    lines = _content_lines(text)
    return {
        "title": "This week in the community",
        "intro": f"{len(lines)} conversation(s) stood out this week — here is the short version.",
        "highlights": [line[:140] for line in lines[:5]],
        "closing": "Reply to this email or drop by the help channel if any of these resonate with you.",
    }


def mock_faq(text: str) -> dict[str, Any]:
    lines = _content_lines(text)
    question = next((line for line in lines if "?" in line), lines[0])
    answers = [line for line in lines if "?" not in line]
    answer = answers[0] if answers else "Confirm the virtual environment is activated, then reinstall the requirements."
    return {
        "question": question[:200],
        "answer": answer[:900],
        "tags": [t for t in TECH_WORDS if t in text.lower()][:5],
        "answer_is_complete": bool(answers),
    }


def mock_testimonial(text: str) -> dict[str, Any]:
    first = _content_lines(text)[0]
    quote = re.sub(r"@\w+", "", first).strip()
    if len(quote.split()) < 25:
        quote += " The support from this community made the difference for me."
    return {"quote": quote[:400], "context": "Shared in the community channel.", "attribution": "Community member"}


MOCK_BUILDERS: dict[type[BaseModel], Callable[[str], dict[str, Any]]] = {
    SentimentResult: mock_sentiment,
    ThemesResult: mock_themes,
    RelevanceResult: mock_relevance,
    LinkedInPost: mock_linkedin,
    NewsletterSection: mock_newsletter,
    FAQEntry: mock_faq,
    Testimonial: mock_testimonial,
}


def mock_structured_llm(schema: type[BaseModel]) -> RunnableLambda:
    """A runnable that maps a prompt value to a dict valid for ``schema``."""
    builder = MOCK_BUILDERS[schema]

    def _run(prompt_value: Any) -> dict[str, Any]:
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return builder(_last_message_text(text))

    return RunnableLambda(_run)


def mock_consolidator() -> Consolidator:
    """Consolidator whose three analyzers are keyword heuristics."""
    return Consolidator.from_llms(
        sentiment_llm=mock_structured_llm(SentimentResult),
        themes_llm=mock_structured_llm(ThemesResult),
        relevance_llm=mock_structured_llm(RelevanceResult),
    )


def mock_generators() -> Generators:
    """All four generators backed by templated mocks."""
    return Generators(
        linkedin=LinkedInGenerator(llm=mock_structured_llm(LinkedInPost)),
        newsletter=NewsletterGenerator(llm=mock_structured_llm(NewsletterSection)),
        faq=FAQGenerator(llm=mock_structured_llm(FAQEntry)),
        testimonial=TestimonialGenerator(llm=mock_structured_llm(Testimonial)),
    )


def mock_components(*, storage: AssetStorage | None = None, engine: DecisionEngine | None = None) -> PipelineComponents:
    """Full pipeline wiring that never calls Groq."""
    return PipelineComponents(
        consolidator=mock_consolidator(),
        engine=engine if engine is not None else DecisionEngine(),
        generators=mock_generators(),
        storage=storage if storage is not None else AssetStorage(),
    )
