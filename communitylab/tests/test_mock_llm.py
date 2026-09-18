"""The offline mock LLMs must produce schema-valid, sensible outputs for the demo mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.prompt_values import StringPromptValue
from src.analysis import RelevanceResult, SentimentResult, ThemesResult
from src.generators import FAQEntry, LinkedInPost, NewsletterSection
from src.generators import testimonials as tm
from src.oci import AssetStorage, StorageSettings
from src.orchestration import compile_pipeline, initial_state, summarize_state
from src.utils.mock_llm import MOCK_BUILDERS, mock_components, mock_structured_llm


@pytest.mark.parametrize("schema", list(MOCK_BUILDERS))
def test_every_mock_output_validates_against_its_schema(schema: type) -> None:
    prompt = StringPromptValue(
        text='System: brief\nHuman: - id: 1 | discord #help | author: ana\n  text: """Any idea why pip fails? 😕"""'
    )
    payload = mock_structured_llm(schema).invoke(prompt)
    schema.model_validate(payload)


def test_sentiment_heuristics() -> None:
    run = mock_structured_llm(SentimentResult).invoke
    assert SentimentResult.model_validate(run("Huge shout-out, thanks all! 🎉")).label == "positive"
    assert SentimentResult.model_validate(run("Honestly frustrated, the lab is broken")).label == "negative"
    assert SentimentResult.model_validate(run("Meeting at 5")).label == "neutral"
    assert SentimentResult.model_validate(run("Thanks, but it is still broken")).label == "mixed"


def test_theme_and_relevance_heuristics() -> None:
    themes = ThemesResult.model_validate(
        mock_structured_llm(ThemesResult).invoke("Getting ModuleNotFoundError: No module named 'langgraph'?")
    )
    assert themes.primary_category == "technical_question"
    assert "langgraph" in themes.technologies
    assert themes.topics == ["missing module in venv"]

    relevance = RelevanceResult.model_validate(
        mock_structured_llm(RelevanceResult).invoke("Huge shout-out 🎉 I finally landed the job, thanks!")
    )
    assert relevance.score >= 0.7
    assert relevance.suggested_channels == ["linkedin", "testimonial"]


def test_generator_mocks_reuse_the_brief_text() -> None:
    brief = (
        'Human: - id: 1\n  text: """How do I fix the OCI compartment error?"""\n- id: 2\n  text: """Set the OCID."""'
    )
    faq = FAQEntry.model_validate(mock_structured_llm(FAQEntry).invoke(brief))
    assert faq.question.startswith("How do I fix") and faq.answer == "Set the OCID."
    post = LinkedInPost.model_validate(mock_structured_llm(LinkedInPost).invoke(brief))
    assert "How do I fix the OCI compartment error" in post.body
    section = NewsletterSection.model_validate(mock_structured_llm(NewsletterSection).invoke(brief))
    assert len(section.highlights) == 2
    quote = tm.Testimonial.model_validate(mock_structured_llm(tm.Testimonial).invoke("@bob Great course, thanks!"))
    assert "@bob" not in quote.quote


def test_full_pipeline_runs_offline_with_mocks(sample_path: Path, tmp_path: Path) -> None:
    storage = AssetStorage(StorageSettings(force_local=True, local_dir=tmp_path))
    graph = compile_pipeline(mock_components(storage=storage))
    final = graph.invoke(initial_state("json", sample_path, run_id="mock", options={"force_generate": True}))
    summary = summarize_state(final)
    assert summary["validated"] == 5 and summary["analyzed"] == 5 and summary["errors"] == []
    assert {a["channel"] for a in summary["generated_assets"]} >= {"linkedin", "newsletter", "testimonial"}
    assert (tmp_path / "analysis").exists() and (tmp_path / "decisions").exists()
