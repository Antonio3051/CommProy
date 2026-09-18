"""Tests for the LangGraph pipeline in ``src/orchestration/router.py``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph
from src.analysis import Consolidator, EnrichedMessage, RelevanceResult, SentimentResult, ThemesResult
from src.decisions import DecisionEngine, DecisionReport
from src.generators import FAQEntry, FAQGenerator, GeneratedAsset, LinkedInGenerator, NewsletterGenerator
from src.generators import testimonials as tm
from src.ingest.validator import CommunityMessage
from src.oci import AssetStorage, StorageSettings
from src.orchestration import (
    ALL_CHANNELS,
    NODE_ANALYZE,
    NODE_DECIDE,
    NODE_GENERATE,
    NODE_INGEST,
    NODE_STORE,
    Generators,
    PipelineComponents,
    State,
    build_graph,
    compile_pipeline,
    has_publishable_content,
    initial_state,
    route_after_decide,
    route_after_ingest,
    run_pipeline,
    summarize_state,
)
from src.orchestration.router import make_generate_node, make_ingest_node, make_store_node

T0 = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)

POSITIVE = {"label": "positive", "score": 0.8, "confidence": 0.9}
NEUTRAL = {"label": "neutral", "score": 0.0, "confidence": 0.9}
SUCCESS = {"primary_category": "success_story", "topics": ["deployment"], "technologies": ["python"]}
CHATTER = {"primary_category": "community_chatter", "topics": []}
HIGH = {"score": 0.7, "suggested_channels": ["linkedin"]}
NOISE = {"score": 0.05, "suggested_channels": []}

LINKEDIN = {"hook": "Hook!", "body": "Body text.", "hashtags": ["#a"]}
NEWSLETTER = {"title": "This week", "intro": "Hi", "highlights": ["x"], "closing": "bye"}
FAQ = {"question": "Q?", "answer": "A."}
TESTIMONIAL = {"quote": "Great community, learnt a lot.", "context": "", "attribution": "Member"}


def fake_llm(payload: dict[str, Any], calls: list[Any] | None = None) -> RunnableLambda:
    def _run(value: Any) -> dict[str, Any]:
        if calls is not None:
            calls.append(value)
        return dict(payload)

    return RunnableLambda(_run)


def consolidator(
    sentiment: dict[str, Any] = POSITIVE, themes: dict[str, Any] = SUCCESS, relevance: dict[str, Any] = HIGH
):
    return Consolidator.from_llms(
        sentiment_llm=fake_llm(sentiment), themes_llm=fake_llm(themes), relevance_llm=fake_llm(relevance)
    )


def generators(calls: dict[str, list[Any]] | None = None) -> Generators:
    calls = calls if calls is not None else {}
    for name in ALL_CHANNELS:
        calls.setdefault(name, [])
    return Generators(
        linkedin=LinkedInGenerator(llm=fake_llm(LINKEDIN, calls["linkedin"])),
        newsletter=NewsletterGenerator(llm=fake_llm(NEWSLETTER, calls["newsletter"])),
        faq=FAQGenerator(llm=fake_llm(FAQ, calls["faq"])),
        testimonial=tm.TestimonialGenerator(llm=fake_llm(TESTIMONIAL, calls["testimonial"])),
    )


def local_storage(tmp_path: Path) -> AssetStorage:
    return AssetStorage(StorageSettings(force_local=True, local_dir=tmp_path / "bucket"))


@pytest.fixture
def components(tmp_path: Path) -> PipelineComponents:
    return PipelineComponents(consolidator=consolidator(), generators=generators(), storage=local_storage(tmp_path))


@pytest.fixture
def graph(components: PipelineComponents) -> CompiledStateGraph:
    return compile_pipeline(components)


def enriched(
    message_id: str,
    content: str = "hello",
    *,
    score: float = 0.0,
    label: str = "neutral",
    category: str = "community_chatter",
    relevance: float = 0.1,
    channels: list[str] | None = None,
) -> EnrichedMessage:
    message = CommunityMessage(
        message_id=message_id,
        source="discord",
        channel="general",
        author="alice",
        author_id="a1",
        content=content,
        timestamp=T0,
    )
    return EnrichedMessage(
        message=message,
        sentiment=SentimentResult(label=label, score=score),
        themes=ThemesResult(primary_category=category),
        relevance=RelevanceResult(score=relevance, suggested_channels=channels or []),
    )


class TestState:
    def test_initial_state_keys(self):
        state = initial_state("json", "x.json", options={"language": "es"}, run_id="r1")
        assert state["run_id"] == "r1"
        assert state["source"] == "json" and state["target"] == "x.json"
        assert state["options"] == {"language": "es"}
        assert state["errors"] == [] and "raw_data" not in state
        assert state["started_at"].tzinfo is not None

    def test_initial_state_with_raw_data_skips_loader(self):
        state = initial_state("webhook", None, raw_data=[{"id": "1", "content": "hi"}])
        assert state["raw_data"] == [{"id": "1", "content": "hi"}]

    def test_state_declares_required_channels(self):
        for key in ("raw_data", "validated_data", "analysis_results", "decisions", "generated_assets"):
            assert key in State.__annotations__


class TestGraphShape:
    def test_nodes_and_edges(self, components: PipelineComponents):
        graph = build_graph(components).compile().get_graph()
        assert {NODE_INGEST, NODE_ANALYZE, NODE_DECIDE, NODE_GENERATE, NODE_STORE} <= set(graph.nodes)
        edges = {(e.source, e.target, e.conditional) for e in graph.edges}
        assert ("__start__", NODE_INGEST, False) in edges
        assert (NODE_INGEST, NODE_ANALYZE, True) in edges and (NODE_INGEST, END, True) in edges
        assert (NODE_ANALYZE, NODE_DECIDE, False) in edges
        assert (NODE_DECIDE, NODE_GENERATE, True) in edges and (NODE_DECIDE, NODE_STORE, True) in edges
        assert (NODE_GENERATE, NODE_STORE, False) in edges
        assert (NODE_STORE, END, False) in edges

    def test_compile_with_checkpointer(self, components: PipelineComponents, sample_path: Path):
        graph = compile_pipeline(components, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        final = graph.invoke(initial_state("json", sample_path), config=config)
        assert final["decisions"] is not None
        assert graph.get_state(config).values["run_id"] == final["run_id"]


class TestRouting:
    def test_route_after_ingest(self):
        assert route_after_ingest({"validated_data": []}) == END
        assert route_after_ingest({}) == END
        assert route_after_ingest({"validated_data": [object()]}) == NODE_ANALYZE  # type: ignore[list-item]

    def test_no_publishable_content_skips_generators(self):
        state: State = {"analysis_results": [enriched("m1")], "decisions": DecisionEngine().run([])}
        assert not has_publishable_content(state)
        assert route_after_decide(state) == NODE_STORE

    def test_success_story_triggers_generators(self):
        items = [enriched("m1", score=0.9, label="positive", category="success_story", relevance=0.7)]
        state: State = {"analysis_results": items, "decisions": DecisionEngine().run(items)}
        assert has_publishable_content(state)
        assert route_after_decide(state) == NODE_GENERATE

    def test_recurring_faq_topic_triggers_generators(self):
        items = [
            EnrichedMessage(
                message=CommunityMessage(
                    message_id=f"q{i}",
                    source="discord",
                    channel="help",
                    author=f"user{i}",
                    author_id=f"u{i}",
                    content="ModuleNotFoundError: No module named 'langgraph' — how do I fix this?",
                    timestamp=T0 + timedelta(hours=i),
                ),
                sentiment=SentimentResult(label="neutral", score=0.0),
                themes=ThemesResult(primary_category="technical_question", topics=["venv"]),
                relevance=RelevanceResult(score=0.5, suggested_channels=["faq"]),
            )
            for i in range(3)
        ]
        report = DecisionEngine().run(items)
        assert report.recurring_topics and report.recurring_topics[0].action in ("faq", "faq_and_mentorship")
        assert route_after_decide({"analysis_results": items, "decisions": report}) == NODE_GENERATE

    def test_force_and_skip_flags(self):
        state: State = {"analysis_results": [], "decisions": None}
        assert route_after_decide({**state, "options": {"force_generate": True}}) == NODE_GENERATE
        state["analysis_results"] = [
            enriched("m1", score=0.9, label="positive", category="success_story", relevance=0.8)
        ]
        assert route_after_decide({**state, "options": {"skip_generate": True}}) == NODE_STORE


class TestNodes:
    def test_ingest_node_loads_normalizes_validates(self, components: PipelineComponents, sample_path: Path):
        out = make_ingest_node(components)(initial_state("json", sample_path))
        assert len(out["raw_data"]) == 5 and len(out["validated_data"]) == 5
        assert out["validation_errors"] == [] and out["errors"] == []
        assert all(isinstance(m, CommunityMessage) for m in out["validated_data"])

    def test_ingest_node_uses_preloaded_raw_data(self, components: PipelineComponents, sample_payload):
        out = make_ingest_node(components)(initial_state("json", None, raw_data=sample_payload))
        assert len(out["validated_data"]) == 5

    def test_ingest_node_reports_loader_failure(self, components: PipelineComponents, tmp_path: Path):
        out = make_ingest_node(components)(initial_state("json", tmp_path / "missing.json"))
        assert out["validated_data"] == [] and out["errors"][0].startswith("ingest:")

    def test_ingest_node_counts_invalid_rows(self, components: PipelineComponents):
        raw = [
            {"id": "1", "content": "fine", "author": "a", "timestamp": "2026-09-14T12:00:00Z"},
            {"id": "2", "content": "no timestamp", "author": "b"},
        ]
        out = make_ingest_node(components)(initial_state("webhook", None, raw_data=raw))
        assert len(out["validated_data"]) == 1 and len(out["validation_errors"]) == 1
        assert out["errors"] == ["ingest: 1 of 2 rows failed validation"]

    def test_generate_node_respects_channels_and_limit(self, components: PipelineComponents):
        calls: dict[str, list[Any]] = {}
        components.generators = generators(calls)
        items = [
            enriched(f"m{i}", score=0.9, label="positive", category="success_story", relevance=0.8) for i in range(4)
        ]
        state: State = {
            "analysis_results": items,
            "decisions": DecisionEngine().run(items),
            "options": {"channels": ["linkedin"], "max_posts": 2},
        }
        out = make_generate_node(components)(state)
        assert [a.channel for a in out["generated_assets"]] == ["linkedin", "linkedin"]
        assert len(calls["linkedin"]) == 2 and not calls["newsletter"] and not calls["testimonial"]

    def test_generate_node_disabled_generator_is_skipped(self, components: PipelineComponents):
        components.generators = Generators(newsletter=generators().newsletter)
        items = [enriched("m1", score=0.9, label="positive", category="success_story", relevance=0.8)]
        out = make_generate_node(components)({"analysis_results": items, "decisions": None, "options": {}})
        assert [a.channel for a in out["generated_assets"]] == ["newsletter"]

    def test_generate_node_newsletter_failure_is_recorded(self, components: PipelineComponents):
        def _boom(_: Any) -> Any:
            raise RuntimeError("groq down")

        components.generators = Generators(newsletter=NewsletterGenerator(llm=RunnableLambda(_boom)))
        items = [enriched("m1", score=0.9, label="positive", category="success_story", relevance=0.8)]
        out = make_generate_node(components)({"analysis_results": items, "decisions": None, "options": {}})
        assert out["generated_assets"] == [] and out["errors"][0].startswith("generate: newsletter failed")

    def test_store_node_saves_assets_and_decisions(self, components: PipelineComponents, tmp_path: Path):
        asset = GeneratedAsset(channel="faq", title="Q", content=FAQEntry(question="Q", answer="A"), markdown="# Q")
        report = DecisionEngine().run([enriched("m1")])
        out = make_store_node(components)({"generated_assets": [asset], "decisions": report, "run_id": "abc"})
        names = [r.object_name for r in out["storage_results"]]
        assert names[:2] == ["faq/" + asset.filename("md").split("/")[1], "faq/" + asset.filename("json").split("/")[1]]
        assert names[2].startswith("decisions/") and names[2].endswith("-abc-report.json")
        assert names[3].endswith("-abc-summary.md")
        assert all(r.backend == "local" for r in out["storage_results"])
        assert (tmp_path / "bucket" / names[2]).exists()
        assert out["finished_at"].tzinfo is not None

    def test_store_node_can_skip_decisions(self, components: PipelineComponents):
        report = DecisionEngine().run([])
        out = make_store_node(components)(
            {"generated_assets": [], "decisions": report, "options": {"store_decisions": False}}
        )
        assert out["storage_results"] == []


class TestEndToEnd:
    def test_full_run_generates_and_stores(self, graph: CompiledStateGraph, sample_path: Path, tmp_path: Path):
        final = graph.invoke(initial_state("json", sample_path, run_id="e2e"))
        assert len(final["raw_data"]) == 5
        assert len(final["validated_data"]) == 5
        assert len(final["analysis_results"]) == 5 and all(e.is_complete for e in final["analysis_results"])
        assert isinstance(final["decisions"], DecisionReport) and final["decisions"].message_count == 5
        channels = {a.channel for a in final["generated_assets"]}
        assert {"linkedin", "testimonial", "newsletter"} <= channels
        stored = final["storage_results"]
        assert len(stored) == 2 * len(final["generated_assets"]) + 2
        assert any(r.object_name.endswith("-e2e-report.json") for r in stored)
        assert (tmp_path / "bucket" / "decisions").is_dir()
        assert final["errors"] == []

    def test_empty_ingestion_ends_early(self, graph: CompiledStateGraph):
        final = graph.invoke(initial_state("webhook", []))
        assert final["validated_data"] == []
        assert "analysis_results" not in final and "decisions" not in final
        assert "storage_results" not in final

    def test_noise_only_skips_generators_but_stores_decisions(self, tmp_path: Path, sample_path: Path):
        comps = PipelineComponents(
            consolidator=consolidator(NEUTRAL, CHATTER, NOISE), generators=generators(), storage=local_storage(tmp_path)
        )
        final = compile_pipeline(comps).invoke(initial_state("json", sample_path, run_id="noise"))
        assert "generated_assets" not in final
        assert [r.object_name.split("/")[0] for r in final["storage_results"]] == ["decisions", "decisions"]

    def test_run_pipeline_with_discord_payload(self, components: PipelineComponents, discord_payload):
        final = run_pipeline(
            "discord",
            discord_payload,
            components=components,
            loader_kwargs={"channel": "general"},
            options={"channels": ["linkedin"], "language": "es"},
        )
        assert len(final["validated_data"]) == len(discord_payload)
        assert {a.channel for a in final["generated_assets"]} == {"linkedin"}
        assert all(a.language == "es" for a in final["generated_assets"])

    def test_analysis_errors_are_surfaced(self, tmp_path: Path, sample_path: Path):
        def _boom(_: Any) -> Any:
            raise RuntimeError("groq down")

        cons = Consolidator.from_llms(
            sentiment_llm=RunnableLambda(_boom), themes_llm=fake_llm(SUCCESS), relevance_llm=fake_llm(HIGH)
        )
        comps = PipelineComponents(consolidator=cons, generators=generators(), storage=local_storage(tmp_path))
        final = compile_pipeline(comps).invoke(initial_state("json", sample_path))
        assert final["errors"] == ["analyze: partial analysis for 5 message(s)"]
        assert all(e.sentiment is None for e in final["analysis_results"])

    def test_summarize_state(self, graph: CompiledStateGraph, sample_path: Path):
        final = graph.invoke(initial_state("json", sample_path, run_id="sum"))
        summary = summarize_state(final)
        assert summary["run_id"] == "sum" and summary["validated"] == 5 and summary["analyzed"] == 5
        assert summary["highest_level"] in {"INFO", "MEDIO", "ALTO", "CRÍTICO"}
        assert len(summary["generated_assets"]) == len(final["generated_assets"])
        assert all(item["backend"] == "local" for item in summary["stored"])
        assert summary["finished_at"] is not None

    def test_summarize_partial_state(self):
        summary = summarize_state({"run_id": "x", "errors": ["ingest: boom"]})
        assert summary["validated"] == 0 and summary["highest_level"] is None and summary["errors"] == ["ingest: boom"]
