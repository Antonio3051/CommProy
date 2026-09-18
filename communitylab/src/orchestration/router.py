"""LangGraph pipeline wiring ingestion → analysis → decisions → generators → storage.

::

    START ─▶ ingest ─┬─(no valid rows)─▶ END
                     └─▶ analyze ─▶ decide ─┬─(nothing worth publishing)─▶ store ─▶ END
                                            └─▶ generate ─▶ store ─▶ END

The graph is built from a :class:`PipelineComponents` bundle so every heavy
dependency (Groq-backed consolidator/generators, OCI storage) can be replaced
with fakes in tests or CLI dry runs::

    graph = compile_pipeline()                       # real components (needs GROQ_API_KEY)
    result = graph.invoke(initial_state("json", "data/sample/messages.json"))
    result["decisions"].executive_summary
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.analysis import Consolidator, EnrichedMessage
from src.decisions import DecisionEngine, DecisionReport
from src.generators import (
    FAQGenerator,
    GeneratedAsset,
    LinkedInGenerator,
    NewsletterGenerator,
    TestimonialGenerator,
    select_highlights,
    select_success_stories,
    select_testimonial_candidates,
)
from src.ingest import RawRecord, RowError, SourceName, load, normalize, validate
from src.ingest.validator import CommunityMessage
from src.oci import AssetStorage, UploadResult
from src.utils.llm_clients import LLMSettings

logger = logging.getLogger(__name__)

GeneratorChannel = Literal["linkedin", "newsletter", "faq", "testimonial"]
ALL_CHANNELS: tuple[GeneratorChannel, ...] = ("linkedin", "newsletter", "faq", "testimonial")

NODE_INGEST = "ingest"
NODE_ANALYZE = "analyze"
NODE_DECIDE = "decide"
NODE_GENERATE = "generate"
NODE_STORE = "store"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class PipelineOptions(TypedDict, total=False):
    """Per-run knobs read by the nodes (all optional)."""

    language: str
    community_name: str
    channels: Sequence[GeneratorChannel]
    max_posts: int
    force_generate: bool
    skip_generate: bool
    store_decisions: bool
    reference_time: datetime


class State(TypedDict, total=False):
    """Everything the graph carries between nodes.

    Input keys are set by :func:`initial_state`; each node fills the key it
    owns. ``errors`` is a reducer channel, so nodes append to it instead of
    overwriting.
    """

    run_id: str
    started_at: datetime
    source: SourceName
    target: Any
    loader_kwargs: dict[str, Any]
    options: PipelineOptions

    raw_data: list[RawRecord]
    validated_data: list[CommunityMessage]
    validation_errors: list[RowError]
    analysis_results: list[EnrichedMessage]
    decisions: DecisionReport | None
    generated_assets: list[GeneratedAsset]
    storage_results: list[UploadResult]
    errors: Annotated[list[str], operator.add]
    finished_at: datetime


def initial_state(
    source: SourceName,
    target: Any,
    *,
    raw_data: Sequence[Mapping[str, Any]] | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
    options: PipelineOptions | None = None,
    run_id: str | None = None,
) -> State:
    """Build the input state for one run.

    Args:
        source: Loader to use (``json``/``csv`` take a path, the rest a payload).
        target: Path or payload handed to :func:`src.ingest.load`.
        raw_data: Already-loaded records; when given the loader is skipped.
        loader_kwargs: Extra arguments for the loader (``channel=`` …).
        options: Generation/storage knobs, see :class:`PipelineOptions`.
        run_id: Identifier used in stored object names (default: random).
    """
    state: State = {
        "run_id": run_id or uuid4().hex[:12],
        "started_at": datetime.now(UTC),
        "source": source,
        "target": target,
        "loader_kwargs": dict(loader_kwargs or {}),
        "options": dict(options or {}),  # type: ignore[typeddict-item]
        "errors": [],
    }
    if raw_data is not None:
        state["raw_data"] = [dict(r) for r in raw_data]
    return state


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
@dataclass
class Generators:
    """The four content generators (any may be ``None`` to disable that channel)."""

    linkedin: LinkedInGenerator | None = None
    newsletter: NewsletterGenerator | None = None
    faq: FAQGenerator | None = None
    testimonial: TestimonialGenerator | None = None

    @classmethod
    def default(cls, settings: LLMSettings | None = None) -> Generators:
        """Groq-backed generators for every channel."""
        return cls(
            linkedin=LinkedInGenerator(settings=settings),
            newsletter=NewsletterGenerator(settings=settings),
            faq=FAQGenerator(settings=settings),
            testimonial=TestimonialGenerator(settings=settings),
        )


@dataclass
class PipelineComponents:
    """Dependencies used by the nodes; swap any of them for fakes."""

    consolidator: Consolidator
    engine: DecisionEngine = field(default_factory=DecisionEngine)
    generators: Generators = field(default_factory=Generators)
    storage: AssetStorage = field(default_factory=AssetStorage)
    max_concurrency: int | None = 4

    @classmethod
    def default(cls, settings: LLMSettings | None = None, *, storage: AssetStorage | None = None) -> PipelineComponents:
        """Production wiring: Groq analyzers + generators, OCI storage with local fallback."""
        return cls(
            consolidator=Consolidator(settings=settings),
            generators=Generators.default(settings),
            storage=storage if storage is not None else AssetStorage(),
        )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def make_ingest_node(components: PipelineComponents) -> Any:  # noqa: ARG001 - symmetry with the other factories
    """Node 1: Loader → Normalizer → Validator."""

    def ingest(state: State) -> State:
        errors: list[str] = []
        raw = state.get("raw_data")
        if raw is None:
            try:
                raw = load(state["source"], state["target"], **state.get("loader_kwargs", {}))
            except Exception as exc:  # noqa: BLE001 - surfaced through state["errors"]
                logger.error("Ingestion failed: %s", exc)
                return {"raw_data": [], "validated_data": [], "validation_errors": [], "errors": [f"ingest: {exc}"]}
        frame = normalize(raw, default_source=str(state.get("source", "webhook")))
        result = validate(frame)
        if result.errors:
            errors.append(f"ingest: {len(result.errors)} of {result.total} rows failed validation")
        logger.info("Ingested %d raw records -> %d valid messages", len(raw), len(result.valid))
        return {
            "raw_data": list(raw),
            "validated_data": list(result.valid),
            "validation_errors": list(result.errors),
            "errors": errors,
        }

    return ingest


def make_analyze_node(components: PipelineComponents) -> Any:
    """Node 2: Sentiment + Themes + Relevance → Consolidator."""

    def analyze(state: State) -> State:
        messages = state.get("validated_data", [])
        enriched = components.consolidator.enrich_many(messages, max_concurrency=components.max_concurrency)
        failed = [e.message_id for e in enriched if e.errors]
        errors = [f"analyze: partial analysis for {len(failed)} message(s)"] if failed else []
        logger.info("Analysed %d messages (%d partial)", len(enriched), len(failed))
        return {"analysis_results": enriched, "errors": errors}

    return analyze


def make_decide_node(components: PipelineComponents) -> Any:
    """Node 3: Decision Engine (R1-R3, notifications, executive summary)."""

    def decide(state: State) -> State:
        options = state.get("options", {})
        report = components.engine.run(state.get("analysis_results", []), reference_time=options.get("reference_time"))
        logger.info("Decisions: highest level %s, %d actions", report.highest_level.value, len(report.actions))
        return {"decisions": report}

    return decide


def make_generate_node(components: PipelineComponents) -> Any:
    """Node 4: content generators, one per channel requested in ``options.channels``."""

    def generate(state: State) -> State:
        enriched = state.get("analysis_results", [])
        report = state.get("decisions")
        options = state.get("options", {})
        language = options.get("language", "en")
        channels = tuple(options.get("channels", ALL_CHANNELS))
        limit = options.get("max_posts", 3)
        gens = components.generators
        assets: list[GeneratedAsset] = []
        errors: list[str] = []

        if "linkedin" in channels and gens.linkedin is not None:
            assets += gens.linkedin.generate_many(enriched, language=language, limit=limit)
        if "testimonial" in channels and gens.testimonial is not None:
            assets += gens.testimonial.generate_many(enriched, language=language, limit=limit)
        if "faq" in channels and gens.faq is not None and report is not None:
            assets += gens.faq.generate_many(report.recurring_topics, enriched, language=language)
        if "newsletter" in channels and gens.newsletter is not None and select_highlights(enriched):
            try:
                assets.append(
                    gens.newsletter.generate(
                        enriched, language=language, community_name=options.get("community_name", "the community")
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a failed newsletter must not sink the run
                logger.warning("Newsletter generation failed: %s", exc)
                errors.append(f"generate: newsletter failed ({exc})")
        logger.info("Generated %d asset(s) for channels %s", len(assets), ", ".join(channels))
        return {"generated_assets": assets, "errors": errors}

    return generate


def make_store_node(components: PipelineComponents) -> Any:
    """Node 5: persist generated assets and the decision report (OCI or ``data/`` fallback)."""

    def store(state: State) -> State:
        options = state.get("options", {})
        results: list[UploadResult] = []
        errors: list[str] = []
        storage = components.storage
        try:
            results += storage.upload_assets(state.get("generated_assets", []))
            report = state.get("decisions")
            if report is not None and options.get("store_decisions", True):
                stamp = report.generated_at.strftime("%Y%m%d-%H%M%S")
                base = f"decisions/{stamp}-{state.get('run_id', 'run')}"
                results.append(storage.upload_json(report.model_dump(mode="json"), f"{base}-report.json"))
                results.append(storage.upload_text(report.executive_summary, f"{base}-summary.md"))
        except Exception as exc:  # noqa: BLE001 - storage must never crash the graph
            logger.error("Storage failed: %s", exc)
            errors.append(f"store: {exc}")
        local = sum(1 for r in results if r.backend == "local")
        logger.info("Stored %d object(s) (%d via local fallback)", len(results), local)
        return {"storage_results": results, "errors": errors, "finished_at": datetime.now(UTC)}

    return store


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_ingest(state: State) -> str:
    """Skip the LLM stages when nothing validated."""
    return NODE_ANALYZE if state.get("validated_data") else END


def has_publishable_content(state: State) -> bool:
    """Whether the Decision Engine output / analysis justifies calling the generators.

    True when at least one of: an R2 topic flagged for a FAQ, a success story
    or testimonial candidate, or enough medium-relevance messages for a
    newsletter highlight.
    """
    enriched = state.get("analysis_results", [])
    report = state.get("decisions")
    if report is not None and any(t.action in ("faq", "faq_and_mentorship") for t in report.recurring_topics):
        return True
    if select_success_stories(enriched) or select_testimonial_candidates(enriched):
        return True
    return bool(select_highlights(enriched))


def route_after_decide(state: State) -> str:
    """Generators run only when there is something worth publishing (or when forced)."""
    options = state.get("options", {})
    if options.get("skip_generate"):
        return NODE_STORE
    if options.get("force_generate"):
        return NODE_GENERATE
    return NODE_GENERATE if has_publishable_content(state) else NODE_STORE


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(components: PipelineComponents) -> StateGraph:
    """Assemble the (uncompiled) ``StateGraph`` with the five nodes and their edges."""
    graph: StateGraph = StateGraph(State)
    graph.add_node(NODE_INGEST, make_ingest_node(components))
    graph.add_node(NODE_ANALYZE, make_analyze_node(components))
    graph.add_node(NODE_DECIDE, make_decide_node(components))
    graph.add_node(NODE_GENERATE, make_generate_node(components))
    graph.add_node(NODE_STORE, make_store_node(components))

    graph.add_edge(START, NODE_INGEST)
    graph.add_conditional_edges(NODE_INGEST, route_after_ingest, {NODE_ANALYZE: NODE_ANALYZE, END: END})
    graph.add_edge(NODE_ANALYZE, NODE_DECIDE)
    graph.add_conditional_edges(NODE_DECIDE, route_after_decide, {NODE_GENERATE: NODE_GENERATE, NODE_STORE: NODE_STORE})
    graph.add_edge(NODE_GENERATE, NODE_STORE)
    graph.add_edge(NODE_STORE, END)
    return graph


def compile_pipeline(
    components: PipelineComponents | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the pipeline; defaults to production components (needs ``GROQ_API_KEY``)."""
    return build_graph(components or PipelineComponents.default()).compile(checkpointer=checkpointer)


def run_pipeline(
    source: SourceName,
    target: Any,
    *,
    components: PipelineComponents | None = None,
    options: PipelineOptions | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> State:
    """One-shot helper: compile, invoke, return the final state."""
    graph = compile_pipeline(components)
    return graph.invoke(initial_state(source, target, loader_kwargs=loader_kwargs, options=options, run_id=run_id))


def summarize_state(state: State) -> dict[str, Any]:
    """Compact, JSON-friendly view of a finished run (for APIs, bots and logs)."""
    report = state.get("decisions")
    return {
        "run_id": state.get("run_id"),
        "source": state.get("source"),
        "raw_records": len(state.get("raw_data", [])),
        "validated": len(state.get("validated_data", [])),
        "validation_errors": len(state.get("validation_errors", [])),
        "analyzed": len(state.get("analysis_results", [])),
        "highest_level": report.highest_level.value if report is not None else None,
        "actions": len(report.actions) if report is not None else 0,
        "executive_summary": report.executive_summary if report is not None else "",
        "generated_assets": [
            {"channel": a.channel, "title": a.title, "sources": list(a.source_message_ids)}
            for a in state.get("generated_assets", [])
        ],
        "stored": [{"backend": r.backend, "location": r.location} for r in state.get("storage_results", [])],
        "errors": list(state.get("errors", [])),
        "started_at": state["started_at"].isoformat() if "started_at" in state else None,
        "finished_at": state["finished_at"].isoformat() if "finished_at" in state else None,
    }
