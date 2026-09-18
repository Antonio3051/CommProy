"""FastAPI entry points that trigger the LangGraph pipeline.

Run locally with::

    uvicorn src.api.webhook:app --reload

Endpoints:

* ``GET  /health``                  – liveness + whether the graph is compiled.
* ``POST /webhook/{source}``        – ingest a Discord/Slack/generic payload and run the
  pipeline synchronously (``?async=true`` to queue it and return a ``run_id``).
* ``POST /runs``                    – run the pipeline over a JSON/CSV path on the server.
* ``GET  /runs/{run_id}``           – status/summary of a queued run.

The compiled graph is created lazily on first use so importing this module
never requires Groq or OCI credentials; tests inject fakes via
:func:`create_app`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.orchestration.router import (
    GeneratorChannel,
    PipelineOptions,
    State,
    compile_pipeline,
    initial_state,
    summarize_state,
)

logger = logging.getLogger(__name__)

WebhookSource = Literal["discord", "slack", "webhook"]
RunStatus = Literal["queued", "running", "completed", "failed"]
GraphFactory = Callable[[], CompiledStateGraph]


class RunOptions(BaseModel):
    """Subset of :class:`PipelineOptions` accepted over HTTP."""

    language: str = "en"
    community_name: str = "the community"
    channels: list[GeneratorChannel] = Field(default_factory=lambda: ["linkedin", "newsletter", "faq", "testimonial"])
    max_posts: int = Field(default=3, ge=0, le=20)
    force_generate: bool = False
    skip_generate: bool = False
    store_decisions: bool = True

    def to_pipeline_options(self) -> PipelineOptions:
        """Convert to the TypedDict the graph nodes read."""
        return {
            "language": self.language,
            "community_name": self.community_name,
            "channels": tuple(self.channels),
            "max_posts": self.max_posts,
            "force_generate": self.force_generate,
            "skip_generate": self.skip_generate,
            "store_decisions": self.store_decisions,
        }


class WebhookRequest(BaseModel):
    """Body of ``POST /webhook/{source}``."""

    payload: Any = Field(description="Raw Discord/Slack export or a list of records.")
    channel: str | None = Field(default=None, description="Channel name to stamp on records lacking one.")
    options: RunOptions = Field(default_factory=RunOptions)


class FileRunRequest(BaseModel):
    """Body of ``POST /runs``: run over a file already on the server."""

    source: Literal["json", "csv"] = "json"
    path: str
    options: RunOptions = Field(default_factory=RunOptions)


class RunResponse(BaseModel):
    """Status of one pipeline run."""

    run_id: str
    status: RunStatus
    submitted_at: datetime
    summary: dict[str, Any] | None = None
    error: str | None = None


class RunRegistry:
    """In-memory store of runs (swap for Redis/DB when scaling out)."""

    def __init__(self) -> None:
        self._runs: dict[str, RunResponse] = {}

    def create(self, run_id: str) -> RunResponse:
        run = RunResponse(run_id=run_id, status="queued", submitted_at=datetime.now(UTC))
        self._runs[run_id] = run
        return run

    def update(self, run_id: str, **changes: Any) -> RunResponse:
        run = self._runs[run_id].model_copy(update=changes)
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunResponse | None:
        return self._runs.get(run_id)


def create_app(graph_factory: GraphFactory = compile_pipeline) -> FastAPI:
    """Build the FastAPI app; ``graph_factory`` is called once, on first request."""
    app = FastAPI(title="CommunityLab API", version="0.1.0")
    registry = RunRegistry()
    cache: dict[str, CompiledStateGraph] = {}
    app.state.registry = registry

    def graph() -> CompiledStateGraph:
        if "graph" not in cache:
            cache["graph"] = graph_factory()
        return cache["graph"]

    def execute(state: State) -> RunResponse:
        run_id = state["run_id"]
        registry.update(run_id, status="running")
        try:
            final = graph().invoke(state)
        except Exception as exc:  # noqa: BLE001 - reported through the run status
            logger.exception("Run %s failed", run_id)
            return registry.update(run_id, status="failed", error=str(exc))
        return registry.update(run_id, status="completed", summary=summarize_state(final))

    async def run_or_queue(state: State, background: BackgroundTasks, queue: bool) -> RunResponse:
        run = registry.create(state["run_id"])
        if queue:
            background.add_task(execute, state)
            return run
        return await asyncio.to_thread(execute, state)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "graph_compiled": "graph" in cache, "time": datetime.now(UTC).isoformat()}

    @app.post("/webhook/{source}", response_model=RunResponse)
    async def webhook(
        source: WebhookSource,
        body: WebhookRequest,
        background: BackgroundTasks,
        queue: bool = Query(default=False, alias="async", description="Return immediately with a run_id."),
    ) -> RunResponse:
        loader_kwargs = {"channel": body.channel} if body.channel else {}
        if source == "webhook":
            loader_kwargs = {}
        state = initial_state(
            source, body.payload, loader_kwargs=loader_kwargs, options=body.options.to_pipeline_options()
        )
        return await run_or_queue(state, background, queue)

    @app.post("/runs", response_model=RunResponse)
    async def run_file(
        body: FileRunRequest,
        background: BackgroundTasks,
        queue: bool = Query(default=False, alias="async"),
    ) -> RunResponse:
        state = initial_state(body.source, body.path, options=body.options.to_pipeline_options())
        return await run_or_queue(state, background, queue)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str) -> RunResponse:
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id!r}")
        return run

    return app


app = create_app()
