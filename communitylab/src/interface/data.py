"""Read side of the Streamlit UI: finalized pipeline outputs from storage.

Everything the views render comes from the same :class:`~src.oci.AssetStorage`
the LangGraph ``store`` node writes to, so the UI works identically against
the OCI bucket or the local ``data/`` fallback::

    decisions/<stamp>-<run_id>-report.json     -> DecisionReport
    analysis/<stamp>-<run_id>-enriched.json    -> list[EnrichedMessage]
    <channel>/<stamp>-<slug>.json              -> generated assets
    curation/status.json                        -> approve/reject verdicts

This module has no Streamlit dependency so it can be unit-tested directly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from src.analysis import EnrichedMessage
from src.decisions import DecisionReport
from src.oci import AssetStorage
from src.orchestration.router import (
    ALL_CHANNELS,
    PipelineComponents,
    PipelineOptions,
    compile_pipeline,
    initial_state,
    summarize_state,
)
from src.prompts.system_prompts import Channel
from src.utils.mock_llm import mock_components

logger = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "messages.json"
CURATION_OBJECT = "curation/status.json"

CurationStatus = Literal["pending", "approved", "rejected"]

_REPORT_RE = re.compile(r"^decisions/(?P<stamp>\d{8}-\d{6})-(?P<run_id>.+)-report\.json$")


@dataclass(frozen=True)
class RunRef:
    """One stored pipeline run, identified by its report object."""

    stamp: str
    run_id: str

    @property
    def label(self) -> str:
        dt = datetime.strptime(self.stamp, "%Y%m%d-%H%M%S")
        return f"{dt:%Y-%m-%d %H:%M:%S} UTC · {self.run_id}"

    @property
    def report_object(self) -> str:
        return f"decisions/{self.stamp}-{self.run_id}-report.json"

    @property
    def summary_object(self) -> str:
        return f"decisions/{self.stamp}-{self.run_id}-summary.md"

    @property
    def enriched_object(self) -> str:
        return f"analysis/{self.stamp}-{self.run_id}-enriched.json"


@dataclass(frozen=True)
class StoredAsset:
    """A generated asset as read back from storage, plus its curation verdict."""

    object_name: str
    channel: Channel
    title: str
    markdown: str
    generated_at: datetime
    language: str = "en"
    content: dict[str, Any] = field(default_factory=dict)
    source_message_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    status: CurationStatus = "pending"

    @classmethod
    def from_record(
        cls, object_name: str, record: Mapping[str, Any], status: CurationStatus = "pending"
    ) -> StoredAsset:
        generated_at = datetime.fromisoformat(str(record.get("generated_at", datetime.now(UTC).isoformat())))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        return cls(
            object_name=object_name,
            channel=cast(Channel, str(record.get("channel", object_name.split("/")[0]))),
            title=str(record.get("title", object_name)),
            markdown=str(record.get("markdown", "")),
            generated_at=generated_at,
            language=str(record.get("language", "en")),
            content=dict(record.get("content") or {}),
            source_message_ids=tuple(str(x) for x in record.get("source_message_ids", ())),
            metadata=dict(record.get("metadata") or {}),
            status=status,
        )


class OutputsRepository:
    """Typed accessors over the objects the pipeline stores.

    Args:
        storage: Where to read from (and where curation verdicts are written).
    """

    def __init__(self, storage: AssetStorage | None = None) -> None:
        self.storage = storage if storage is not None else AssetStorage()

    # ----------------------------------------------------------------- helpers
    @property
    def backend(self) -> str:
        """``"oci"`` or ``"local"`` depending on what the storage resolved to."""
        return "local" if self.storage.client is None else "oci"

    @property
    def location(self) -> str:
        """Human-readable description of where outputs live."""
        if self.backend == "local":
            return str(self.storage.settings.local_dir)
        return f"oci://{self.storage.settings.bucket}"

    def _names(self, prefix: str) -> list[str]:
        """Object names under ``prefix`` with the configured storage prefix stripped."""
        configured = self.storage.settings.prefix.strip("/")
        names = self.storage.list_objects(prefix)
        if configured:
            names = [n[len(configured) + 1 :] if n.startswith(configured + "/") else n for n in names]
        return names

    def _read_json(self, object_name: str) -> Any:
        return json.loads(self.storage.download(object_name).decode("utf-8"))

    # -------------------------------------------------------------------- runs
    def list_runs(self) -> list[RunRef]:
        """Stored runs, newest first."""
        refs: list[RunRef] = []
        for name in self._names("decisions/"):
            match = _REPORT_RE.match(name)
            if match:
                refs.append(RunRef(stamp=match["stamp"], run_id=match["run_id"]))
        return sorted(refs, key=lambda r: (r.stamp, r.run_id), reverse=True)

    def latest_run(self) -> RunRef | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    def load_report(self, run: RunRef) -> DecisionReport:
        """The decision engine output of ``run``."""
        return DecisionReport.model_validate(self._read_json(run.report_object))

    def load_summary(self, run: RunRef) -> str:
        """Executive summary Markdown of ``run`` (empty string if missing)."""
        try:
            return self.storage.download(run.summary_object).decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - OCI raises ServiceError, local raises OSError
            logger.debug("No summary for %s: %s", run.run_id, exc)
            return ""

    def load_enriched(self, run: RunRef) -> list[EnrichedMessage]:
        """Enriched messages analysed in ``run`` (empty if the run predates analysis storage)."""
        try:
            records = self._read_json(run.enriched_object)
        except Exception as exc:  # noqa: BLE001 - OCI raises ServiceError
            logger.debug("No enriched analysis for %s: %s", run.run_id, exc)
            return []
        items: list[EnrichedMessage] = []
        for record in records:
            try:
                items.append(EnrichedMessage.from_record(record))
            except Exception as exc:  # noqa: BLE001 - skip corrupt rows, keep the rest
                logger.warning("Skipping unreadable enriched record: %s", exc)
        return items

    # ------------------------------------------------------------------ assets
    def load_assets(self, channels: Sequence[str] = ALL_CHANNELS) -> list[StoredAsset]:
        """All stored assets for ``channels``, newest first, with curation status attached."""
        statuses = self.load_curation()
        assets: list[StoredAsset] = []
        for channel in channels:
            for name in self._names(f"{channel}/"):
                if not name.endswith(".json"):
                    continue
                try:
                    record = self._read_json(name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping unreadable asset %s: %s", name, exc)
                    continue
                status = cast(CurationStatus, statuses.get(name, {}).get("status", "pending"))
                assets.append(StoredAsset.from_record(name, record, status))
        return sorted(assets, key=lambda a: a.generated_at, reverse=True)

    # ---------------------------------------------------------------- curation
    def load_curation(self) -> dict[str, dict[str, Any]]:
        """``{object_name: {"status": ..., "decided_at": ..., "note": ...}}``."""
        if CURATION_OBJECT not in self._names("curation/"):
            return {}
        try:
            data = self._read_json(CURATION_OBJECT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Curation file unreadable, starting fresh: %s", exc)
            return {}
        return {str(k): dict(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def set_status(self, object_name: str, status: CurationStatus, *, note: str = "") -> dict[str, dict[str, Any]]:
        """Record an approve/reject verdict for an asset and persist the curation file."""
        statuses = self.load_curation()
        if status == "pending":
            statuses.pop(object_name, None)
        else:
            statuses[object_name] = {
                "status": status,
                "decided_at": datetime.now(UTC).isoformat(),
                "note": note,
            }
        self.storage.upload_json(statuses, CURATION_OBJECT)
        return statuses

    # ---------------------------------------------------------------- pipeline
    def run_pipeline(
        self,
        target: str | Path = SAMPLE_PATH,
        *,
        demo: bool = False,
        options: PipelineOptions | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the LangGraph pipeline on a JSON file and store outputs in this repository.

        Args:
            target: Path to a JSON export (defaults to ``data/sample/messages.json``).
            demo: Use the keyword mocks from :mod:`src.utils.mock_llm` instead of Groq.
            options: Generation options forwarded to the graph.
            run_id: Explicit run identifier (random when ``None``).

        Returns:
            :func:`summarize_state` of the final state.
        """
        components = mock_components(storage=self.storage) if demo else PipelineComponents.default(storage=self.storage)
        graph = compile_pipeline(components)
        state = initial_state("json", str(target), options=options, run_id=run_id)
        return summarize_state(graph.invoke(state))


# --------------------------------------------------------------------------- #
# DataFrame helpers shared by the views (pure functions, easy to test)
# --------------------------------------------------------------------------- #
def enriched_frame(items: Sequence[EnrichedMessage]) -> pd.DataFrame:
    """One row per analysed message with the columns the dashboard charts need."""
    rows = []
    for item in items:
        rows.append(
            {
                "message_id": item.message.message_id,
                "timestamp": item.message.timestamp,
                "source": item.message.source,
                "channel": item.message.channel,
                "author": item.message.author,
                "sentiment_label": item.sentiment.label if item.sentiment else None,
                "sentiment_score": item.sentiment.score if item.sentiment else None,
                "category": item.themes.primary_category if item.themes else None,
                "topics": list(item.themes.topics) if item.themes else [],
                "technologies": list(item.themes.technologies) if item.themes else [],
                "relevance_score": item.relevance.score if item.relevance else None,
                "relevance_tier": item.relevance.tier if item.relevance else None,
                "content": item.message.content,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def sentiment_trend(frame: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Mean sentiment and message volume per period (``freq`` is a pandas offset alias)."""
    if frame.empty or "sentiment_score" not in frame:
        return pd.DataFrame(columns=["period", "mean_sentiment", "messages"])
    scored = frame.dropna(subset=["sentiment_score"])
    if scored.empty:
        return pd.DataFrame(columns=["period", "mean_sentiment", "messages"])
    grouped = scored.set_index("timestamp").resample(freq)["sentiment_score"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] > 0].reset_index()
    grouped.columns = ["period", "mean_sentiment", "messages"]
    return grouped


def topic_frequencies(frame: pd.DataFrame, report: DecisionReport | None = None) -> dict[str, int]:
    """How often each topic/technology appears; recurring topics from the report get a boost."""
    counts: dict[str, int] = {}
    if not frame.empty:
        for column in ("topics", "technologies"):
            if column not in frame:
                continue
            for values in frame[column]:
                for value in values or []:
                    key = str(value).strip().lower()
                    if key:
                        counts[key] = counts.get(key, 0) + 1
    if report is not None:
        for topic in report.recurring_topics:
            key = topic.topic.strip().lower()
            counts[key] = counts.get(key, 0) + topic.occurrences
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
