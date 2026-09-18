"""Streamlit UI: the read-side repository, the pure view helpers and the app itself (via ``AppTest``)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import streamlit as st
from src.decisions import DecisionEngine, DecisionReport, EscalationLevel
from src.interface import alerts, curation, dashboard
from src.interface.data import (
    CURATION_OBJECT,
    OutputsRepository,
    RunRef,
    StoredAsset,
    enriched_frame,
    sentiment_trend,
    topic_frequencies,
)
from src.oci import AssetStorage, StorageSettings
from streamlit.testing.v1 import AppTest

APP_FILE = Path(__file__).resolve().parents[1] / "src" / "interface" / "app.py"


@pytest.fixture
def repo(tmp_path: Path) -> OutputsRepository:
    return OutputsRepository(AssetStorage(StorageSettings(force_local=True, local_dir=tmp_path)))


@pytest.fixture
def populated(repo: OutputsRepository, sample_path: Path) -> OutputsRepository:
    repo.run_pipeline(sample_path, demo=True, run_id="r1", options={"force_generate": True})
    return repo


# --------------------------------------------------------------------------- #
# OutputsRepository
# --------------------------------------------------------------------------- #
class TestRepository:
    def test_empty_storage(self, repo: OutputsRepository) -> None:
        assert repo.backend == "local"
        assert repo.list_runs() == [] and repo.latest_run() is None
        assert repo.load_assets() == [] and repo.load_curation() == {}

    def test_run_pipeline_persists_everything(self, populated: OutputsRepository) -> None:
        run = populated.latest_run()
        assert run is not None and run.run_id == "r1"
        assert run.report_object.startswith("decisions/") and run.enriched_object.startswith("analysis/")
        report = populated.load_report(run)
        assert isinstance(report, DecisionReport) and report.message_count == 5
        assert len(populated.load_enriched(run)) == 5
        assert "Decision Engine summary" in populated.load_summary(run)
        assets = populated.load_assets()
        assert {a.channel for a in assets} >= {"linkedin", "newsletter", "testimonial"}
        assert all(a.status == "pending" and a.markdown for a in assets)

    def test_runs_sorted_newest_first(self, repo: OutputsRepository, sample_path: Path) -> None:
        repo.run_pipeline(sample_path, demo=True, run_id="a", options={"skip_generate": True})
        repo.run_pipeline(sample_path, demo=True, run_id="b", options={"skip_generate": True})
        runs = repo.list_runs()
        assert [r.run_id for r in runs] == ["b", "a"] or runs[0].stamp >= runs[1].stamp
        assert "UTC" in runs[0].label

    def test_missing_enriched_and_summary_are_soft_failures(self, repo: OutputsRepository) -> None:
        run = RunRef(stamp="20260101-000000", run_id="ghost")
        assert repo.load_enriched(run) == []
        assert repo.load_summary(run) == ""

    def test_curation_round_trip(self, populated: OutputsRepository) -> None:
        first = populated.load_assets()[0]
        populated.set_status(first.object_name, "approved", note="ship it")
        assert populated.load_curation()[first.object_name]["note"] == "ship it"
        assert populated.load_assets()[0].status == "approved"
        populated.set_status(first.object_name, "rejected")
        assert populated.load_assets()[0].status == "rejected"
        populated.set_status(first.object_name, "pending")
        assert populated.load_curation() == {}
        assert CURATION_OBJECT in populated.storage.list_objects("curation/")

    def test_storage_prefix_is_stripped_from_names(self, tmp_path: Path, sample_path: Path) -> None:
        storage = AssetStorage(StorageSettings(force_local=True, local_dir=tmp_path, prefix="team-a"))
        repo = OutputsRepository(storage)
        repo.run_pipeline(sample_path, demo=True, run_id="p", options={"skip_generate": True})
        run = repo.latest_run()
        assert run is not None and not run.report_object.startswith("team-a/")
        assert repo.load_report(run).message_count == 5

    def test_stored_asset_from_partial_record(self) -> None:
        asset = StoredAsset.from_record(
            "faq/x.json", {"title": "T", "markdown": "m", "generated_at": "2026-01-01T00:00:00"}
        )
        assert asset.channel == "faq" and asset.generated_at.tzinfo is UTC and asset.status == "pending"


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_enriched_frame_and_trend(self, populated: OutputsRepository) -> None:
        run = populated.latest_run()
        assert run is not None
        frame = enriched_frame(populated.load_enriched(run))
        assert len(frame) == 5 and frame["timestamp"].is_monotonic_increasing
        trend = sentiment_trend(frame, "D")
        assert list(trend.columns) == ["period", "mean_sentiment", "messages"]
        assert trend["messages"].sum() == 5
        assert sentiment_trend(enriched_frame([])).empty

    def test_topic_frequencies_boosts_recurring(self, populated: OutputsRepository) -> None:
        run = populated.latest_run()
        assert run is not None
        frame = enriched_frame(populated.load_enriched(run))
        base = topic_frequencies(frame)
        assert base and list(base.values()) == sorted(base.values(), reverse=True)
        report = populated.load_report(run)
        boosted = topic_frequencies(frame, report)
        for topic in report.recurring_topics:
            assert boosted[topic.topic.lower()] >= base.get(topic.topic.lower(), 0) + topic.occurrences

    def test_word_cloud_layout_has_no_overlaps(self) -> None:
        freqs = {f"word{i}": 40 - i for i in range(30)} | {"langgraph": 50, "oci": 12}
        layout = dashboard.layout_word_cloud(freqs, width=800, height=400)
        assert len(layout) == len(freqs)
        assert layout.loc[layout["word"] == "langgraph", "size"].item() == layout["size"].max()
        boxes = [
            (
                r.x - 0.3 * r.size * len(r.word),
                r.y - r.size * 0.55,
                r.x + 0.3 * r.size * len(r.word),
                r.y + r.size * 0.55,
            )
            for r in layout.itertuples()
        ]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                assert a[2] <= b[0] + 1e-6 or a[0] >= b[2] - 1e-6 or a[3] <= b[1] + 1e-6 or a[1] >= b[3] - 1e-6
        assert dashboard.layout_word_cloud({}).empty

    def test_headline_metrics(self, populated: OutputsRepository) -> None:
        run = populated.latest_run()
        assert run is not None
        report = populated.load_report(run)
        tiles = dashboard.headline_metrics(enriched_frame(populated.load_enriched(run)), report)
        assert tiles["Messages analysed"] == "5" and tiles["Highest level"] == report.highest_level.value
        empty = dashboard.headline_metrics(enriched_frame([]), None)
        assert empty["Mean sentiment"] == "n/a" and empty["Highest level"] == "—"

    def test_curation_filters(self) -> None:
        now = datetime.now(UTC)
        items = [
            StoredAsset("linkedin/a.json", "linkedin", "A", "", now, status="approved"),
            StoredAsset("faq/b.json", "faq", "B", "", now),
            StoredAsset("faq/c.json", "faq", "C", "", now, status="rejected"),
        ]
        assert [a.title for a in curation.filter_assets(items, channels=["faq"])] == ["B", "C"]
        assert [a.title for a in curation.filter_assets(items, statuses=["approved"])] == ["A"]
        assert curation.status_counts(items) == {"pending": 1, "approved": 1, "rejected": 1}

    def test_alert_helpers(self, populated: OutputsRepository) -> None:
        run = populated.latest_run()
        assert run is not None
        report = populated.load_report(run)
        urgent = alerts.urgent_escalations(report)
        assert all(e.level >= EscalationLevel.ALTO for e in urgent)
        assert [e.level.rank for e in urgent] == sorted((e.level.rank for e in urgent), reverse=True)
        everything = alerts.urgent_escalations(report, EscalationLevel.INFO)
        assert len(everything) == len(report.escalations)
        members = alerts.flagged_members(report)
        assert all(r.level != "none" for r in members)
        assert len(alerts.escalations_frame(urgent)) == len(urgent)
        assert set(alerts.members_frame(members).columns) >= {"author", "risk_score", "signals"}
        empty = DecisionEngine().run([])
        assert alerts.urgent_escalations(empty) == [] and alerts.flagged_members(empty) == []


# --------------------------------------------------------------------------- #
# The Streamlit app (headless, via AppTest)
# --------------------------------------------------------------------------- #
@pytest.fixture
def app_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("COMMUNITYLAB_FORCE_LOCAL", "1")
    monkeypatch.setenv("COMMUNITYLAB_LOCAL_DIR", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    st.cache_resource.clear()
    return AppTest.from_file(str(APP_FILE), default_timeout=120)


class TestApp:
    def test_empty_state_and_navigation(self, app_test: AppTest) -> None:
        at = app_test.run()
        assert not at.exception
        assert at.sidebar.radio[0].value == "Dashboard"
        assert "No pipeline outputs" in at.info[0].value
        assert [t.label for t in at.sidebar.toggle] == [
            "Read from local fallback only",
            "Demo mode (mock LLM, no Groq)",
        ]
        assert at.sidebar.toggle[1].value is True, "demo mode defaults on when GROQ_API_KEY is unset"

        at.sidebar.radio[0].set_value("Curation").run()
        assert not at.exception and "No generated assets" in at.info[0].value
        at.sidebar.radio[0].set_value("Alerts").run()
        assert not at.exception and "No decision reports" in at.info[0].value

    def test_run_pipeline_then_browse_views(self, app_test: AppTest, tmp_path: Path) -> None:
        at = app_test.run()
        at.sidebar.button[0].click().run()
        assert not at.exception
        assert at.sidebar.success and "5 messages" in at.sidebar.success[0].value
        assert (tmp_path / "decisions").is_dir() and (tmp_path / "analysis").is_dir()

        labels = {m.label: m.value for m in at.metric}
        assert labels["Messages analysed"] == "5" and labels["Highest level"] in {"INFO", "MEDIO", "ALTO", "CRÍTICO"}
        assert at.selectbox[0].label == "Pipeline run"

        at.sidebar.radio[0].set_value("Curation").run()
        assert not at.exception
        counts = {m.label: m.value for m in at.metric}
        assert counts["Assets"] == counts["Pending"] and int(counts["Assets"]) >= 3
        next(b for b in at.button if b.label.startswith("✅")).click().run()
        assert not at.exception
        counts = {m.label: m.value for m in at.metric}
        assert counts["Approved"] == "1"

        at.sidebar.radio[0].set_value("Alerts").run()
        assert not at.exception
        assert {m.label for m in at.metric} >= {"CRÍTICO", "ALTO", "MEDIO", "INFO", "Members at risk"}
        assert [s.value for s in at.subheader][:2] == ["Escalations", "Members at risk (R1)"]

    def test_pipeline_failure_is_shown_not_raised(self, app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: object, **__: object) -> dict[str, object]:
            raise RuntimeError("groq exploded")

        monkeypatch.setattr(OutputsRepository, "run_pipeline", boom)
        at = app_test.run()
        at.sidebar.button[0].click().run()
        assert not at.exception
        assert "groq exploded" in at.sidebar.error[0].value
