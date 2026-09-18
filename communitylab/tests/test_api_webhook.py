"""Tests for the FastAPI trigger in ``src/api/webhook.py`` (no Groq/OCI needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from src.api import RunOptions, create_app
from src.api.webhook import app as default_app
from src.orchestration import PipelineComponents, compile_pipeline

from tests.test_orchestration import consolidator, generators, local_storage


@pytest.fixture
def factory_calls() -> list[str]:
    return []


@pytest.fixture
def client(tmp_path: Path, factory_calls: list[str]) -> TestClient:
    def factory():
        factory_calls.append("built")
        comps = PipelineComponents(
            consolidator=consolidator(), generators=generators(), storage=local_storage(tmp_path)
        )
        return compile_pipeline(comps)

    return TestClient(create_app(factory))


class TestRunOptions:
    def test_defaults_map_to_pipeline_options(self):
        opts = RunOptions().to_pipeline_options()
        assert opts["language"] == "en" and opts["channels"] == ("linkedin", "newsletter", "faq", "testimonial")
        assert opts["max_posts"] == 3 and opts["store_decisions"] is True

    def test_invalid_channel_rejected(self):
        with pytest.raises(ValueError):
            RunOptions(channels=["tiktok"])  # type: ignore[list-item]


class TestEndpoints:
    def test_health_before_and_after_first_run(self, client: TestClient, sample_path: Path, factory_calls: list[str]):
        assert client.get("/health").json()["graph_compiled"] is False
        assert factory_calls == []
        client.post("/runs", json={"source": "json", "path": str(sample_path)})
        assert client.get("/health").json()["graph_compiled"] is True
        client.post("/runs", json={"source": "json", "path": str(sample_path)})
        assert factory_calls == ["built"]

    def test_run_file_sync(self, client: TestClient, sample_path: Path):
        response = client.post(
            "/runs",
            json={"source": "json", "path": str(sample_path), "options": {"channels": ["linkedin"], "max_posts": 1}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed" and body["error"] is None
        summary = body["summary"]
        assert summary["validated"] == 5 and summary["analyzed"] == 5
        assert [a["channel"] for a in summary["generated_assets"]] == ["linkedin"]
        assert summary["stored"] and summary["errors"] == []

    def test_webhook_discord_sync(self, client: TestClient, discord_payload: list[dict[str, Any]]):
        response = client.post(
            "/webhook/discord",
            json={"payload": discord_payload, "channel": "general", "options": {"skip_generate": True}},
        )
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["source"] == "discord" and summary["validated"] == len(discord_payload)
        assert summary["generated_assets"] == []
        assert [s["location"].split("/")[-2] for s in summary["stored"]] == ["decisions", "decisions"]

    def test_webhook_slack_and_generic(self, client: TestClient, slack_payload: dict[str, Any]):
        slack = client.post("/webhook/slack", json={"payload": slack_payload, "channel": "help"})
        assert slack.status_code == 200 and slack.json()["summary"]["validated"] >= 1
        generic = client.post(
            "/webhook/webhook",
            json={
                "payload": [
                    {"id": "w1", "author": "x", "content": "webhook hello", "timestamp": "2026-09-14T12:00:00Z"}
                ]
            },
        )
        assert generic.status_code == 200 and generic.json()["summary"]["validated"] == 1

    def test_unknown_source_is_422(self, client: TestClient):
        assert client.post("/webhook/telegram", json={"payload": []}).status_code == 422

    def test_async_run_returns_run_id_then_completes(self, client: TestClient, sample_path: Path):
        response = client.post("/runs?async=true", json={"source": "json", "path": str(sample_path)})
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        # TestClient executes background tasks before returning, so the run is already finished.
        status = client.get(f"/runs/{run_id}").json()
        assert status["status"] == "completed" and status["summary"]["run_id"] == run_id

    def test_unknown_run_is_404(self, client: TestClient):
        assert client.get("/runs/nope").status_code == 404

    def test_failed_graph_is_reported_not_raised(self, tmp_path: Path):
        def factory():
            raise RuntimeError("no GROQ_API_KEY")

        client = TestClient(create_app(factory))
        response = client.post("/runs", json={"source": "json", "path": str(tmp_path / "x.json")})
        assert response.status_code == 200
        assert response.json()["status"] == "failed" and "GROQ_API_KEY" in response.json()["error"]


def test_default_app_imports_without_credentials():
    routes = {route.path for route in default_app.routes}
    assert {"/health", "/webhook/{source}", "/runs", "/runs/{run_id}"} <= routes
    assert TestClient(default_app).get("/health").json()["graph_compiled"] is False
