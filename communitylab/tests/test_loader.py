"""Tests for :mod:`src.ingest.loader`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from src.ingest import loader
from src.ingest.loader import LoaderError


def test_load_json_reads_sample_and_keeps_declared_source(sample_path: Path) -> None:
    records = loader.load_json(sample_path)

    assert len(records) == 5
    assert {record["source"] for record in records} == {"discord", "slack"}
    assert records[0]["author"]["username"] == "marta.dev"


def test_load_json_unwraps_envelope_and_tags_source(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"messages": [{"id": "1", "content": "hi"}]}), encoding="utf-8")

    records = loader.load_json(path)

    assert records == [{"id": "1", "content": "hi", "source": "json"}]


def test_load_json_rejects_invalid_document(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(LoaderError):
        loader.load_json(path)


def test_load_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_json(tmp_path / "nope.json")


def test_load_csv_converts_blank_cells_to_none(tmp_path: Path) -> None:
    path = tmp_path / "messages.csv"
    path.write_text(
        "id,author,content,timestamp,thread_id\n"
        "1,alice,hello,2026-01-01T00:00:00Z,\n"
        "2,bob,world,2026-01-01T00:01:00Z,1\n",
        encoding="utf-8",
    )

    records = loader.load_csv(path)

    assert len(records) == 2
    assert records[0]["thread_id"] is None
    assert records[1]["thread_id"] == "1"
    assert all(record["source"] == "csv" for record in records)
    assert records[0]["id"] == "1"  # ids stay strings


def test_load_discord_accepts_list_dict_and_string(discord_payload: list[dict[str, Any]]) -> None:
    from_list = loader.load_discord(discord_payload, channel="help")
    from_dict = loader.load_discord({"messages": discord_payload})
    from_str = loader.load_discord(json.dumps(discord_payload))

    assert [r["id"] for r in from_list] == [r["id"] for r in from_dict] == [r["id"] for r in from_str]
    assert all(r["source"] == "discord" for r in from_list)
    assert all(r["channel"] == "help" for r in from_list)
    assert "channel" not in from_dict[0]


def test_load_slack_unwraps_history_response(slack_payload: dict[str, Any]) -> None:
    records = loader.load_slack(slack_payload, channel="#wins")

    assert len(records) == 2
    assert records[0]["source"] == "slack"
    assert records[0]["channel"] == "#wins"


def test_load_slack_surfaces_api_errors() -> None:
    with pytest.raises(LoaderError, match="not_authed"):
        loader.load_slack({"ok": False, "error": "not_authed"})


@pytest.mark.parametrize(
    "body",
    [
        b'{"id": "1", "content": "hi"}',
        '[{"id": "1", "content": "hi"}]',
        {"data": [{"id": "1", "content": "hi"}]},
    ],
)
def test_load_webhook_accepts_common_shapes(body: Any) -> None:
    records = loader.load_webhook(body)

    assert records == [{"id": "1", "content": "hi", "source": "webhook"}]


def test_load_webhook_rejects_non_records() -> None:
    with pytest.raises(LoaderError):
        loader.load_webhook("[1, 2, 3]")
    with pytest.raises(LoaderError):
        loader.load_webhook('"just a string"')


def test_fetch_discord_channel_uses_injected_http(discord_payload: list[dict[str, Any]]) -> None:
    calls: list[tuple[str, Mapping[str, str]]] = []

    def fake_get(url: str, headers: Mapping[str, str]) -> bytes:
        calls.append((url, headers))
        return json.dumps(discord_payload).encode()

    records = loader.fetch_discord_channel("990", "TOKEN", limit=500, channel="help", http_get=fake_get)

    assert len(records) == 2
    url, headers = calls[0]
    assert url.endswith("/channels/990/messages?limit=100")  # clamped
    assert headers["Authorization"] == "Bot TOKEN"
    assert records[0]["channel"] == "help"


def test_fetch_slack_channel_uses_injected_http(slack_payload: dict[str, Any]) -> None:
    seen: dict[str, Any] = {}

    def fake_get(url: str, headers: Mapping[str, str]) -> bytes:
        seen["url"] = url
        seen["headers"] = headers
        return json.dumps(slack_payload).encode()

    records = loader.fetch_slack_channel("C01", "xoxb-1", limit=50, http_get=fake_get)

    assert len(records) == 2
    assert "conversations.history?channel=C01&limit=50" in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer xoxb-1"


def test_load_dispatcher(sample_path: Path, slack_payload: dict[str, Any]) -> None:
    assert len(loader.load("json", sample_path)) == 5
    assert len(loader.load("slack", slack_payload)) == 2
    with pytest.raises(LoaderError, match="Unknown source"):
        loader.load("carrier-pigeon", {})  # type: ignore[arg-type]
