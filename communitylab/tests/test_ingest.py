"""End-to-end ingestion tests: Loader → Normalizer → Validator for every source.

The per-module suites (``test_loader.py``, ``test_normalizer.py``,
``test_validator.py``) cover edge cases; this file guards the contract the
LangGraph ``ingest`` node relies on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from src.ingest import COMMON_SCHEMA, CommunityMessage, load, normalize, validate


def ingest(source: str, target: Any, **kwargs: Any) -> tuple[pd.DataFrame, list[CommunityMessage]]:
    frame = normalize(load(source, target, **kwargs), default_source=source)  # type: ignore[arg-type]
    return frame, list(validate(frame).valid)


def test_sample_json_round_trip(sample_path: Path) -> None:
    frame, valid = ingest("json", sample_path)
    assert list(frame.columns)[: len(COMMON_SCHEMA)] == list(COMMON_SCHEMA)
    assert len(valid) == 5
    assert {m.source for m in valid} == {"discord", "slack"}
    assert all(m.timestamp.tzinfo is not None for m in valid)
    assert all(m.content == m.content.strip() and "\u200b" not in m.content for m in valid)


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "messages.csv"
    path.write_text(
        "message_id,source,channel,author,content,timestamp,reactions\n"
        "c1,slack,general,ana,Hello team,2026-09-10T10:00:00Z,1\n"
        "c2,slack,general,ana,,2026-09-10T10:01:00Z,\n"
        "c3,slack,general,ben,Second message,2026-09-10T10:02:00Z,\n",
        encoding="utf-8",
    )
    frame, valid = ingest("csv", path)
    assert len(frame) == 2, "blank content is dropped by the normalizer"
    assert [m.message_id for m in valid] == ["c1", "c3"]
    assert valid[1].reactions == 0


def test_discord_round_trip(discord_payload: list[dict[str, Any]]) -> None:
    _, valid = ingest("discord", discord_payload, channel="help")
    assert [m.author for m in valid] == ["Alice", "bob"]
    assert valid[0].reactions == 3
    assert valid[1].thread_id == "1300000000000000001"
    assert all(m.source == "discord" and m.channel == "help" for m in valid)


def test_slack_round_trip(slack_payload: dict[str, Any]) -> None:
    _, valid = ingest("slack", slack_payload, channel="wins")
    assert len(valid) == 2
    assert valid[0].author == "priya" or valid[0].author == "Priya N"
    assert valid[0].reactions == 3
    assert valid[1].thread_id == valid[0].message_id


def test_webhook_round_trip_deduplicates() -> None:
    body = json.dumps(
        {
            "messages": [
                {"id": "w1", "user": "kim", "text": "Deploy failed again 😕", "time": "2026-09-10T10:00:00Z"},
                {"id": "w1", "user": "kim", "text": "Deploy failed again 😕", "time": "2026-09-10T10:00:00Z"},
                {"id": "w2", "user": "lee", "text": "Looking into it", "time": "2026-09-10T10:03:00Z"},
            ]
        }
    )
    frame, valid = ingest("webhook", body)
    assert len(frame) == 2
    assert [m.message_id for m in valid] == ["w1", "w2"]
    assert all(m.source == "webhook" for m in valid)


def test_invalid_rows_are_reported_not_raised() -> None:
    records = [
        {"id": "ok", "author": "ana", "content": "fine", "timestamp": "2026-09-10T10:00:00Z"},
        {"id": "bad-ts", "author": "ana", "content": "fine", "timestamp": "not a date"},
    ]
    result = validate(normalize(records))
    assert [m.message_id for m in result.valid] == ["ok"]
    assert len(result.errors) == 1 and result.errors[0].message_id == "bad-ts"
    assert result.total == 2


def test_unknown_source_is_rejected(sample_path: Path) -> None:
    with pytest.raises((ValueError, KeyError)):
        load("email", sample_path)  # type: ignore[arg-type]
