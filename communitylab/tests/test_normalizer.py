"""Tests for :mod:`src.ingest.normalizer`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
from src.ingest import loader
from src.ingest.normalizer import (
    COMMON_SCHEMA,
    NormalizationError,
    clean_text,
    count_reactions,
    normalize,
    parse_timestamp,
    to_records,
)


def test_normalize_sample_matches_schema(sample_payload: list[dict[str, Any]]) -> None:
    frame = normalize(sample_payload)

    assert list(frame.columns) == list(COMMON_SCHEMA)
    assert len(frame) == 5
    assert str(frame["timestamp"].dtype).startswith("datetime64") and frame["timestamp"].dt.tz is not None
    assert frame["reactions"].dtype == "int64"
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["author"].tolist() == ["marta.dev", "carlos_mentor", "priya.n", "tomas.r", "lee"]
    assert frame["reactions"].tolist() == [2, 5, 39, 8, 3]
    assert frame.loc[1, "thread_id"] == "1287431190524231681"


def test_normalize_discord_payload(discord_payload: list[dict[str, Any]]) -> None:
    frame = normalize(loader.load_discord(discord_payload, channel="help-oci"))

    first, second = to_records(frame)
    assert first["author"] == "Alice"  # global_name preferred over username
    assert first["author_id"] == "u_1"
    assert first["channel"] == "help-oci"
    assert first["content"] == "How do I configure the OCI CLI?"
    assert first["reactions"] == 3
    assert first["thread_id"] is None
    assert first["metadata"] == {"type": 0}
    assert second["author"] == "bob"
    assert second["thread_id"] == "1300000000000000001"


def test_normalize_slack_payload(slack_payload: dict[str, Any]) -> None:
    frame = normalize(loader.load_slack(slack_payload, channel="#wins"))

    parent, reply = to_records(frame)
    assert parent["message_id"] == "1726300987.004500"
    assert parent["author"] == "priya"
    assert parent["author_id"] == "U07AB3CD9"
    assert parent["thread_id"] is None  # thread_ts == ts means "parent", not "reply"
    assert parent["reactions"] == 3
    assert parent["timestamp"] == datetime.fromtimestamp(1726300987.0045, tz=UTC)
    assert reply["author"] == "U0ZZ"  # no profile: fall back to the user id
    assert reply["thread_id"] == "1726300987.004500"


def test_normalize_drops_empty_and_duplicates() -> None:
    records = [
        {"id": "1", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"},
        {"id": "1", "content": "hello again", "timestamp": "2026-01-01T00:00:01Z"},
        {"id": "2", "content": "   \u200b ", "timestamp": "2026-01-01T00:00:02Z"},
        {"id": "3", "content": "later", "timestamp": "2026-01-01T00:00:03Z"},
    ]

    frame = normalize(records, default_source="webhook")

    assert frame["message_id"].tolist() == ["1", "3"]
    assert frame["content"].tolist() == ["hello", "later"]
    assert frame["source"].unique().tolist() == ["webhook"]

    kept = normalize(records, drop_empty=False, deduplicate=False)
    assert len(kept) == 4


def test_normalize_lowercases_labels_and_sorts_nat_last() -> None:
    records = [
        {"id": "b", "content": "x", "timestamp": "not a date", "category": "Complaint", "sentiment": "NEGATIVE"},
        {"id": "a", "content": "y", "timestamp": "2026-02-01T00:00:00Z"},
    ]

    frame = normalize(records)

    assert frame["message_id"].tolist() == ["a", "b"]
    assert pd.isna(frame.loc[1, "timestamp"])
    assert frame.loc[1, "category"] == "complaint"
    assert frame.loc[1, "sentiment"] == "negative"


def test_normalize_empty_input_has_schema() -> None:
    frame = normalize([])

    assert frame.empty
    assert list(frame.columns) == list(COMMON_SCHEMA)
    assert to_records(frame) == []


def test_normalize_rejects_non_mapping() -> None:
    with pytest.raises(NormalizationError):
        normalize(["not a record"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello   world ", "hello world"),
        ("line1\r\n\r\n\r\n\r\nline2", "line1\n\nline2"),
        ("zero\u200bwidth", "zerowidth"),
        (None, ""),
        (float("nan"), ""),
        (42, "42"),
    ],
)
def test_clean_text(raw: Any, expected: str) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0),
        (7, 7),
        ("3", 3),
        ("many", 0),
        (-2, 0),
        ({"tada": 2, "fire": 1}, 3),
        ([{"emoji": {"name": "x"}, "count": 4}], 4),
        ([{"name": "tada", "users": ["a", "b"]}], 2),
        (["👍", "👍"], 2),
    ],
)
def test_count_reactions(raw: Any, expected: int) -> None:
    assert count_reactions(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-14T09:12:41Z", datetime(2026, 9, 14, 9, 12, 41, tzinfo=UTC)),
        ("2026-09-14T11:12:41+02:00", datetime(2026, 9, 14, 9, 12, 41, tzinfo=UTC)),
        ("2026-09-14 09:12:41", datetime(2026, 9, 14, 9, 12, 41, tzinfo=UTC)),
        ("1726300987.004500", datetime.fromtimestamp(1726300987.0045, tz=UTC)),
        (1726300987, datetime.fromtimestamp(1726300987, tz=UTC)),
        ("1726300987000", datetime.fromtimestamp(1726300987, tz=UTC)),
    ],
)
def test_parse_timestamp(raw: Any, expected: datetime) -> None:
    parsed = parse_timestamp(raw)

    assert parsed is not None
    assert parsed.to_pydatetime() == expected


@pytest.mark.parametrize("raw", [None, "", "yesterday-ish", float("nan")])
def test_parse_timestamp_unparseable(raw: Any) -> None:
    assert parse_timestamp(raw) is None
