"""Tests for :mod:`src.ingest.validator`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError
from src.ingest import load_json, normalize, validate
from src.ingest.normalizer import COMMON_SCHEMA
from src.ingest.validator import (
    CommunityMessage,
    InvalidBatchError,
    RowError,
    ValidationResult,
    to_frame,
    validate_records,
)

NOW = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "message_id": "1",
        "source": "discord",
        "channel": "general",
        "author": "alice",
        "author_id": "u_1",
        "content": "hello",
        "timestamp": NOW,
        "thread_id": None,
        "reactions": 0,
        "category": None,
        "sentiment": None,
        "metadata": {},
    }
    base.update(overrides)
    return base


def test_end_to_end_sample_is_valid(sample_path) -> None:  # type: ignore[no-untyped-def]
    result = validate(normalize(load_json(sample_path)))

    assert result.is_valid
    assert result.total == 5
    assert result.error_rate == 0.0
    assert {m.sentiment for m in result.valid} == {"neutral", "positive", "negative", "mixed"}
    assert sum(m.is_reply for m in result.valid) == 1
    result.raise_for_errors()  # must not raise


def test_community_message_normalizes_and_defaults() -> None:
    message = CommunityMessage.model_validate(
        _row(
            channel=None,
            author="  ",
            category="Success_Story",
            sentiment="POSITIVE",
            timestamp="2026-09-14T11:00:00+02:00",
        )
    )

    assert message.channel == "unknown"
    assert message.author == "unknown"
    assert message.category == "success_story"
    assert message.sentiment == "positive"
    assert message.timestamp == NOW
    assert message.timestamp.tzinfo is UTC


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", ""),
        ("message_id", ""),
        ("reactions", -1),
        ("sentiment", "angry"),
        ("source", "Not Valid!"),
        ("timestamp", datetime(2026, 1, 1)),  # naive
        ("timestamp", None),
    ],
)
def test_community_message_rejects_bad_values(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        CommunityMessage.model_validate(_row(**{field: value}))


def test_community_message_forbids_unknown_fields_and_is_frozen() -> None:
    with pytest.raises(ValidationError):
        CommunityMessage.model_validate(_row(surprise=1))

    message = CommunityMessage.model_validate(_row())
    with pytest.raises(ValidationError):
        message.content = "mutated"  # type: ignore[misc]


def test_validate_records_collects_row_errors() -> None:
    rows = [_row(), _row(message_id="2", content=""), _row(message_id="3", timestamp=None, reactions=-5)]

    result = validate_records(rows)

    assert len(result.valid) == 1
    assert len(result.errors) == 2
    assert not result.is_valid
    assert result.error_rate == pytest.approx(2 / 3)
    second, third = result.errors
    assert isinstance(second, RowError)
    assert second.row == 1 and second.message_id == "2"
    assert any(e.startswith("content:") for e in second.errors)
    assert third.row == 2
    assert {e.split(":")[0] for e in third.errors} == {"timestamp", "reactions"}
    assert "row 1 (discord/2)" in str(second)

    with pytest.raises(InvalidBatchError) as excinfo:
        result.raise_for_errors()
    assert excinfo.value.result is result
    assert "2 of 3 rows failed" in str(excinfo.value)


def test_validate_frame_requires_schema_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate(pd.DataFrame({"content": ["hi"]}))


def test_validate_frame_handles_nat_and_nan() -> None:
    frame = normalize(
        [
            {"id": "ok", "content": "fine", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "bad-ts", "content": "no date", "timestamp": "???"},
        ]
    )

    result = validate(frame)

    assert [m.message_id for m in result.valid] == ["ok"]
    assert result.valid[0].author == "unknown"
    assert [e.message_id for e in result.errors] == ["bad-ts"]


def test_to_frame_roundtrip(sample_path) -> None:  # type: ignore[no-untyped-def]
    normalized = normalize(load_json(sample_path))
    result = validate(normalized)

    frame = to_frame(result.valid)

    assert list(frame.columns) == list(COMMON_SCHEMA)
    assert frame["message_id"].tolist() == normalized["message_id"].tolist()
    assert frame["reactions"].dtype == "int64"
    assert frame["timestamp"].dt.tz is not None


def test_empty_result() -> None:
    result = ValidationResult()

    assert result.total == 0
    assert result.is_valid
    assert result.error_rate == 0.0
