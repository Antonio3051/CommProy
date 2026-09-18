"""Pydantic validation for normalized community messages.

This is the last gate before data reaches the AI pipeline. It turns each row of
the normalized ``DataFrame`` into a :class:`CommunityMessage` and collects the
rows that fail into :class:`RowError` objects instead of aborting the batch, so
one malformed Slack export never blocks the rest of the ingest.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.ingest.normalizer import COMMON_SCHEMA, to_records

logger = logging.getLogger(__name__)

Sentiment = Literal["positive", "negative", "neutral", "mixed"]
"""Sentiment labels accepted on pre-labelled data."""


class CommunityMessage(BaseModel):
    """One validated community interaction in the common schema.

    Attributes:
        message_id: Identifier unique within ``source``.
        source: Platform that produced the message (``discord``, ``slack``...).
        channel: Channel / room the message was posted in.
        author: Display name of the author.
        author_id: Platform user id, when available.
        content: Cleaned message text; never empty.
        timestamp: Timezone-aware UTC datetime of the post.
        thread_id: Id of the parent message when this is a reply.
        reactions: Total number of reactions (non-negative).
        category: Optional coarse label such as ``technical_question``.
        sentiment: Optional pre-labelled sentiment.
        metadata: Source-specific extras preserved by the normalizer.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    message_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_\-]+$")
    channel: str = Field(default="unknown", min_length=1, max_length=256)
    author: str = Field(default="unknown", min_length=1, max_length=256)
    author_id: str | None = Field(default=None, max_length=128)
    content: str = Field(min_length=1, max_length=20_000)
    timestamp: datetime
    thread_id: str | None = Field(default=None, max_length=128)
    reactions: int = Field(default=0, ge=0)
    category: str | None = Field(default=None, max_length=64)
    sentiment: Sentiment | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("channel", "author", mode="before")
    @classmethod
    def _default_when_missing(cls, value: Any) -> Any:
        """Fall back to ``"unknown"`` for missing channel/author rather than failing."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return "unknown"
        return value

    @field_validator("timestamp")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        """Require an aware datetime and normalize it to UTC."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def _lowercase_labels(cls, value: Any) -> Any:
        """Labels are compared case-insensitively downstream."""
        return value.lower().strip() if isinstance(value, str) else value

    @property
    def is_reply(self) -> bool:
        """Whether this message answers another one."""
        return self.thread_id is not None


class RowError(BaseModel):
    """Describes why a single row was rejected."""

    model_config = ConfigDict(frozen=True)

    row: int = Field(description="0-based position of the row in the input frame.")
    message_id: str | None = Field(default=None, description="Raw message id, if any.")
    source: str | None = Field(default=None, description="Raw source tag, if any.")
    errors: tuple[str, ...] = Field(description="Human-readable ``field: reason`` messages.")

    def __str__(self) -> str:
        ident = self.message_id or "<no id>"
        return f"row {self.row} ({self.source or '?'}/{ident}): " + "; ".join(self.errors)


class ValidationResult(BaseModel):
    """Outcome of validating a batch of normalized rows."""

    model_config = ConfigDict(frozen=True)

    valid: tuple[CommunityMessage, ...] = ()
    errors: tuple[RowError, ...] = ()

    @property
    def total(self) -> int:
        """Number of rows that were examined."""
        return len(self.valid) + len(self.errors)

    @property
    def is_valid(self) -> bool:
        """True when every row passed validation."""
        return not self.errors

    @property
    def error_rate(self) -> float:
        """Fraction of rejected rows, ``0.0`` for an empty batch."""
        return len(self.errors) / self.total if self.total else 0.0

    def raise_for_errors(self) -> None:
        """Raise :class:`InvalidBatchError` if any row was rejected."""
        if self.errors:
            raise InvalidBatchError(self)


class InvalidBatchError(ValueError):
    """Raised by :meth:`ValidationResult.raise_for_errors` when rows were rejected."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        preview = "\n".join(str(error) for error in result.errors[:5])
        more = f"\n... and {len(result.errors) - 5} more" if len(result.errors) > 5 else ""
        super().__init__(f"{len(result.errors)} of {result.total} rows failed validation:\n{preview}{more}")


def _format_errors(exc: ValidationError) -> tuple[str, ...]:
    """Flatten a Pydantic ``ValidationError`` into ``field: reason`` strings."""
    formatted: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        formatted.append(f"{location}: {error['msg']}")
    return tuple(formatted)


def validate_records(records: Iterable[Mapping[str, Any]]) -> ValidationResult:
    """Validate already-normalized dict rows.

    Args:
        records: Mappings whose keys follow :data:`COMMON_SCHEMA`.

    Returns:
        A :class:`ValidationResult` separating good messages from rejected rows.
    """
    valid: list[CommunityMessage] = []
    errors: list[RowError] = []
    for index, record in enumerate(records):
        try:
            valid.append(CommunityMessage.model_validate(dict(record)))
        except ValidationError as exc:
            raw_id = record.get("message_id")
            raw_source = record.get("source")
            errors.append(
                RowError(
                    row=index,
                    message_id=str(raw_id) if raw_id is not None else None,
                    source=str(raw_source) if raw_source is not None else None,
                    errors=_format_errors(exc),
                )
            )
    if errors:
        logger.warning("%d of %d rows failed validation", len(errors), len(valid) + len(errors))
    return ValidationResult(valid=tuple(valid), errors=tuple(errors))


def validate(frame: pd.DataFrame) -> ValidationResult:
    """Validate a normalized ``DataFrame`` row by row.

    Args:
        frame: Output of :func:`src.ingest.normalizer.normalize`.

    Returns:
        A :class:`ValidationResult` separating good messages from rejected rows.

    Raises:
        ValueError: If the frame is missing any column of :data:`COMMON_SCHEMA`.
    """
    missing = [column for column in COMMON_SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")
    return validate_records(to_records(frame))


def to_frame(messages: Iterable[CommunityMessage]) -> pd.DataFrame:
    """Turn validated messages back into a ``DataFrame`` with the common schema.

    Handy for the analysis layer, which prefers Pandas over model instances.
    """
    rows = [message.model_dump() for message in messages]
    frame = pd.DataFrame(rows, columns=list(COMMON_SCHEMA))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["reactions"] = frame["reactions"].astype("int64")
    return frame
