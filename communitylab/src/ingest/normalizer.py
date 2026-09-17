"""Normalizer for the CommunityLab ingestion layer.

Takes the heterogeneous ``RawRecord`` dicts produced by :mod:`src.ingest.loader`
(Discord, Slack, webhook, JSON/CSV exports) and produces a single Pandas
``DataFrame`` that follows :data:`COMMON_SCHEMA`. All cleaning that does not
require an LLM happens here: alias resolution, whitespace/Unicode cleanup,
timestamp parsing, reaction counting, de-duplication and ordering.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Final

import pandas as pd

from src.ingest.loader import RawRecord

logger = logging.getLogger(__name__)

COMMON_SCHEMA: Final[tuple[str, ...]] = (
    "message_id",
    "source",
    "channel",
    "author",
    "author_id",
    "content",
    "timestamp",
    "thread_id",
    "reactions",
    "category",
    "sentiment",
    "metadata",
)
"""Ordered column names every normalized ``DataFrame`` must contain."""

KNOWN_SOURCES: Final[frozenset[str]] = frozenset({"discord", "slack", "webhook", "json", "csv"})

# Field aliases, checked in order. Dotted paths walk nested mappings.
_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "message_id": ("message_id", "id", "ts", "client_msg_id", "uuid"),
    "channel": ("channel", "channel_name", "channel_id", "room", "conversation"),
    "author": (
        "author",
        "author.global_name",
        "author.username",
        "author.name",
        "author_name",
        "user_profile.display_name",
        "user_profile.real_name",
        "user_name",
        "username",
        "user",
        "sender",
    ),
    "author_id": ("author_id", "author.id", "user_id", "user", "sender_id"),
    "content": ("content", "text", "message", "body"),
    "timestamp": ("timestamp", "created_at", "ts", "time", "date"),
    "thread_id": ("thread_id", "message_reference.message_id", "thread_ts", "parent_id", "reply_to"),
    "category": ("category", "label", "type_label", "topic"),
    "sentiment": ("sentiment", "sentiment_label"),
}

# Keys that were mapped into the schema and therefore should not be echoed in metadata.
_CONSUMED_KEYS: Final[frozenset[str]] = frozenset(
    {alias.split(".")[0] for aliases in _ALIASES.values() for alias in aliases} | {"reactions", "source"}
)

_ZERO_WIDTH_RE: Final[re.Pattern[str]] = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_EPOCH_SECONDS_RE: Final[re.Pattern[str]] = re.compile(r"\d{9,10}(\.\d+)?")
_EPOCH_MILLIS_RE: Final[re.Pattern[str]] = re.compile(r"\d{13}")


class NormalizationError(ValueError):
    """Raised when the incoming records cannot be shaped into the common schema."""


# --------------------------------------------------------------------------- #
# Field extraction helpers
# --------------------------------------------------------------------------- #
def _dig(record: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted ``path`` inside nested mappings, returning ``None`` if absent."""
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_missing(value: Any) -> bool:
    """True for ``None``, NaN, and blank strings."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _first(record: Mapping[str, Any], field: str, *, scalar_only: bool = True) -> Any:
    """Return the first non-missing alias value for ``field``.

    Mapping values are skipped when ``scalar_only`` is set so that, for example,
    ``author`` (a dict on Discord) falls through to ``author.username``.
    """
    for alias in _ALIASES[field]:
        value = _dig(record, alias)
        if _is_missing(value):
            continue
        if scalar_only and isinstance(value, (Mapping, list, tuple, set)):
            continue
        return value
    return None


def _as_str(value: Any) -> str | None:
    """Stringify scalars, returning ``None`` for missing values."""
    if _is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def clean_text(text: Any) -> str:
    """Normalize a message body for downstream NLP.

    * Unicode NFC normalization and zero-width character removal.
    * Collapses runs of spaces/tabs into one space (newlines are kept, but at
      most two consecutive ones so paragraphs survive).
    * Strips leading/trailing whitespace on every line and on the whole text.

    Args:
        text: Raw message body; non-strings are coerced with ``str``.

    Returns:
        The cleaned text, or an empty string when ``text`` is missing.
    """
    if _is_missing(text):
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = _ZERO_WIDTH_RE.sub("", normalized).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    return _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


def count_reactions(reactions: Any) -> int:
    """Total the reactions on a message regardless of platform shape.

    Handles Discord (``[{"emoji": {...}, "count": n}]``), Slack
    (``[{"name": "tada", "count": n, "users": [...]}]``), a bare integer, a
    numeric string, or a ``{emoji: count}`` mapping.
    """
    if _is_missing(reactions):
        return 0
    if isinstance(reactions, bool):
        return int(reactions)
    if isinstance(reactions, (int, float)):
        return max(0, int(reactions))
    if isinstance(reactions, str):
        try:
            return max(0, int(float(reactions)))
        except ValueError:
            return 0
    if isinstance(reactions, Mapping):
        return sum(count_reactions(v) for v in reactions.values())
    if isinstance(reactions, Sequence):
        total = 0
        for item in reactions:
            if isinstance(item, Mapping):
                if "count" in item:
                    total += count_reactions(item["count"])
                elif isinstance(item.get("users"), Sequence):
                    total += len(item["users"])
                else:
                    total += 1
            else:
                total += 1
        return total
    return 0


def _epoch_to_timestamp(value: str | int | float, *, unit: str) -> pd.Timestamp:
    """Convert an epoch value to a UTC ``Timestamp`` with microsecond precision.

    Goes through ``Decimal`` so Slack ``ts`` strings like ``"1726300987.004500"``
    do not pick up binary floating-point noise.
    """
    exact = Decimal(str(value))
    micros = int((exact * (1_000_000 if unit == "s" else 1_000)).to_integral_value())
    return pd.Timestamp(micros, unit="us", tz="UTC")


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse ISO-8601 strings, Slack ``ts`` floats and epoch numbers to UTC.

    Returns ``None`` (rendered as ``NaT`` in the frame) when the value cannot be
    parsed so the validator can reject the row instead of the whole batch.
    """
    if _is_missing(value):
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return _epoch_to_timestamp(value, unit="s")
        if isinstance(value, str):
            text = value.strip()
            if _EPOCH_SECONDS_RE.fullmatch(text):
                return _epoch_to_timestamp(text, unit="s")
            if _EPOCH_MILLIS_RE.fullmatch(text):
                return _epoch_to_timestamp(text, unit="ms")
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _resolve_source(record: Mapping[str, Any], default_source: str) -> str:
    raw = _as_str(record.get("source"))
    source = (raw or default_source).lower()
    if source not in KNOWN_SOURCES:
        logger.debug("Unknown source '%s', keeping as-is", source)
    return source


def _flatten(record: Mapping[str, Any], default_source: str) -> dict[str, Any]:
    """Map one raw record onto the common schema (no cross-row cleaning yet)."""
    source = _resolve_source(record, default_source)

    author = _as_str(_first(record, "author"))
    author_id = _as_str(_first(record, "author_id"))

    thread_id = _as_str(_first(record, "thread_id"))
    message_id = _as_str(_first(record, "message_id"))
    # Slack marks thread parents with thread_ts == ts; that is not a reply.
    if source == "slack" and thread_id is not None and thread_id == message_id:
        thread_id = None

    metadata = {k: v for k, v in record.items() if k not in _CONSUMED_KEYS}

    return {
        "message_id": message_id,
        "source": source,
        "channel": _as_str(_first(record, "channel")),
        "author": author,
        "author_id": author_id,
        "content": clean_text(_first(record, "content")),
        "timestamp": parse_timestamp(_first(record, "timestamp")),
        "thread_id": thread_id,
        "reactions": count_reactions(record.get("reactions")),
        "category": (_as_str(_first(record, "category")) or None),
        "sentiment": (_as_str(_first(record, "sentiment")) or None),
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def empty_frame() -> pd.DataFrame:
    """Return an empty ``DataFrame`` with the common schema and proper dtypes."""
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in COMMON_SCHEMA})
    frame["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    frame["reactions"] = pd.Series(dtype="int64")
    return frame


def normalize(
    records: Iterable[RawRecord],
    *,
    default_source: str = "webhook",
    drop_empty: bool = True,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Convert raw records into a clean ``DataFrame`` following :data:`COMMON_SCHEMA`.

    Steps:

    1. Resolve platform-specific field names onto the common schema.
    2. Clean text, parse timestamps to UTC and count reactions.
    3. Lower-case ``source``, ``category`` and ``sentiment`` labels.
    4. Drop rows with empty content (optional) and duplicate
       ``(source, message_id)`` pairs (optional, keeps the first occurrence).
    5. Sort chronologically, unparseable timestamps last.

    Args:
        records: Raw records from any loader.
        default_source: Source tag for records that do not declare one.
        drop_empty: Remove rows whose cleaned ``content`` is empty.
        deduplicate: Remove repeated ``(source, message_id)`` rows.

    Returns:
        A ``DataFrame`` whose columns are exactly :data:`COMMON_SCHEMA`.

    Raises:
        NormalizationError: If any record is not a mapping.
    """
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise NormalizationError(f"Record #{index} is {type(record).__name__}, expected a mapping")
        rows.append(_flatten(record, default_source))

    if not rows:
        return empty_frame()

    frame = pd.DataFrame(rows, columns=list(COMMON_SCHEMA))

    for column in ("source", "category", "sentiment"):
        frame[column] = frame[column].map(lambda v: v.lower() if isinstance(v, str) else None)
    frame["reactions"] = frame["reactions"].fillna(0).astype("int64")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    if drop_empty:
        before = len(frame)
        frame = frame[frame["content"].str.len() > 0]
        dropped = before - len(frame)
        if dropped:
            logger.info("Dropped %d record(s) with empty content", dropped)

    if deduplicate:
        before = len(frame)
        has_id = frame["message_id"].notna()
        frame = pd.concat(
            [frame[has_id].drop_duplicates(subset=["source", "message_id"], keep="first"), frame[~has_id]]
        )
        dropped = before - len(frame)
        if dropped:
            logger.info("Dropped %d duplicate record(s)", dropped)

    frame = frame.sort_values("timestamp", na_position="last", kind="stable").reset_index(drop=True)
    return frame[list(COMMON_SCHEMA)]


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a normalized frame back to plain dicts with ``None`` instead of NaN/NaT.

    Useful right before Pydantic validation or JSON serialization.
    """
    if frame.empty:
        return []
    cleaned = frame.astype(object).where(frame.notna(), None)
    records = cleaned.to_dict(orient="records")
    for record in records:
        ts = record.get("timestamp")
        if isinstance(ts, pd.Timestamp):
            record["timestamp"] = ts.floor("us").to_pydatetime()
        reactions = record.get("reactions")
        if reactions is not None:
            record["reactions"] = int(reactions)
        if record.get("metadata") is None:
            record["metadata"] = {}
    return records
