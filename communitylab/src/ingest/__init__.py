"""Ingestion layer: load raw community data, normalize it, validate it.

raw = load_json("data/sample/messages.json")
frame = normalize(raw)
result = validate(frame)
"""

from src.ingest.loader import (
    HttpGet,
    LoaderError,
    RawRecord,
    SourceName,
    fetch_discord_channel,
    fetch_slack_channel,
    load,
    load_csv,
    load_discord,
    load_json,
    load_slack,
    load_webhook,
)
from src.ingest.normalizer import (
    COMMON_SCHEMA,
    NormalizationError,
    clean_text,
    count_reactions,
    empty_frame,
    normalize,
    parse_timestamp,
    to_records,
)
from src.ingest.validator import (
    CommunityMessage,
    InvalidBatchError,
    RowError,
    Sentiment,
    ValidationResult,
    to_frame,
    validate,
    validate_records,
)

__all__ = [
    "COMMON_SCHEMA",
    "CommunityMessage",
    "HttpGet",
    "InvalidBatchError",
    "LoaderError",
    "NormalizationError",
    "RawRecord",
    "RowError",
    "Sentiment",
    "SourceName",
    "ValidationResult",
    "clean_text",
    "count_reactions",
    "empty_frame",
    "fetch_discord_channel",
    "fetch_slack_channel",
    "load",
    "load_csv",
    "load_discord",
    "load_json",
    "load_slack",
    "load_webhook",
    "normalize",
    "parse_timestamp",
    "to_frame",
    "to_records",
    "validate",
    "validate_records",
]
