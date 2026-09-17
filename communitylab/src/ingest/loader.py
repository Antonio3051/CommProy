"""Loaders for the CommunityLab ingestion layer.

Every loader returns a ``list[RawRecord]`` where ``RawRecord`` is a plain
``dict`` exactly as it came from the source, plus a ``source`` key that tells
the normalizer which platform produced it. Loaders never clean or reshape the
payload; that is the job of :mod:`src.ingest.normalizer`.

Supported sources:

* Flat files: :func:`load_json`, :func:`load_csv`
* Discord:    :func:`load_discord` (payload) / :func:`fetch_discord_channel` (API)
* Slack:      :func:`load_slack`   (payload) / :func:`fetch_slack_channel`   (API)
* Webhooks:   :func:`load_webhook`
* Dispatcher: :func:`load`
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

type RawRecord = dict[str, Any]
"""A single message as returned by its source, untouched except for ``source``."""

SourceName = Literal["json", "csv", "discord", "slack", "webhook"]
"""Identifiers for the platforms the ingestion layer understands."""

HttpGet = Callable[[str, Mapping[str, str]], bytes]
"""Signature of the HTTP GET function used by the ``fetch_*`` helpers.

Takes ``(url, headers)`` and returns the raw response body. Injected so the
network layer can be swapped out in tests.
"""

type PathLike = str | Path

_DISCORD_API = "https://discord.com/api/v10"
_SLACK_API = "https://slack.com/api"
_DEFAULT_TIMEOUT_SECONDS = 30


class LoaderError(ValueError):
    """Raised when a payload or file cannot be interpreted as message records."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _tag(records: Iterable[Mapping[str, Any]], source: SourceName) -> list[RawRecord]:
    """Copy each record into a fresh ``dict`` and stamp it with ``source``.

    A pre-existing ``source`` key is preserved so mixed-source files (like the
    bundled sample dataset) keep their per-record origin.
    """
    tagged: list[RawRecord] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise LoaderError(f"Expected a mapping per record, got {type(record).__name__}")
        item: RawRecord = dict(record)
        item.setdefault("source", source)
        tagged.append(item)
    return tagged


def _unwrap_messages(payload: Any, *, keys: Sequence[str] = ("messages", "data", "items")) -> list[Any]:
    """Return the list of records inside ``payload``.

    Accepts a bare list, a single record, or an envelope such as
    ``{"messages": [...]}`` / ``{"data": [...]}``.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in keys:
            if isinstance(payload.get(key), list):
                return list(payload[key])
        return [payload]
    raise LoaderError(f"Cannot extract records from payload of type {type(payload).__name__}")


def _parse_json(body: str | bytes | bytearray) -> Any:
    """Decode a JSON document, raising :class:`LoaderError` on failure."""
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LoaderError(f"Invalid JSON payload: {exc.msg} (pos {exc.pos})") from exc


def _default_http_get(url: str, headers: Mapping[str, str]) -> bytes:
    """Minimal stdlib HTTP GET used when no custom ``http_get`` is supplied."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=_DEFAULT_TIMEOUT_SECONDS) as response:
        return response.read()


# --------------------------------------------------------------------------- #
# Flat files
# --------------------------------------------------------------------------- #
def load_json(path: PathLike, *, encoding: str = "utf-8") -> list[RawRecord]:
    """Load community messages from a JSON file.

    The document may be a list of records or an envelope with a ``messages`` /
    ``data`` / ``items`` list. Records without a ``source`` key are tagged as
    ``"json"``.

    Args:
        path: Location of the ``.json`` file.
        encoding: Text encoding of the file.

    Returns:
        The raw records contained in the file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        LoaderError: If the file is not valid JSON or has an unexpected shape.
    """
    file_path = Path(path)
    logger.debug("Loading JSON file %s", file_path)
    payload = _parse_json(file_path.read_text(encoding=encoding))
    return _tag(_unwrap_messages(payload), "json")


def load_csv(path: PathLike, *, encoding: str = "utf-8", **read_csv_kwargs: Any) -> list[RawRecord]:
    """Load community messages from a CSV file using Pandas.

    Each row becomes one record. Empty cells are converted to ``None`` so the
    normalizer can treat them uniformly. Extra keyword arguments are forwarded
    to :func:`pandas.read_csv` (e.g. ``sep=";"``).

    Args:
        path: Location of the ``.csv`` file.
        encoding: Text encoding of the file.
        **read_csv_kwargs: Passed through to :func:`pandas.read_csv`.

    Returns:
        The raw records contained in the file, tagged with ``source="csv"``
        unless the CSV already has a ``source`` column.
    """
    file_path = Path(path)
    logger.debug("Loading CSV file %s", file_path)
    frame = pd.read_csv(file_path, encoding=encoding, dtype=str, keep_default_na=False, **read_csv_kwargs)
    frame = frame.replace({"": None})
    records = frame.to_dict(orient="records")
    return _tag(records, "csv")


# --------------------------------------------------------------------------- #
# Discord
# --------------------------------------------------------------------------- #
def load_discord(payload: Any, *, channel: str | None = None) -> list[RawRecord]:
    """Wrap Discord message objects as raw records.

    ``payload`` is whatever the Discord REST API (``GET /channels/{id}/messages``)
    or a DiscordChatExporter dump returned: a list of message objects, or an
    envelope with a ``messages`` list. Message objects are left intact; the
    normalizer knows how to read ``author.username``, ``message_reference`` and
    ``reactions[].emoji.name``.

    Args:
        payload: Parsed JSON from Discord (``list`` or ``dict``) or a JSON string.
        channel: Optional human-readable channel name to attach to every record
            (Discord messages only carry ``channel_id``).

    Returns:
        The Discord messages tagged with ``source="discord"``.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        payload = _parse_json(payload)
    records = _tag(_unwrap_messages(payload), "discord")
    if channel is not None:
        for record in records:
            record.setdefault("channel", channel)
    return records


def fetch_discord_channel(
    channel_id: str,
    bot_token: str,
    *,
    limit: int = 100,
    channel: str | None = None,
    http_get: HttpGet = _default_http_get,
) -> list[RawRecord]:
    """Fetch the most recent messages of a Discord channel through the REST API.

    Args:
        channel_id: Snowflake id of the channel.
        bot_token: Bot token with ``READ_MESSAGE_HISTORY`` permission.
        limit: Number of messages to request (Discord caps this at 100).
        channel: Optional human-readable channel name to attach to the records.
        http_get: HTTP GET implementation; defaults to a stdlib ``urllib`` call.

    Returns:
        The fetched messages tagged with ``source="discord"``.
    """
    query = urllib.parse.urlencode({"limit": max(1, min(limit, 100))})
    url = f"{_DISCORD_API}/channels/{channel_id}/messages?{query}"
    headers = {"Authorization": f"Bot {bot_token}", "User-Agent": "CommunityLab/0.1"}
    logger.info("Fetching up to %d Discord messages from channel %s", limit, channel_id)
    body = http_get(url, headers)
    return load_discord(_parse_json(body), channel=channel)


# --------------------------------------------------------------------------- #
# Slack
# --------------------------------------------------------------------------- #
def load_slack(payload: Any, *, channel: str | None = None) -> list[RawRecord]:
    """Wrap Slack message objects as raw records.

    Accepts the JSON returned by ``conversations.history`` /
    ``conversations.replies`` (``{"ok": true, "messages": [...]}``), a bare list
    of Slack message objects, or a single message. Records keep their Slack
    fields (``ts``, ``user``, ``text``, ``thread_ts``, ``reactions``).

    Args:
        payload: Parsed Slack JSON (``list`` or ``dict``) or a JSON string.
        channel: Optional channel name to attach to every record.

    Returns:
        The Slack messages tagged with ``source="slack"``.

    Raises:
        LoaderError: If the payload is a Slack API error response.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        payload = _parse_json(payload)
    if isinstance(payload, Mapping) and payload.get("ok") is False:
        raise LoaderError(f"Slack API error: {payload.get('error', 'unknown_error')}")
    records = _tag(_unwrap_messages(payload), "slack")
    if channel is not None:
        for record in records:
            record.setdefault("channel", channel)
    return records


def fetch_slack_channel(
    channel_id: str,
    bot_token: str,
    *,
    limit: int = 200,
    channel: str | None = None,
    http_get: HttpGet = _default_http_get,
) -> list[RawRecord]:
    """Fetch recent messages of a Slack channel via ``conversations.history``.

    Args:
        channel_id: Slack channel id (e.g. ``C0123ABCD``).
        bot_token: Bot token with the ``channels:history`` scope.
        limit: Number of messages to request (Slack caps this at 999).
        channel: Optional channel name to attach to the records.
        http_get: HTTP GET implementation; defaults to a stdlib ``urllib`` call.

    Returns:
        The fetched messages tagged with ``source="slack"``.
    """
    query = urllib.parse.urlencode({"channel": channel_id, "limit": max(1, min(limit, 999))})
    url = f"{_SLACK_API}/conversations.history?{query}"
    headers = {"Authorization": f"Bearer {bot_token}", "User-Agent": "CommunityLab/0.1"}
    logger.info("Fetching up to %d Slack messages from channel %s", limit, channel_id)
    body = http_get(url, headers)
    return load_slack(_parse_json(body), channel=channel)


# --------------------------------------------------------------------------- #
# Webhooks
# --------------------------------------------------------------------------- #
def load_webhook(
    body: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any], *, source: SourceName = "webhook"
) -> list[RawRecord]:
    """Turn an inbound webhook body (e.g. from n8n or a custom bot) into records.

    Args:
        body: Raw request body (JSON string/bytes) or the already-parsed JSON.
            A single object, a list, or an envelope with ``messages`` / ``data``
            / ``items`` are all accepted.
        source: Source tag applied to records that do not declare their own.

    Returns:
        The records carried by the webhook.

    Raises:
        LoaderError: If the body is not valid JSON or has an unexpected shape.
    """
    payload = _parse_json(body) if isinstance(body, (str, bytes, bytearray)) else body
    return _tag(_unwrap_messages(payload), source)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def load(source: SourceName, target: Any, **kwargs: Any) -> list[RawRecord]:
    """Load records from ``target`` using the loader registered for ``source``.

    Args:
        source: One of ``"json"``, ``"csv"``, ``"discord"``, ``"slack"``, ``"webhook"``.
        target: A file path for ``json``/``csv``; a payload for the rest.
        **kwargs: Forwarded to the underlying loader.

    Returns:
        The raw records produced by the selected loader.

    Raises:
        LoaderError: If ``source`` is not a known loader.
    """
    loaders: dict[str, Callable[..., list[RawRecord]]] = {
        "json": load_json,
        "csv": load_csv,
        "discord": load_discord,
        "slack": load_slack,
        "webhook": load_webhook,
    }
    try:
        loader = loaders[source]
    except KeyError as exc:
        raise LoaderError(f"Unknown source '{source}'. Expected one of {sorted(loaders)}") from exc
    return loader(target, **kwargs)
