"""Shared fixtures for the CommunityLab test-suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MESSAGES = PROJECT_ROOT / "data" / "sample" / "messages.json"


@pytest.fixture(scope="session")
def sample_path() -> Path:
    """Path to the bundled ``data/sample/messages.json``."""
    return SAMPLE_MESSAGES


@pytest.fixture(scope="session")
def sample_payload(sample_path: Path) -> list[dict[str, Any]]:
    """Parsed contents of the bundled sample dataset."""
    return json.loads(sample_path.read_text(encoding="utf-8"))


@pytest.fixture
def discord_payload() -> list[dict[str, Any]]:
    """Two messages shaped like the Discord REST API (``GET /channels/{id}/messages``)."""
    return [
        {
            "id": "1300000000000000001",
            "channel_id": "990000000000000001",
            "author": {"id": "u_1", "username": "alice", "global_name": "Alice"},
            "content": "  How do I   configure  the OCI CLI? \u200b",
            "timestamp": "2026-09-10T10:00:00.123000+00:00",
            "reactions": [{"emoji": {"name": "👀"}, "count": 2}, {"emoji": {"name": "🙏"}, "count": 1}],
            "type": 0,
        },
        {
            "id": "1300000000000000002",
            "channel_id": "990000000000000001",
            "author": {"id": "u_2", "username": "bob"},
            "content": "Run `oci setup config` first.",
            "timestamp": "2026-09-10T10:05:00+00:00",
            "message_reference": {"message_id": "1300000000000000001"},
            "type": 19,
        },
    ]


@pytest.fixture
def slack_payload() -> dict[str, Any]:
    """A ``conversations.history`` response with a thread parent and a reply."""
    return {
        "ok": True,
        "messages": [
            {
                "type": "message",
                "ts": "1726300987.004500",
                "thread_ts": "1726300987.004500",
                "user": "U07AB3CD9",
                "user_profile": {"real_name": "Priya N", "display_name": "priya"},
                "text": "Shipped my first pipeline!",
                "reactions": [{"name": "tada", "count": 3, "users": ["U1", "U2", "U3"]}],
            },
            {
                "type": "message",
                "ts": "1726301000.000100",
                "thread_ts": "1726300987.004500",
                "user": "U0ZZ",
                "text": "Congrats!",
            },
        ],
    }
