"""Shared fixtures for the CommunityLab test-suite."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.ingest.validator import CommunityMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MESSAGES = PROJECT_ROOT / "data" / "sample" / "messages.json"


class FakeToolChatModel(BaseChatModel):
    """Offline stand-in for ``ChatGroq`` that supports ``with_structured_output``.

    Every call answers with a single tool call carrying ``payload`` (so the
    structured-output parser turns it into the requested Pydantic model), or
    raises ``RuntimeError`` when ``fail`` is set. ``calls`` counts invocations
    so tests can assert on fallback behaviour.
    """

    payload: dict[str, Any]
    name: str = "fake"
    fail: bool = False
    calls: int = 0
    tool_name: str = "structured_output"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeToolChatModel:  # noqa: ARG002
        first = tools[0] if tools else None
        if isinstance(first, type):
            self.tool_name = first.__name__
        elif isinstance(first, dict):
            self.tool_name = first.get("name") or first.get("function", {}).get("name", self.tool_name)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        message = AIMessage(
            content="",
            tool_calls=[{"name": self.tool_name, "args": dict(self.payload), "id": "call_1", "type": "tool_call"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat-model"


@pytest.fixture
def community_message() -> CommunityMessage:
    """A single validated message, ready for the analysis layer."""
    return CommunityMessage(
        message_id="m-1",
        source="discord",
        channel="help-python",
        author="alice",
        author_id="u_1",
        content="I keep getting ModuleNotFoundError for langgraph even after pip install. Any idea?",
        timestamp=datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
        reactions=2,
    )


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
