"""Tests for the Discord bot architecture stub."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from src.bot import DiscordBotConfig, DiscordBotStub, format_staff_message
from src.orchestration import PipelineComponents, compile_pipeline

from tests.test_orchestration import consolidator, generators, local_storage


@pytest.fixture
def bot(tmp_path: Path) -> DiscordBotStub:
    def factory():
        comps = PipelineComponents(
            consolidator=consolidator(), generators=generators(), storage=local_storage(tmp_path)
        )
        return compile_pipeline(comps)

    config = DiscordBotConfig(watched_channels=("general",), batch_size=2, options={"channels": ["testimonial"]})
    return DiscordBotStub(config=config, graph_factory=factory)


def run(coro):
    return asyncio.run(coro)


class TestBuffering:
    def test_buffers_only_watched_channels(self, bot: DiscordBotStub, discord_payload: list[dict[str, Any]]):
        msg = dict(discord_payload[0])
        assert run(bot.on_message({**msg, "channel": {"name": "general"}})) is None
        assert run(bot.on_message({**msg, "channel": {"name": "random"}})) is None
        assert run(bot.on_message({**msg, "channel_id": "12345"})) is None
        assert bot.buffered == 1

    def test_commands_are_not_buffered(self, bot: DiscordBotStub):
        assert run(bot.on_message({"content": "!status", "channel": {"name": "general"}})) == {
            "buffered": 0,
            "last_run": None,
        }
        assert run(bot.on_message({"content": "!unknown", "channel": {"name": "general"}})) is None
        assert bot.buffered == 0

    def test_flush_on_empty_buffer_is_noop(self, bot: DiscordBotStub):
        assert run(bot.flush()) is None
        assert bot._graph is None

    def test_lifecycle_hooks_are_noops(self, bot: DiscordBotStub):
        run(bot.start())
        run(bot.on_ready())
        run(bot.close())
        assert bot._graph is None


class TestPipelineBridge:
    def test_batch_size_triggers_flush(self, bot: DiscordBotStub, discord_payload: list[dict[str, Any]]):
        messages = [{**m, "channel": {"name": "general"}} for m in discord_payload[:2]]
        results = [run(bot.on_message(m)) for m in messages]
        assert results[0] is None
        summary = results[1]
        assert summary is not None and summary["source"] == "discord" and summary["validated"] == 2
        assert {a["channel"] for a in summary["generated_assets"]} <= {"testimonial"}
        assert bot.buffered == 0

    def test_flush_command_runs_pipeline(self, bot: DiscordBotStub, discord_payload: list[dict[str, Any]]):
        run(bot.on_message({**discord_payload[0], "channel": {"name": "general"}}))
        summary = run(bot.on_message({"content": "!flush", "channel": {"name": "general"}}))
        assert summary is not None and summary["validated"] == 1
        status = run(bot.on_message({"content": "!status", "channel": {"name": "general"}}))
        assert status == {"buffered": 0, "last_run": summary}


def test_format_staff_message():
    text = format_staff_message(
        {
            "run_id": "abc",
            "validated": 4,
            "highest_level": "ALTO",
            "actions": 2,
            "generated_assets": [{"channel": "faq"}],
            "executive_summary": "Summary here.",
            "errors": ["analyze: partial"],
        }
    )
    assert text.startswith("**CommunityLab run `abc`**")
    assert "highest level **ALTO**" in text and "Actions: 2 · Assets: 1" in text
    assert "Summary here." in text and "- analyze: partial" in text
