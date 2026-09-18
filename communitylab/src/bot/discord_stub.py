"""Architecture placeholder for the Discord bot.

The real bot (``discord.py``) is out of scope for now; this stub fixes the
contract so the rest of the system can be wired against it:

* :class:`DiscordBotStub` collects messages from ``on_message`` callbacks into
  an in-memory buffer shaped like the Discord export the ingestion
  loader already understands.
* :meth:`DiscordBotStub.flush` hands the buffer to the LangGraph pipeline
  (``source="discord"``) and returns the run summary, which a future
  implementation will post back to a staff channel.

No network calls are made and ``discord.py`` is not imported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from src.orchestration.router import PipelineOptions, State, compile_pipeline, initial_state, summarize_state

logger = logging.getLogger(__name__)

GraphFactory = Callable[[], CompiledStateGraph]


@dataclass
class DiscordBotConfig:
    """Settings the eventual bot will read from the environment."""

    token_env_var: str = "DISCORD_BOT_TOKEN"
    command_prefix: str = "!"
    watched_channels: tuple[str, ...] = ()
    staff_channel: str | None = None
    batch_size: int = 25
    options: PipelineOptions = field(default_factory=lambda: {"language": "en"})


@dataclass
class DiscordBotStub:
    """Buffers Discord messages and forwards batches to the pipeline.

    Args:
        config: Bot settings.
        graph_factory: Builds the compiled graph (lazily, on first flush).
    """

    config: DiscordBotConfig = field(default_factory=DiscordBotConfig)
    graph_factory: GraphFactory = compile_pipeline
    _buffer: list[dict[str, Any]] = field(default_factory=list, init=False)
    _graph: CompiledStateGraph | None = field(default=None, init=False)
    _last_summary: dict[str, Any] | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (mirror discord.py's Client events)
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Would log in with the token from ``config.token_env_var``; no-op here."""
        logger.info("DiscordBotStub.start(): not connecting (stub)")

    async def close(self) -> None:
        """Would disconnect the gateway; no-op here."""
        logger.info("DiscordBotStub.close(): nothing to close (stub)")

    async def on_ready(self) -> None:
        """Gateway ready event."""
        logger.info("DiscordBotStub ready; watching channels %s", self.config.watched_channels or "(all)")

    async def on_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Buffer one message (Discord API shape: ``id``, ``content``, ``author``, ``channel`` …).

        Messages starting with the command prefix are treated as commands and
        never analysed. When the buffer reaches ``batch_size`` it is flushed.
        """
        content = str(message.get("content", ""))
        channel = _channel_name(message)
        if content.startswith(self.config.command_prefix):
            return await self.on_command(content[len(self.config.command_prefix) :].strip(), message)
        if self.config.watched_channels and channel not in self.config.watched_channels:
            return None
        self._buffer.append(dict(message))
        if len(self._buffer) >= self.config.batch_size:
            return await self.flush()
        return None

    async def on_command(self, command: str, message: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Minimal command surface kept for the future bot (``!flush``, ``!status``)."""
        if command == "flush":
            return await self.flush()
        if command == "status":
            return {"buffered": len(self._buffer), "last_run": self._last_summary}
        logger.debug("Ignoring unknown command %r", command)
        return None

    # ------------------------------------------------------------------ #
    # Pipeline bridge
    # ------------------------------------------------------------------ #
    @property
    def buffered(self) -> int:
        """Number of messages waiting to be analysed."""
        return len(self._buffer)

    def graph(self) -> CompiledStateGraph:
        """The compiled pipeline (built on first use)."""
        if self._graph is None:
            self._graph = self.graph_factory()
        return self._graph

    async def flush(self) -> dict[str, Any] | None:
        """Run the pipeline over the buffered messages and clear the buffer."""
        if not self._buffer:
            return None
        payload, self._buffer = list(self._buffer), []
        state: State = initial_state("discord", payload, options=self.config.options)
        final = self.graph().invoke(state)
        self._last_summary = summarize_state(final)
        logger.info("Flushed %d Discord message(s): %s", len(payload), self._last_summary["highest_level"])
        await self.notify_staff(self._last_summary)
        return self._last_summary

    async def notify_staff(self, summary: dict[str, Any]) -> str:
        """Format the summary for the staff channel (sending is left to the real bot)."""
        text = format_staff_message(summary)
        logger.info("[%s] %s", self.config.staff_channel or "staff", text.splitlines()[0])
        return text


def _channel_name(message: dict[str, Any]) -> str:
    channel = message.get("channel")
    if isinstance(channel, dict):
        return str(channel.get("name") or channel.get("id") or "")
    return str(channel or message.get("channel_id") or "")


def format_staff_message(summary: dict[str, Any]) -> str:
    """Discord-flavoured Markdown digest of a run summary."""
    lines = [
        f"**CommunityLab run `{summary.get('run_id')}`** — {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        f"Messages: {summary.get('validated', 0)} analysed, highest level **{summary.get('highest_level') or 'n/a'}**",
        f"Actions: {summary.get('actions', 0)} · Assets: {len(summary.get('generated_assets', []))}",
    ]
    if summary.get("executive_summary"):
        lines += ["", str(summary["executive_summary"])]
    if summary.get("errors"):
        lines += ["", "Warnings:"] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(lines)
