"""Run the Decision Engine on enriched records and print the executive summary.

Input is a JSON file with enriched records, as written by
``scripts/analyze_sample.py --json out.json`` (i.e. ``EnrichedMessage.to_record``
rows) or a JSON list of ``EnrichedMessage`` model dumps. Requires no API key.

Usage::

    python scripts/analyze_sample.py --json /tmp/enriched.json   # needs GROQ_API_KEY
    python scripts/decide_sample.py /tmp/enriched.json
    python scripts/decide_sample.py /tmp/enriched.json --notifications --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import EnrichedMessage, RelevanceResult, SentimentResult, ThemesResult  # noqa: E402
from src.decisions import DecisionEngine, EscalationLevel  # noqa: E402
from src.ingest.validator import CommunityMessage  # noqa: E402

MESSAGE_FIELDS = tuple(CommunityMessage.model_fields)


def _from_record(record: dict[str, Any]) -> EnrichedMessage:
    """Rebuild an ``EnrichedMessage`` from a flattened ``to_record`` row."""
    if "message" in record:  # already a model dump
        return EnrichedMessage.model_validate(record)

    def sub(prefix: str) -> dict[str, Any]:
        return {k[len(prefix) :]: v for k, v in record.items() if k.startswith(prefix) and v is not None}

    message = CommunityMessage.model_validate({k: record[k] for k in MESSAGE_FIELDS if k in record})
    sentiment, themes, relevance = sub("sentiment_"), sub("themes_"), sub("relevance_")
    return EnrichedMessage(
        message=message,
        sentiment=SentimentResult.model_validate(sentiment) if sentiment else None,
        themes=ThemesResult.model_validate(themes) if themes else None,
        relevance=RelevanceResult.model_validate(relevance) if relevance else None,
        errors=record.get("errors") or {},
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="JSON file with enriched records.")
    parser.add_argument("--now", help="Reference time (ISO 8601) for inactivity/SLA; default: current UTC time.")
    parser.add_argument("--notifications", action="store_true", help="Also print every notification body.")
    parser.add_argument("--json", type=Path, metavar="OUT", help="Write the full DecisionReport to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    enriched = [_from_record(r) for r in payload]
    if not enriched:
        print("No enriched records found.", file=sys.stderr)
        return 1

    reference_time = datetime.fromisoformat(args.now).astimezone(UTC) if args.now else None
    report = DecisionEngine().run(enriched, reference_time=reference_time)

    print(report.executive_summary)
    if args.notifications:
        for notification in report.notifications:
            if notification.kind == "summary":
                continue
            print(f"\n=== {notification.title}\n-> {notification.audience}\n{notification.body}")

    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote decision report to {args.json}")

    return 1 if report.highest_level is EscalationLevel.CRITICO else 0


if __name__ == "__main__":
    raise SystemExit(main())
