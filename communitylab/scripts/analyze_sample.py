"""Run ingestion + AI analysis on a JSON/CSV file and print the enriched rows.

Requires ``GROQ_API_KEY`` in the environment.

Usage::

    python scripts/analyze_sample.py                       # uses data/sample/messages.json
    python scripts/analyze_sample.py path/to/messages.csv --model llama-3.1-8b-instant
    python scripts/analyze_sample.py --json out.json       # also dump the enriched records
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import Consolidator, to_frame, to_records  # noqa: E402
from src.ingest import load_csv, load_json, normalize, validate  # noqa: E402
from src.utils.llm_clients import LLMSettings, MissingAPIKeyError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        nargs="?",
        default=PROJECT_ROOT / "data" / "sample" / "messages.json",
        type=Path,
        help="JSON or CSV file to analyse (default: bundled sample).",
    )
    parser.add_argument("--model", help="Override the primary Groq model.")
    parser.add_argument("--json", type=Path, metavar="OUT", help="Write enriched records to this JSON file.")
    parser.add_argument("--strict", action="store_true", help="Abort on the first analyzer failure.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    raw = load_csv(args.path) if args.path.suffix.lower() == ".csv" else load_json(args.path)
    messages = validate(normalize(raw)).valid
    if not messages:
        print("No valid messages to analyse.", file=sys.stderr)
        return 1

    settings = LLMSettings(primary_model=args.model) if args.model else LLMSettings()
    try:
        settings.resolve_api_key()
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    consolidator = Consolidator(settings=settings, strict=args.strict)
    enriched = consolidator.enrich_many(messages)

    frame = to_frame(enriched)
    columns = [
        "message_id",
        "author",
        "sentiment_label",
        "sentiment_score",
        "themes_primary_category",
        "relevance_score",
        "relevance_tier",
    ]
    present = [c for c in columns if c in frame.columns]
    print(frame[present].to_string(index=False))

    failed = [e for e in enriched if e.errors]
    if failed:
        print(f"\n{len(failed)} message(s) with analyzer errors:", file=sys.stderr)
        for item in failed:
            print(f"  {item.message_id}: {item.errors}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(to_records(enriched), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {len(enriched)} enriched records to {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
