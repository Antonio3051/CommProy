"""Run the ingestion pipeline on a JSON/CSV file and print a summary.

Usage::

    python scripts/ingest_sample.py                      # uses data/sample/messages.json
    python scripts/ingest_sample.py path/to/messages.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import load_csv, load_json, normalize, validate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        nargs="?",
        default=PROJECT_ROOT / "data" / "sample" / "messages.json",
        type=Path,
        help="JSON or CSV file to ingest (default: bundled sample).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    raw = load_csv(args.path) if args.path.suffix.lower() == ".csv" else load_json(args.path)
    frame = normalize(raw)
    result = validate(frame)

    print(f"Loaded {len(raw)} raw record(s) -> {len(frame)} normalized row(s) -> {len(result.valid)} valid message(s)")
    print(
        frame[
            ["message_id", "source", "channel", "author", "timestamp", "reactions", "category", "sentiment"]
        ].to_string()
    )
    for error in result.errors:
        print(f"REJECTED {error}", file=sys.stderr)
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
