"""Generate marketing assets from enriched records and store them (OCI or local ``data/``).

Input is a JSON file of enriched records as written by
``scripts/analyze_sample.py --json out.json``. Generation needs ``GROQ_API_KEY``;
storage uses OCI when ``~/.oci/config`` is valid and otherwise writes to
``data/communitylab-activos-marketing/``.

Usage::

    python scripts/analyze_sample.py --json /tmp/enriched.json
    python scripts/generate_sample.py /tmp/enriched.json
    python scripts/generate_sample.py /tmp/enriched.json --only linkedin faq --language es --local
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import EnrichedMessage  # noqa: E402
from src.decisions import DecisionEngine  # noqa: E402
from src.generators import (  # noqa: E402
    FAQGenerator,
    GeneratedAsset,
    LinkedInGenerator,
    NewsletterGenerator,
    TestimonialGenerator,
)
from src.oci import AssetStorage, StorageSettings  # noqa: E402
from src.utils.llm_clients import MissingAPIKeyError  # noqa: E402

CHANNELS = ("linkedin", "newsletter", "faq", "testimonial")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="JSON file with enriched records.")
    parser.add_argument("--only", nargs="+", choices=CHANNELS, default=CHANNELS, help="Channels to generate.")
    parser.add_argument("--language", default="en", help="Output language (default: en).")
    parser.add_argument("--limit", type=int, default=3, help="Max LinkedIn posts / testimonials (default: 3).")
    parser.add_argument("--local", action="store_true", help="Skip OCI and save under data/ only.")
    parser.add_argument("--community", default="the community", help="Community name for the newsletter.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    enriched = [EnrichedMessage.from_record(r) for r in json.loads(args.path.read_text(encoding="utf-8"))]
    if not enriched:
        print("No enriched records found.", file=sys.stderr)
        return 1

    assets: list[GeneratedAsset] = []
    try:
        if "linkedin" in args.only:
            assets += LinkedInGenerator().generate_many(enriched, language=args.language, limit=args.limit)
        if "newsletter" in args.only:
            try:
                assets.append(
                    NewsletterGenerator().generate(enriched, language=args.language, community_name=args.community)
                )
            except ValueError as exc:
                print(f"Newsletter skipped: {exc}", file=sys.stderr)
        if "faq" in args.only:
            report = DecisionEngine().run(enriched)
            assets += FAQGenerator().generate_many(report.recurring_topics, enriched, language=args.language)
        if "testimonial" in args.only:
            assets += TestimonialGenerator().generate_many(enriched, language=args.language, limit=args.limit)
    except MissingAPIKeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not assets:
        print("Nothing to generate for the selected channels.", file=sys.stderr)
        return 1

    storage = AssetStorage(StorageSettings(force_local=True) if args.local else None)
    results = storage.upload_assets(assets)
    print(f"Generated {len(assets)} asset(s); stored via {'OCI' if storage.is_remote else 'local fallback'}:")
    for result in results:
        print(f"  [{result.backend}] {result.location}")
    for asset in assets:
        print(f"\n=== {asset.channel}: {asset.title}\n{asset.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
