"""Run the full LangGraph pipeline (ingest → analyse → decide → generate → store) on a file.

Needs ``GROQ_API_KEY`` for the analysis and generation nodes. Storage goes to
OCI when ``~/.oci/config`` is valid, otherwise to ``data/communitylab-activos-marketing/``.

Usage::

    python scripts/run_pipeline.py data/sample/messages.json
    python scripts/run_pipeline.py export.csv --source csv --channels linkedin faq --language es --local
    python scripts/run_pipeline.py data/sample/messages.json --skip-generate --json /tmp/run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.oci import AssetStorage, StorageSettings  # noqa: E402
from src.orchestration import (  # noqa: E402
    ALL_CHANNELS,
    PipelineComponents,
    PipelineOptions,
    compile_pipeline,
    initial_state,
    summarize_state,
)
from src.utils.llm_clients import MissingAPIKeyError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="JSON or CSV file with raw community messages.")
    parser.add_argument("--source", choices=("json", "csv"), default="json")
    parser.add_argument("--channels", nargs="+", choices=ALL_CHANNELS, default=list(ALL_CHANNELS))
    parser.add_argument("--language", default="en")
    parser.add_argument("--community", default="the community")
    parser.add_argument("--max-posts", type=int, default=3)
    parser.add_argument("--force-generate", action="store_true", help="Run generators even with no trigger.")
    parser.add_argument("--skip-generate", action="store_true", help="Decisions only; no content generation.")
    parser.add_argument("--local", action="store_true", help="Force local storage (skip OCI).")
    parser.add_argument("--json", type=Path, help="Write the run summary to this file.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    storage = AssetStorage(StorageSettings(force_local=True)) if args.local else AssetStorage()
    try:
        components = PipelineComponents.default(storage=storage)
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    options: PipelineOptions = {
        "language": args.language,
        "community_name": args.community,
        "channels": tuple(args.channels),
        "max_posts": args.max_posts,
        "force_generate": args.force_generate,
        "skip_generate": args.skip_generate,
    }
    graph = compile_pipeline(components)
    final = graph.invoke(initial_state(args.source, args.path, options=options))
    summary = summarize_state(final)

    print(f"run {summary['run_id']}: {summary['raw_records']} raw -> {summary['validated']} valid -> ", end="")
    print(f"{summary['analyzed']} analysed; highest level {summary['highest_level']}; {summary['actions']} actions")
    for asset in summary["generated_assets"]:
        print(f"  asset  {asset['channel']:<11} {asset['title']}")
    for stored in summary["stored"]:
        print(f"  stored [{stored['backend']}] {stored['location']}")
    for err in summary["errors"]:
        print(f"  warn   {err}")
    if summary["executive_summary"]:
        print("\n" + summary["executive_summary"])
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsummary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
