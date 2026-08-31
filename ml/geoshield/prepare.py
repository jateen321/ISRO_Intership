"""Prepare and audit the official xBD challenge training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .xbd import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and prepare an xBD dataset directory.")
    parser.add_argument("--input", type=Path, required=False, help="Path to the downloaded xBD split")
    parser.add_argument("--output", type=Path, required=False, help="Directory for generated tiles and audit")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input is None or args.output is None:
        print("Preparation ready. Supply --input and --output after manually downloading xBD.")
        return 0
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    audit = prepare_dataset(args.input, args.output, tile_size=args.tile_size, limit=args.limit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
