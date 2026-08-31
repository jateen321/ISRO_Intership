"""Prepare xBD records and tiles (full rasterization is added in Step 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and prepare an xBD dataset directory.")
    parser.add_argument("--input", type=Path, required=False, help="Path to the downloaded xBD training directory")
    parser.add_argument("--output", type=Path, required=False, help="Directory for generated records and tiles")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input is None or args.output is None:
        print("Preparation scaffold ready. Supply --input and --output after downloading xBD.")
        return 0
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"input": str(args.input), "output": str(args.output), "tile_size": args.tile_size, "limit": args.limit, "status": "audit pending"}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
