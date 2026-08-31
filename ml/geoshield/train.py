"""Training entry point; dataset loading is implemented in the preparation stage."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GeoShield segmentation model.")
    parser.add_argument("--model", choices=("post_only", "siamese"), required=True)
    parser.add_argument("--data", type=Path, help="Prepared tile directory")
    parser.add_argument("--output", type=Path, default=Path("ml/checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data is None:
        print(f"Training scaffold ready for {args.model}. Supply --data after xBD preparation.")
        return 0
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    raise NotImplementedError("Training loop is implemented after the dataset preparation gate.")


if __name__ == "__main__":
    raise SystemExit(main())
