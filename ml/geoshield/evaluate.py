"""Evaluation entry point for reproducible held-out metrics."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GeoShield checkpoint on a held-out split.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/evaluation.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is None or args.data is None:
        print("Evaluation scaffold ready. Supply --checkpoint and --data after training.")
        return 0
    raise NotImplementedError("Evaluation loop is implemented after the training gate.")


if __name__ == "__main__":
    raise SystemExit(main())
