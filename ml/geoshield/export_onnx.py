"""ONNX export entry point for the validated Siamese checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a GeoShield Siamese model to ONNX.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("public/models/geoshield-siamese.onnx"))
    parser.add_argument("--opset", type=int, default=18)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is None:
        print("ONNX export scaffold ready. Supply --checkpoint after model validation.")
        return 0
    raise NotImplementedError("ONNX export is implemented after the training and parity gates.")


if __name__ == "__main__":
    raise SystemExit(main())
