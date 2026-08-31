"""Evaluation entry point for reproducible held-out metrics.

Loads a checkpoint saved by ``geoshield.train`` (model architecture and
image size come from the checkpoint's own recorded config, not CLI flags,
so evaluation can't silently mismatch what a checkpoint was trained as) and
reports the same confusion-matrix-derived metrics
(``damage_macro_f1``, per-class F1/IoU, localization F1) the training loop
already computes and tests, run once over a held-out split rather than a
mini-batch during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainConfig
from .dataset import PreparedTileDataset, load_records, record_ids_for_split
from .models import PostOnlyUNet, SiameseUNet, require_torch, torch
from .train import _loader, _run_epoch, select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GeoShield checkpoint on a held-out split.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, help="Prepared tile directory containing records.json")
    parser.add_argument("--split-manifest", type=Path, help="Defaults to <data>/split_manifest.json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/evaluation.json"))
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    require_torch()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    data_root = args.data if args.data.is_dir() else args.data.parent
    records_path = args.data / "records.json" if args.data.is_dir() else args.data
    if not records_path.exists():
        raise FileNotFoundError(f"Prepared records.json not found at {records_path}")
    manifest_path = args.split_manifest or (data_root / "split_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("A split_manifest.json is required for evaluation")

    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_name = config.get("model")
    if model_name not in ("post_only", "siamese"):
        raise ValueError(f"Checkpoint config is missing a valid model name: {model_name!r}")
    image_size = int(config.get("image_size", TrainConfig.image_size))

    records = load_records(records_path)
    split_ids = record_ids_for_split(manifest_path, args.split)
    split_records = [row for row in records if str(row["identifier"]) in split_ids]
    if not split_records:
        raise ValueError(f"No prepared records matched split {args.split!r} in {manifest_path}")

    dataset = PreparedTileDataset(
        split_records, data_root, training=False, image_size=image_size, seed=int(config.get("seed", 42))
    )
    loader = _loader(dataset, args.batch_size, shuffle=False)

    model_class = PostOnlyUNet if model_name == "post_only" else SiameseUNet
    model = model_class(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # class_weights=None: unlike training, evaluation reports metrics on the
    # true class distribution, not a loss shaped by training-time rebalancing.
    loss, metrics = _run_epoch(model, loader, None, None, model_name, device, training=False)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "model": model_name,
        "split": args.split,
        "num_records": len(split_records),
        "num_tiles": len(dataset),
        "loss": loss,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is None or args.data is None:
        print("Evaluation scaffold ready. Supply --checkpoint and --data after training.")
        return 0
    result = evaluate(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
