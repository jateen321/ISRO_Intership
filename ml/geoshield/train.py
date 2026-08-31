"""Train post-only or Siamese GeoShield segmentation models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .config import IGNORE_INDEX, NUM_CLASSES, TrainConfig
from .dataset import PreparedTileDataset, class_pixel_counts, inverse_sqrt_class_weights, load_records, record_ids_for_split
from .losses import combined_segmentation_loss
from .metrics import confusion_matrix, summarize_metrics
from .models import PostOnlyUNet, SiameseUNet, require_torch, torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GeoShield segmentation model.")
    parser.add_argument("--model", choices=("post_only", "siamese"), required=True)
    parser.add_argument("--data", type=Path, help="Prepared tile directory containing records.json")
    parser.add_argument("--split-manifest", type=Path, help="Checksum-bound split_manifest.json")
    parser.add_argument("--output", type=Path, default=Path("ml/checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet encoder weights")
    parser.add_argument("--smoke-test", action="store_true", help="Run an eight-tile synthetic overfit smoke test")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    require_torch()
    return {
        "before": torch.stack([row["before"] for row in batch]),
        "after": torch.stack([row["after"] for row in batch]),
        "mask": torch.stack([row["mask"] for row in batch]),
        "identifier": [row["identifier"] for row in batch],
    }


class _SyntheticDataset:
    """Eight deterministic 64px samples used only by --smoke-test."""

    def __init__(self, image_size: int = 64):
        require_torch()
        self.samples = []
        for index in range(8):
            before = torch.zeros(3, image_size, image_size)
            after = torch.zeros(3, image_size, image_size)
            mask = torch.zeros(image_size, image_size, dtype=torch.long)
            class_id = (index % 4) + 1
            y0, y1 = 12 + (index % 2) * 8, 40 + (index % 2) * 8
            x0, x1 = 12 + (index // 4) * 8, 40 + (index // 4) * 8
            mask[y0:y1, x0:x1] = class_id
            after[:, y0:y1, x0:x1] = class_id / 4.0
            self.samples.append({"before": before, "after": after, "mask": mask, "identifier": f"smoke-{index}"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _loader(dataset, batch_size: int, *, shuffle: bool):
    require_torch()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=_collate)


def _forward(model, batch: dict[str, object], model_name: str, device):
    if model_name == "post_only":
        return model(batch["after"].to(device))
    return model(batch["before"].to(device), batch["after"].to(device))


def _run_epoch(model, loader, optimizer, class_weights, model_name, device, *, training: bool, scaler=None):
    require_torch()
    model.train(training)
    total_loss = 0.0
    batches = 0
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    amp_enabled = device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
    with torch.set_grad_enabled(training):
        for batch in loader:
            target = batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits = _forward(model, batch, model_name, device)
                loss, _ = combined_segmentation_loss(logits, target, class_weights, IGNORE_INDEX)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            prediction = logits.detach().argmax(dim=1).cpu().numpy()
            matrix += confusion_matrix(target.detach().cpu().numpy(), prediction, NUM_CLASSES, IGNORE_INDEX)
            total_loss += float(loss.detach().cpu())
            batches += 1
    if batches == 0:
        raise ValueError("Data loader produced no batches")
    return total_loss / batches, summarize_metrics(matrix)


def _save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, config: dict[str, object]) -> None:
    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    require_torch()
    seed_everything(args.seed)
    device = select_device()
    smoke = bool(args.smoke_test)
    image_size = 64 if smoke else TrainConfig.image_size
    batch_size = 2 if smoke else args.batch_size
    # Keep the smoke path bounded while giving the tiny fixture enough updates
    # to demonstrate that the model can memorize a handful of tiles; 12 epochs
    # only reached ~0.44 val macro-F1, so raise the cap until it actually overfits.
    epochs = min(args.epochs, 60) if smoke else args.epochs
    learning_rate = 1e-3 if smoke and args.learning_rate == 1e-4 else args.learning_rate

    if smoke:
        train_dataset = _SyntheticDataset(image_size)
        val_dataset = _SyntheticDataset(image_size)
        counts = np.bincount(np.concatenate([sample["mask"].numpy().reshape(-1) for sample in train_dataset.samples]), minlength=NUM_CLASSES)
        data_root = Path(".")
    else:
        if args.data is None:
            raise ValueError("--data is required for real training; use --smoke-test for the synthetic gate")
        data_root = args.data if args.data.is_dir() else args.data.parent
        records_path = args.data / "records.json" if args.data.is_dir() else args.data
        if not records_path.exists():
            raise FileNotFoundError(f"Prepared records.json not found at {records_path}")
        manifest_path = args.split_manifest or (data_root / "split_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError("A split_manifest.json is required for real training")
        records = load_records(records_path)
        train_ids = record_ids_for_split(manifest_path, "train")
        val_ids = record_ids_for_split(manifest_path, "val")
        train_records = [row for row in records if str(row["identifier"]) in train_ids]
        val_records = [row for row in records if str(row["identifier"]) in val_ids]
        train_dataset = PreparedTileDataset(train_records, data_root, training=True, image_size=image_size, seed=args.seed)
        val_dataset = PreparedTileDataset(val_records, data_root, training=False, image_size=image_size, seed=args.seed)
        counts = class_pixel_counts(train_dataset, num_classes=NUM_CLASSES)

    class_weights = inverse_sqrt_class_weights(counts).to(device)
    train_loader = _loader(train_dataset, batch_size, shuffle=True)
    val_loader = _loader(val_dataset, batch_size, shuffle=False)
    model_class = PostOnlyUNet if args.model == "post_only" else SiameseUNet
    model = model_class(pretrained=not args.no_pretrained and not smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_metric = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    config = {
        "model": args.model,
        "seed": args.seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "class_weights": [float(value) for value in class_weights.detach().cpu()],
        "data": str(data_root),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.model}_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    history_path = args.output / f"{args.model}_history.json"
    history: list[dict[str, object]] = []
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    stale_epochs = 0
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "epoch"):
            train_dataset.epoch = epoch
        train_loss, train_metrics = _run_epoch(model, train_loader, optimizer, class_weights, args.model, device, training=True, scaler=scaler)
        val_loss, val_metrics = _run_epoch(model, val_loader, optimizer, class_weights, args.model, device, training=False)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        metric = float(val_metrics["damage_macro_f1"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= TrainConfig.early_stopping_patience and not smoke:
                break
        _save_checkpoint(args.output / f"{args.model}_last.pt", model, optimizer, scheduler, epoch, best_metric, config)
        if improved:
            _save_checkpoint(args.output / f"{args.model}_best.pt", model, optimizer, scheduler, epoch, best_metric, config)
        # Written every epoch, not just at the end: a crash or kill mid-run
        # (a real risk on multi-hour real-data training) would otherwise lose
        # the entire loss/metric curve even though per-epoch checkpoints
        # survive, and there'd be no way to observe progress on a long run
        # in progress.
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {"model": args.model, "best_validation_damage_macro_f1": best_metric, "epochs_completed": len(history), "device": str(device), "output": str(args.output)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
