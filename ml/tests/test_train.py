import json
import math

import pytest

torch = pytest.importorskip("torch")

from geoshield.train import _loader, build_parser, train  # noqa: E402


def _smoke_args(tmp_path, model="post_only", **overrides):
    argv = [
        "--model", model,
        "--smoke-test",
        "--output", str(tmp_path),
        "--seed", "42",
    ]
    for flag, value in overrides.items():
        argv += [flag, str(value)]
    return build_parser().parse_args(argv)


def test_loader_stays_serial_without_cuda(monkeypatch):
    class TinyDataset:
        def __len__(self):
            return 17

        def __getitem__(self, index):
            return {"before": torch.zeros(3, 2, 2), "after": torch.zeros(3, 2, 2), "mask": torch.zeros(2, 2, dtype=torch.long), "identifier": str(index)}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = _loader(TinyDataset(), 2, shuffle=False)
    assert loader.num_workers == 0


def test_smoke_test_overfits_eight_tiles(tmp_path):
    result = train(_smoke_args(tmp_path, **{"--epochs": 60}))
    assert result["best_validation_damage_macro_f1"] > 0.85

    history = json.loads((tmp_path / "post_only_history.json").read_text(encoding="utf-8"))
    assert len(history) == 60
    losses = [record["train_loss"] for record in history]
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    assert (tmp_path / "post_only_best.pt").exists()
    assert (tmp_path / "post_only_last.pt").exists()


def test_checkpoint_resume_continues_training(tmp_path):
    first = train(_smoke_args(tmp_path, **{"--epochs": 5}))
    assert first["epochs_completed"] == 5

    checkpoint = torch.load(tmp_path / "post_only_last.pt", map_location="cpu")
    assert checkpoint["epoch"] == 4
    # Early-stopping's patience counter must be checkpointed like best_metric
    # is, or a crash-and-resume silently resets how close a run was to
    # stopping (a real risk: multi-hour real-data runs get interrupted).
    assert "stale_epochs" in checkpoint and isinstance(checkpoint["stale_epochs"], int)

    resumed_args = _smoke_args(tmp_path, **{"--epochs": 8, "--resume": tmp_path / "post_only_last.pt"})
    second = train(resumed_args)

    # Resume must pick up after epoch 4 and preserve the prior curve, not restart
    # from zero or discard the epochs already recorded before the checkpoint.
    assert second["epochs_completed"] == 8

    history = json.loads((tmp_path / "post_only_history.json").read_text(encoding="utf-8"))
    assert [record["epoch"] for record in history] == list(range(8))
    assert all(math.isfinite(record["train_loss"]) for record in history)


def test_siamese_smoke_test_trains_through_fused_branch(tmp_path):
    # Proves the shared-encoder + per-scale concat-fusion forward/backward
    # path is wired correctly, independent of the real xBD dataset.
    result = train(_smoke_args(tmp_path, model="siamese", **{"--epochs": 60}))
    assert result["best_validation_damage_macro_f1"] > 0.85

    history = json.loads((tmp_path / "siamese_history.json").read_text(encoding="utf-8"))
    assert all(math.isfinite(record["train_loss"]) for record in history)
