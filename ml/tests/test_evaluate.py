import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geoshield.evaluate import build_parser, evaluate  # noqa: E402
from geoshield.models import PostOnlyUNet, SiameseUNet  # noqa: E402


def _make_prepared_fixture(root, identifiers=("alpha", "beta")):
    from PIL import Image

    tile_root = root / "tiles"
    for folder in ("before", "after", "mask"):
        (tile_root / folder).mkdir(parents=True, exist_ok=True)
    records = []
    for identifier in identifiers:
        Image.new("RGB", (8, 8), (10, 20, 30)).save(tile_root / "before" / f"{identifier}.png")
        Image.new("RGB", (8, 8), (30, 40, 50)).save(tile_root / "after" / f"{identifier}.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 2
        Image.fromarray(mask, mode="L").save(tile_root / "mask" / f"{identifier}.png")
        records.append(
            {
                "identifier": identifier,
                "event": "event",
                "tiles": [
                    {
                        "before": f"tiles/before/{identifier}.png",
                        "after": f"tiles/after/{identifier}.png",
                        "mask": f"tiles/mask/{identifier}.png",
                    }
                ],
            }
        )
    (root / "records.json").write_text(json.dumps(records), encoding="utf-8")
    manifest = {"records": {"train": ["alpha"], "val": [], "test": list(identifiers)}}
    (root / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_checkpoint(path, model_name, image_size=8):
    model_class = PostOnlyUNet if model_name == "post_only" else SiameseUNet
    model = model_class(pretrained=False)
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "config": {"model": model_name, "image_size": image_size, "seed": 42},
        },
        path,
    )


def test_evaluate_post_only_reports_confusion_matrix_metrics(tmp_path):
    data_dir = tmp_path / "prepared"
    _make_prepared_fixture(data_dir)
    checkpoint_path = tmp_path / "post_only_best.pt"
    _write_checkpoint(checkpoint_path, "post_only")

    args = build_parser().parse_args(
        [
            "--checkpoint", str(checkpoint_path),
            "--data", str(data_dir),
            "--output", str(tmp_path / "eval.json"),
        ]
    )
    result = evaluate(args)

    assert result["model"] == "post_only"
    assert result["split"] == "test"
    assert result["num_records"] == 2
    assert result["num_tiles"] == 2
    assert 0.0 <= result["damage_macro_f1"] <= 1.0
    assert len(result["per_class_f1"]) == 5
    assert len(result["confusion_matrix"]) == 5

    saved = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert saved == result


def test_evaluate_siamese_uses_both_before_and_after(tmp_path):
    data_dir = tmp_path / "prepared"
    _make_prepared_fixture(data_dir, identifiers=("gamma",))
    checkpoint_path = tmp_path / "siamese_best.pt"
    _write_checkpoint(checkpoint_path, "siamese")

    args = build_parser().parse_args(
        [
            "--checkpoint", str(checkpoint_path),
            "--data", str(data_dir),
            "--output", str(tmp_path / "eval.json"),
        ]
    )
    result = evaluate(args)
    assert result["model"] == "siamese"
    assert result["num_tiles"] == 1


def test_evaluate_rejects_missing_checkpoint(tmp_path):
    data_dir = tmp_path / "prepared"
    _make_prepared_fixture(data_dir)
    args = build_parser().parse_args(
        [
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--data", str(data_dir),
        ]
    )
    with pytest.raises(FileNotFoundError):
        evaluate(args)


def test_evaluate_rejects_empty_split(tmp_path):
    data_dir = tmp_path / "prepared"
    _make_prepared_fixture(data_dir)
    checkpoint_path = tmp_path / "post_only_best.pt"
    _write_checkpoint(checkpoint_path, "post_only")

    args = build_parser().parse_args(
        [
            "--checkpoint", str(checkpoint_path),
            "--data", str(data_dir),
            "--split", "val",
        ]
    )
    with pytest.raises(ValueError, match="No prepared records matched split"):
        evaluate(args)
