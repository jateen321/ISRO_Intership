import json

import numpy as np
import pytest

from geoshield.dataset import PreparedTileDataset, class_pixel_counts, inverse_sqrt_class_weights


torch = pytest.importorskip("torch")


def test_prepared_tiles_load_relative_paths_and_synchronized_shapes(tmp_path):
    from PIL import Image

    tile_root = tmp_path / "tiles"
    for folder in ("before", "after", "mask"):
        (tile_root / folder).mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tile_root / "before" / "sample.png")
    Image.new("RGB", (8, 8), (30, 40, 50)).save(tile_root / "after" / "sample.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 4
    Image.fromarray(mask, mode="L").save(tile_root / "mask" / "sample.png")
    records = [
        {
            "identifier": "sample",
            "event": "event",
            "tiles": [{"before": "tiles/before/sample.png", "after": "tiles/after/sample.png", "mask": "tiles/mask/sample.png"}],
        }
    ]
    dataset = PreparedTileDataset(records, tmp_path, training=True, image_size=8, seed=42)
    item = dataset[0]
    assert tuple(item["before"].shape) == (3, 8, 8)
    assert tuple(item["after"].shape) == (3, 8, 8)
    assert tuple(item["mask"].shape) == (8, 8)
    assert set(item["mask"].unique().tolist()) == {0, 4}
    assert class_pixel_counts(dataset).tolist() == [48, 0, 0, 0, 16]


def test_inverse_sqrt_weights_are_finite_and_capped():
    weights = inverse_sqrt_class_weights([1000, 100, 10, 1, 1])
    assert torch.isfinite(weights).all()
    assert float(weights.max()) <= 5.0
    assert weights.shape == (5,)


def test_augmentation_varies_across_epochs_and_is_reproducible(tmp_path):
    from PIL import Image

    tile_root = tmp_path / "tiles"
    for folder in ("before", "after", "mask"):
        (tile_root / folder).mkdir(parents=True)
    # An asymmetric gradient and an off-center mask block: unlike a solid-color
    # image with a centered mask, flips/rotations of this fixture are distinguishable.
    gradient = np.zeros((8, 8, 3), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            gradient[y, x] = (x * 30, y * 30, 5)
    Image.fromarray(gradient, mode="RGB").save(tile_root / "before" / "sample.png")
    Image.fromarray(gradient, mode="RGB").save(tile_root / "after" / "sample.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 5:7] = 4
    Image.fromarray(mask, mode="L").save(tile_root / "mask" / "sample.png")
    records = [
        {
            "identifier": "sample",
            "tiles": [{"before": "tiles/before/sample.png", "after": "tiles/after/sample.png", "mask": "tiles/mask/sample.png"}],
        }
    ]
    dataset = PreparedTileDataset(records, tmp_path, training=True, image_size=8, seed=42)

    seen = set()
    for epoch in range(8):
        dataset.epoch = epoch
        seen.add(dataset[0]["mask"].numpy().tobytes())
    assert len(seen) > 1, "augmentation must vary across epochs, not repeat one fixed transform"

    dataset.epoch = 3
    first = dataset[0]["mask"].numpy().copy()
    dataset.epoch = 3
    second = dataset[0]["mask"].numpy().copy()
    assert np.array_equal(first, second), "the same epoch must reproduce the same transform"

