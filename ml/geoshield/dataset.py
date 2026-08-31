"""Prepared xBD tile datasets and deterministic training utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - keeps preparation tools importable
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Install the ML dependencies before loading training data")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def _normalise_image(path: Path, image_size: int | None = None):
    _require_torch()
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image_size is not None and image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _load_mask(path: Path, image_size: int | None = None):
    _require_torch()
    with Image.open(path) as source:
        mask = source.convert("L")
        if image_size is not None and mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        array = np.asarray(mask, dtype=np.int64)
    valid_values = set(np.unique(array).tolist())
    if not valid_values.issubset({0, 1, 2, 3, 4, 255}):
        raise ValueError(f"Mask {path} contains unsupported labels: {sorted(valid_values)}")
    return torch.from_numpy(array).long()


def _augment_pair(before, after, mask, rng: random.Random):
    """Apply one geometric transform to both images and the mask."""

    if rng.random() < 0.5:
        before = torch.flip(before, dims=(-1,))
        after = torch.flip(after, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if rng.random() < 0.5:
        before = torch.flip(before, dims=(-2,))
        after = torch.flip(after, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    quarter_turns = rng.randrange(4)
    if quarter_turns:
        before = torch.rot90(before, quarter_turns, dims=(-2, -1))
        after = torch.rot90(after, quarter_turns, dims=(-2, -1))
        mask = torch.rot90(mask, quarter_turns, dims=(-2, -1))
    return before, after, mask


def load_records(records_path: Path) -> list[dict[str, object]]:
    """Load and validate the prepared records.json list."""

    rows = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Prepared records must be a JSON list")
    for row in rows:
        if not isinstance(row, dict) or not row.get("identifier") or not isinstance(row.get("tiles"), list):
            raise ValueError("Each prepared record requires identifier and tiles fields")
    return rows


def record_ids_for_split(manifest_path: Path, split: str) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        values = manifest["records"][split]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Split manifest has no records for {split!r}") from error
    if not isinstance(values, list):
        raise ValueError(f"Split {split!r} must be a list of record identifiers")
    return {str(value) for value in values}


class PreparedTileDataset(Dataset):
    """Flatten prepared record tiles into paired samples."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        root: Path,
        *,
        training: bool = False,
        image_size: int | None = 512,
        seed: int = 42,
    ):
        _require_torch()
        self.root = root
        self.training = training
        self.image_size = image_size
        self.seed = seed
        self.epoch = 0
        self.samples: list[tuple[str, str, str, str]] = []
        for record in records:
            identifier = str(record["identifier"])
            tiles = record["tiles"]
            if not isinstance(tiles, list):
                raise ValueError(f"Record {identifier} has no tile list")
            for tile in tiles:
                if not isinstance(tile, Mapping) or not all(key in tile for key in ("before", "after", "mask")):
                    raise ValueError(f"Record {identifier} contains an invalid tile")
                self.samples.append(
                    (
                        identifier,
                        str(tile["before"]),
                        str(tile["after"]),
                        str(tile["mask"]),
                    )
                )
        if not self.samples:
            raise ValueError("No prepared tiles were found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        identifier, before_path, after_path, mask_path = self.samples[index]
        before = _normalise_image(_resolve(self.root, before_path), self.image_size)
        after = _normalise_image(_resolve(self.root, after_path), self.image_size)
        mask = _load_mask(_resolve(self.root, mask_path), self.image_size)
        if self.training:
            # Vary the transform per epoch (not just per sample) so training
            # augmentation is not a single fixed transform reapplied every epoch.
            sample_seed = self.seed + index * 100_003 + self.epoch
            before, after, mask = _augment_pair(before, after, mask, random.Random(sample_seed))
        return {"before": before, "after": after, "mask": mask, "identifier": identifier}


def class_pixel_counts(dataset: PreparedTileDataset, *, num_classes: int = 5) -> np.ndarray:
    """Count valid mask pixels without loading image tensors into memory."""

    counts = np.zeros(num_classes, dtype=np.int64)
    for _, _, _, mask_path in dataset.samples:
        with Image.open(_resolve(dataset.root, mask_path)) as source:
            values = np.asarray(source.convert("L"), dtype=np.int64)
        values = values[(values >= 0) & (values < num_classes)]
        counts += np.bincount(values, minlength=num_classes)[:num_classes]
    return counts


def inverse_sqrt_class_weights(counts: Iterable[int], *, cap: float = 5.0):
    """Return inverse-square-root frequency weights normalized around one."""

    _require_torch()
    values = np.asarray(list(counts), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional sequence")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("at least one class must have positive pixel support")
    positive = values > 0
    weights = np.zeros_like(values)
    weights[positive] = 1.0 / np.sqrt(values[positive])
    weights[positive] /= weights[positive].mean()
    weights[positive] = np.minimum(weights[positive], cap)
    return torch.tensor(weights, dtype=torch.float32)
