"""xBD discovery, annotation rasterization, and tile preparation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

try:
    from shapely import wkt
except ImportError:  # pragma: no cover - exercised only in an incomplete env
    wkt = None  # type: ignore[assignment]


DAMAGE_LABELS = {
    "no-damage": 1,
    "undamaged": 1,
    "minor-damage": 2,
    "minor": 2,
    "major-damage": 3,
    "major": 3,
    "destroyed": 4,
}


@dataclass(frozen=True)
class XBDRecord:
    event: str
    identifier: str
    before_image: str
    after_image: str
    post_label: str


def _label_path(root: Path, identifier: str) -> Path | None:
    for directory in (root / "labels", root / "label", root / "targets"):
        candidate = directory / f"{identifier}_post_disaster.json"
        if candidate.exists():
            return candidate
    return None


def discover_records(root: Path) -> tuple[list[XBDRecord], list[str]]:
    """Find complete pre/post/annotation records under an xBD split directory."""

    image_dir = root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected xBD images directory: {image_dir}")

    records: list[XBDRecord] = []
    missing: list[str] = []
    for before_path in sorted(image_dir.glob("*_pre_disaster.*")):
        identifier = before_path.name.rsplit("_pre_disaster", 1)[0]
        after_path = next(image_dir.glob(f"{identifier}_post_disaster.*"), None)
        label_path = _label_path(root, identifier)
        if after_path is None or label_path is None:
            missing.append(identifier)
            continue
        event = identifier.split("_", 1)[0]
        records.append(XBDRecord(event, identifier, str(before_path), str(after_path), str(label_path)))
    return records, missing


def _features(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    raw = payload.get("features", [])
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
    elif isinstance(raw, list):
        yield from (item for item in raw if isinstance(item, dict))


def read_label_features(path: Path) -> list[tuple[int, str | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parsed: list[tuple[int, str | None]] = []
    for feature in _features(payload):
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        raw_label = properties.get("subtype", feature.get("subtype"))
        normalized = str(raw_label).strip().lower().replace("_", "-").replace(" ", "-") if raw_label else ""
        label = DAMAGE_LABELS.get(normalized)
        parsed.append((label if label is not None else 255, feature.get("wkt")))
    return parsed


def _polygon_parts(geometry: Any) -> Iterable[list[tuple[float, float]]]:
    if geometry is None:
        return
    if geometry.geom_type == "Polygon":
        yield [(float(x), float(y)) for x, y in geometry.exterior.coords]
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield [(float(x), float(y)) for x, y in polygon.exterior.coords]


def rasterize_label(path: Path, size: tuple[int, int]) -> tuple[Image.Image, dict[str, int]]:
    """Rasterize xBD WKT polygons into a single-channel 0..4/255 mask."""

    if wkt is None:
        raise RuntimeError("Install Shapely before rasterizing xBD labels")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    stats = {"invalid_polygons": 0, "ignored_polygons": 0, "background_polygons": 0}
    for label, raw_wkt in read_label_features(path):
        if not raw_wkt:
            stats["invalid_polygons"] += 1
            continue
        try:
            geometry = wkt.loads(raw_wkt)
            parts = list(_polygon_parts(geometry))
            if not parts:
                stats["invalid_polygons"] += 1
                continue
        except Exception:
            stats["invalid_polygons"] += 1
            continue
        if label == 255:
            stats["ignored_polygons"] += 1
        elif label == 1:
            stats["background_polygons"] += 1
        for points in parts:
            draw.polygon(points, fill=label)
    return mask, stats


def _tile(image: Image.Image, left: int, top: int, tile_size: int) -> Image.Image:
    return image.crop((left, top, left + tile_size, top + tile_size))


def _write_visual_audit(entries: list[tuple[Image.Image, Image.Image]], destination: Path) -> None:
    if not entries:
        return
    thumb_size = 256
    sheet = Image.new("RGB", (thumb_size * len(entries), thumb_size), (12, 23, 22))
    for index, (image, mask) in enumerate(entries):
        image = image.convert("RGB").resize((thumb_size, thumb_size))
        overlay = Image.new("RGB", image.size, (238, 101, 113))
        color_mask = mask.resize(image.size).point(lambda value: 120 if value else 0)
        image = Image.composite(overlay, image, color_mask)
        sheet.paste(image, (index * thumb_size, 0))
    sheet.save(destination)


def prepare_dataset(root: Path, output: Path, tile_size: int = 512, limit: int | None = None) -> dict[str, Any]:
    records, missing = discover_records(root)
    if limit is not None:
        records = records[:limit]
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    tile_root = output / "tiles"
    before_root, after_root, mask_root = tile_root / "before", tile_root / "after", tile_root / "mask"
    for directory in (before_root, after_root, mask_root):
        directory.mkdir(parents=True, exist_ok=True)

    accepted = 0
    invalid: list[dict[str, str]] = []
    ignored = 0
    class_pixels = {str(label): 0 for label in range(5)}
    class_buildings = {str(label): 0 for label in range(1, 5)}
    prepared_records: list[dict[str, Any]] = []
    audit_images: list[tuple[Image.Image, Image.Image]] = []

    for record in records:
        try:
            before = Image.open(record.before_image).convert("RGB")
            after = Image.open(record.after_image).convert("RGB")
            if before.size != after.size or before.width != before.height or before.width % tile_size:
                raise ValueError(f"expected matching square dimensions divisible by {tile_size}")
            mask, mask_stats = rasterize_label(Path(record.post_label), before.size)
            ignored += mask_stats["ignored_polygons"]
            labels = read_label_features(Path(record.post_label))
            for label, _ in labels:
                if label in class_buildings:
                    class_buildings[str(label)] += 1
            mask_values = mask.tobytes()
            for value in range(5):
                class_pixels[str(value)] += mask_values.count(bytes([value]))
            if len(audit_images) < 4:
                audit_images.append((after, mask))
            tile_paths: list[dict[str, str]] = []
            tile_index = 0
            for top in range(0, before.height, tile_size):
                for left in range(0, before.width, tile_size):
                    suffix = f"{record.identifier}_{tile_index:02d}.png"
                    before_path = before_root / suffix
                    after_path = after_root / suffix
                    mask_path = mask_root / suffix
                    _tile(before, left, top, tile_size).save(before_path)
                    _tile(after, left, top, tile_size).save(after_path)
                    _tile(mask, left, top, tile_size).save(mask_path)
                    tile_paths.append({"before": str(before_path), "after": str(after_path), "mask": str(mask_path)})
                    tile_index += 1
            prepared_records.append({"event": record.event, "identifier": record.identifier, "tiles": tile_paths, "classes": [label for label, _ in labels]})
            accepted += 1
        except Exception as error:
            invalid.append({"identifier": record.identifier, "error": str(error)})

    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(json.dumps(prepared_records, indent=2) + "\n", encoding="utf-8")
    _write_visual_audit(audit_images, output / "visual_audit.png")
    audit = {
        "input": str(root),
        "tile_size": tile_size,
        "records_discovered": len(records),
        "records_accepted": accepted,
        "missing_pairs": sorted(missing),
        "invalid_records": invalid,
        "ignored_polygons": ignored,
        "class_pixels": class_pixels,
        "class_buildings": class_buildings,
        "status": "complete",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def record_to_dict(record: XBDRecord) -> dict[str, str]:
    return asdict(record)
