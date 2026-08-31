import json

import pytest

from geoshield.xbd import discover_records, prepare_dataset, rasterize_label


pytest.importorskip("PIL.Image")
pytest.importorskip("shapely")


def _make_fixture(root):
    from PIL import Image as PILImage

    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for suffix, color in (("pre", (30, 40, 50)), ("post", (90, 100, 110))):
        PILImage.new("RGB", (512, 512), color).save(root / "images" / f"alpha-01_{suffix}_disaster.png")
    label = {
        "features": {
            "xy": [
                {"properties": {"subtype": "no-damage"}, "wkt": "POLYGON ((20 20, 100 20, 100 100, 20 100, 20 20))"},
                {"properties": {"subtype": "destroyed"}, "wkt": "POLYGON ((150 150, 250 150, 250 250, 150 250, 150 150))"},
            ]
        }
    }
    (root / "labels" / "alpha-01_post_disaster.json").write_text(json.dumps(label), encoding="utf-8")


def test_discovery_and_rasterization(tmp_path):
    _make_fixture(tmp_path)
    records, missing = discover_records(tmp_path)
    assert missing == []
    assert len(records) == 1
    mask, stats = rasterize_label(tmp_path / "labels" / "alpha-01_post_disaster.json", (512, 512))
    assert mask.getpixel((30, 30)) == 1
    assert mask.getpixel((180, 180)) == 4
    assert stats["invalid_polygons"] == 0


def test_prepare_writes_one_tile_for_512_input(tmp_path):
    _make_fixture(tmp_path / "source")
    audit = prepare_dataset(tmp_path / "source", tmp_path / "prepared")
    assert audit["records_accepted"] == 1
    assert len(list((tmp_path / "prepared" / "tiles" / "mask").glob("*.png"))) == 1
    assert (tmp_path / "prepared" / "audit.json").exists()
    # The fixture's one label has a "no-damage" polygon and a "destroyed"
    # polygon; class_buildings must count both (regression for a dict
    # key-type mismatch: str(label) keys vs. int label lookups meant this
    # counter silently stayed all-zero regardless of input).
    assert audit["class_buildings"] == {"1": 1, "2": 0, "3": 0, "4": 1}
