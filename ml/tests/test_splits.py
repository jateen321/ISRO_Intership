import json

from geoshield.splits import assert_class_support, assert_disjoint, build_event_split, write_split_manifest


def test_event_split_is_deterministic_and_disjoint():
    records = [
        {"event": event, "classes": [1, 2, 3, 4]}
        for event in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    ]
    first = build_event_split(records)
    second = build_event_split(records)
    assert first == second
    assert_disjoint(first)
    assert sorted(sum(first.values(), [])) == sorted({row["event"] for row in records})
    assert_class_support(records, first)


def test_split_manifest_binds_records_checksum(tmp_path):
    records = [
        {"event": event, "identifier": f"{event}-01", "classes": [1, 2, 3, 4]}
        for event in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    ]
    records_path = tmp_path / "records.json"
    manifest_path = tmp_path / "split_manifest.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    manifest = write_split_manifest(records_path, manifest_path)
    assert manifest["records_sha256"]
    assert set(manifest["events"]) == {"train", "val", "test"}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["version"] == 1
