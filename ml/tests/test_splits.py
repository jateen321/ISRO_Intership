from geoshield.splits import assert_disjoint, build_event_split


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
