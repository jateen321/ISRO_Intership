"""Deterministic event-grouped train/validation/test split utilities."""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Mapping, Sequence


def _event_histogram(records: Sequence[Mapping[str, object]]) -> dict[str, Counter[int]]:
    histograms: dict[str, Counter[int]] = {}
    for record in records:
        event = str(record["event"])
        histogram = histograms.setdefault(event, Counter())
        for label in record.get("classes", ()):  # type: ignore[union-attr]
            histogram[int(label)] += 1
    return histograms


def build_event_split(
    records: Iterable[Mapping[str, object]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Assign event names to disjoint splits using a seeded greedy objective.

    The objective balances pair count and observed class support. It deliberately
    operates on event groups so a disaster cannot leak between splits.
    """

    rows = list(records)
    if not rows:
        raise ValueError("At least one record is required")
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain three values summing to 1")

    histograms = _event_histogram(rows)
    events = sorted(histograms)
    random.Random(seed).shuffle(events)
    events.sort(key=lambda event: sum(histograms[event].values()), reverse=True)
    total_pairs = len(rows)
    target_pairs = [total_pairs * ratio for ratio in ratios]
    total_classes = Counter(label for histogram in histograms.values() for label in histogram.elements())
    targets = [{label: count * ratio for label, count in total_classes.items()} for ratio in ratios]
    splits: list[list[str]] = [[], [], []]
    pair_counts = [0, 0, 0]
    class_counts = [Counter(), Counter(), Counter()]

    for event in events:
        event_size = sum(histograms[event].values())
        histogram = histograms[event]

        def objective(index: int) -> float:
            size_error = abs((pair_counts[index] + event_size) - target_pairs[index]) / max(total_pairs, 1)
            class_error = sum(
                abs(class_counts[index][label] + histogram[label] - targets[index][label])
                for label in total_classes
            ) / max(sum(total_classes.values()), 1)
            return size_error + class_error

        index = min(range(3), key=lambda candidate: (objective(candidate), len(splits[candidate])))
        splits[index].append(event)
        pair_counts[index] += event_size
        class_counts[index].update(histogram)

    result = {name: sorted(values) for name, values in zip(("train", "val", "test"), splits)}
    if not all(result.values()):
        raise ValueError("Could not assign at least one event to every split")
    return result


def assert_disjoint(split_manifest: Mapping[str, Sequence[str]]) -> None:
    seen: set[str] = set()
    for split, events in split_manifest.items():
        overlap = seen.intersection(events)
        if overlap:
            raise ValueError(f"Events appear in multiple splits ({split}): {sorted(overlap)}")
        seen.update(events)
