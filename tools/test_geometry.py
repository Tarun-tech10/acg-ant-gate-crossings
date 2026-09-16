"""Assertions for the two pieces that must match the task statement exactly.

    python3 tools/test_geometry.py --data data

The crossing rule and the metric are the only code that is specified rather than
learned, so a silent deviation would be invisible in training loss and fatal to the
score. These check them against the statement, then against the real targets.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acg.data import build_sequences, load_tables
from acg.metric import row_score
from acg.track import gate_axis, predict_events, window_events

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def series(points):
    """One track: {frame index: (x, y)}."""
    return {0: {i: np.array(p, dtype=float) for i, p in points.items()}}


@check("vertical gate, left to right, midway between frames")
def _():
    # gate at x=0.5; x goes 0.4 -> 0.6 between frames 3 and 4, y 0.2 -> 0.4
    ev = window_events(series({3: (0.4, 0.2), 4: (0.6, 0.4)}), 0, 20, 0, 0.5)
    assert len(ev) == 1, ev
    t, d, p = ev[0]
    assert abs(t - 3.5) < 1e-9, t          # alpha = -(-0.1)/0.2 = 0.5
    assert d == 1, d                       # increasing x
    assert abs(p - 0.3) < 1e-9, p          # y interpolated with the same alpha


@check("horizontal gate, decreasing y gives direction -1")
def _():
    ev = window_events(series({0: (0.25, 0.8), 1: (0.75, 0.2)}), 0, 20, 1, 0.5)
    assert len(ev) == 1, ev
    t, d, p = ev[0]
    assert d == -1, d
    assert abs(t - (0 + 0.5)) < 1e-9, t    # alpha = -0.3 / -0.6 = 0.5
    assert abs(p - 0.5) < 1e-9, p          # x interpolated


@check("landing exactly on the gate counts, leaving it does not")
def _():
    # d0 < 0 <= d1 : arriving at the gate from the left is a crossing
    assert len(window_events(series({0: (0.4, 0.5), 1: (0.5, 0.5)}), 0, 20, 0, 0.5)) == 1
    # d0 = 0, d1 > 0 : starting on the gate is not, since d0 < 0 fails both ways
    assert len(window_events(series({0: (0.5, 0.5), 1: (0.6, 0.5)}), 0, 20, 0, 0.5)) == 0


@check("no crossing when both endpoints are on the same side")
def _():
    assert window_events(series({0: (0.1, 0.5), 1: (0.4, 0.5)}), 0, 20, 0, 0.5) == []
    assert window_events(series({0: (0.6, 0.5), 1: (0.9, 0.5)}), 0, 20, 0, 0.5) == []


@check("a track absent at one endpoint yields no event")
def _():
    # present at 0 and 2 only: the rule needs both ends of a consecutive pair
    assert window_events(series({0: (0.4, 0.5), 2: (0.6, 0.5)}), 0, 20, 0, 0.5) == []
    # fill_gap=1 interpolates the hole and recovers it
    ev = window_events(series({0: (0.4, 0.5), 2: (0.6, 0.5)}), 0, 20, 0, 0.5, fill_gap=1)
    assert len(ev) == 1, ev


@check("back and forth across the gate yields two opposite events")
def _():
    ev = window_events(series({0: (0.45, 0.5), 1: (0.55, 0.5), 2: (0.45, 0.5)}),
                       0, 20, 0, 0.5)
    assert len(ev) == 2, ev
    assert sorted(e[1] for e in ev) == [-1, 1], ev


@check("gate_axis reads both gate orientations")
def _():
    assert gate_axis([0.65, 0, 0.65, 1]) == (0, 0.65)     # vertical -> compare x
    assert gate_axis([0, 0.35, 1, 0.35]) == (1, 0.35)     # horizontal -> compare y


@check("metric: identical lists score 1, direction mismatch scores 0")
def _():
    a = [[4.25, 1, 0.6], [10.5, -1, 0.2]]
    assert abs(row_score(a, a) - 1.0) < 1e-12
    flipped = [[4.25, -1, 0.6], [10.5, 1, 0.2]]
    assert row_score(a, flipped) == 0.0


@check("metric: tolerances fall to zero at 2 frames and 0.08")
def _():
    t = [[10.0, 1, 0.5]]
    assert abs(row_score([[11.0, 1, 0.5]], t) - 0.5) < 1e-12       # half the time budget
    assert abs(row_score([[10.0, 1, 0.54]], t) - 0.5) < 1e-12      # half the position budget
    # At the boundary itself the similarity is zero up to float rounding: 0.58-0.5
    # is not exactly 0.08 in binary, so demanding == 0.0 would test the float
    # representation rather than the metric.
    assert row_score([[12.0, 1, 0.5]], t) < 1e-9
    assert row_score([[10.0, 1, 0.58]], t) < 1e-9
    assert row_score([[12.5, 1, 0.5]], t) == 0.0                   # past it, exactly zero
    assert row_score([[10.0, 1, 0.62]], t) == 0.0


@check("metric: a duplicate prediction is penalised, not rewarded twice")
def _():
    t = [[10.0, 1, 0.5]]
    assert abs(row_score([[10.0, 1, 0.5]], t) - 1.0) < 1e-12
    # two perfect copies: 2*1 / (2 + 1)
    assert abs(row_score([[10.0, 1, 0.5]] * 2, t) - 2.0 / 3.0) < 1e-12


@check("metric: an empty prediction scores zero")
def _():
    assert row_score([], [[10.0, 1, 0.5]]) == 0.0


def data_checks(data_dir):
    train, _, centers = load_tables(data_dir)
    chains = build_sequences(list(train["fr"]))

    with open(os.path.join(data_dir, "schema_example.json")) as fh:
        example = json.load(fh)
    row = train[train["id"] == example["id"]]
    assert len(row) == 1, "schema_example id is not in train.csv"
    assert abs(row_score(example["crossing_events"], row.iloc[0]["ev"]) - 1.0) < 1e-9
    print("  schema_example.json scores exactly 1.0 against its train.csv row")

    sample = train.head(300)
    queries = [(r.fr, r.g) for r in sample.itertuples()]
    targets = [r.ev for r in sample.itertuples()]
    preds = predict_events(queries, chains, centers)
    scores = [row_score(p, t) for p, t in zip(preds, targets)]
    mean = float(np.mean(scores))
    assert mean > 0.99, f"geometry reproduces targets at only {mean:.4f}"
    print(f"  supplied centres reproduce {len(sample)} training rows at {mean:.4f}")

    every = np.concatenate([np.asarray(t, dtype=float).reshape(-1, 3) for t in targets])
    assert every[:, 0].min() >= 0 and every[:, 0].max() <= 19
    assert set(np.unique(every[:, 1]).astype(int)) <= {-1, 1}
    assert every[:, 2].min() >= 0 and every[:, 2].max() <= 1
    print("  training targets obey the stated ranges")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print()
    if os.path.isdir(args.data):
        data_checks(args.data)
    else:
        print(f"  (skipped data checks, {args.data} not found)")
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print(f"all {len(CHECKS)} geometry and metric checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
