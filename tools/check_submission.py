"""Validate a submission CSV against every format rule in the task statement.

    python3 tools/check_submission.py data submission.csv

Checks the rules that cause a zero rather than a low score: column names and order,
exact test-ID coverage, JSON well-formedness, triple shape, finite values, time in
[0,19], position in [0,1], direction exactly +1/-1 as an integer, the 200-event and
50,000-character cell caps, and the 64 MiB file cap.
"""
import argparse
import json
import math
import os
import sys

import pandas as pd

MAX_EVENTS = 200
MAX_CELL = 50_000
MAX_BYTES = 64 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("submission")
    args = ap.parse_args()

    problems = []
    size = os.path.getsize(args.submission)
    if size > MAX_BYTES:
        problems.append(f"file is {size} bytes, over the {MAX_BYTES} byte cap")

    sub = pd.read_csv(args.submission, dtype={"id": str})
    if list(sub.columns) != ["id", "crossing_events"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['id', 'crossing_events']")

    test = pd.read_csv(os.path.join(args.public_dir, "test.csv"), dtype={"id": str})
    want, got = set(test["id"]), set(sub["id"])
    if len(sub) != len(sub["id"].unique()):
        problems.append("duplicate ids present")
    if want - got:
        problems.append(f"{len(want - got)} test ids missing, e.g. {sorted(want - got)[:3]}")
    if got - want:
        problems.append(f"{len(got - want)} unknown ids present, e.g. {sorted(got - want)[:3]}")

    n_events, n_empty, bad_rows = 0, 0, 0
    for row in sub.itertuples():
        cell = row.crossing_events
        if not isinstance(cell, str):
            bad_rows += 1
            problems.append(f"{row.id}: cell is not a string")
            continue
        if len(cell) > MAX_CELL:
            problems.append(f"{row.id}: cell is {len(cell)} chars, over {MAX_CELL}")
        try:
            events = json.loads(cell)
        except Exception as exc:
            bad_rows += 1
            problems.append(f"{row.id}: JSON parse failed ({exc})")
            continue
        if not isinstance(events, list):
            bad_rows += 1
            problems.append(f"{row.id}: top level is not a list")
            continue
        if len(events) > MAX_EVENTS:
            problems.append(f"{row.id}: {len(events)} events, over {MAX_EVENTS}")
        if not events:
            n_empty += 1
        for e in events:
            n_events += 1
            if not isinstance(e, list) or len(e) != 3:
                problems.append(f"{row.id}: event {e} is not a 3-element list")
                continue
            t, d, p = e
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
                problems.append(f"{row.id}: time {t!r} is not a finite number")
            elif not 0 <= t <= 19:
                problems.append(f"{row.id}: time {t} outside [0,19]")
            if isinstance(d, bool) or not isinstance(d, int) or d not in (1, -1):
                problems.append(f"{row.id}: direction {d!r} is not integer 1 or -1")
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
                problems.append(f"{row.id}: position {p!r} is not a finite number")
            elif not 0 <= p <= 1:
                problems.append(f"{row.id}: position {p} outside [0,1]")

    print(f"rows {len(sub)}  events {n_events}  "
          f"mean {n_events / max(len(sub), 1):.2f}/row  empty rows {n_empty}")
    print(f"file size {size / 1024:.0f} KiB")
    if n_empty:
        print(f"WARNING: {n_empty} empty rows score exactly zero (every target list is nonempty)")
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems[:25]:
            print("  -", p)
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1
    print("\nPASS: submission is well formed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
