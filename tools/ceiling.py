"""Reproduce the design evidence in the README. CPU only, about a minute.

    python3 tools/ceiling.py --data data

Feeds the supplied training centres through the association and geometry stages and
scores the result against the training targets. This isolates how much of the task is
detection: the ceiling is what a perfect detector would score, and the perturbation
tables show how each kind of detector error is punished.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acg.data import H, W, build_sequences, load_tables
from acg.metric import mean_score
from acg.track import predict_events

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def run(queries, chains, centers, targets, **kw):
    return mean_score(predict_events(queries, chains, centers, **kw), targets)


def smooth(centers, chains, window):
    """Centred moving average along each recording, matched by nearest neighbour."""
    from scipy.optimize import linear_sum_assignment
    out = {f: v.copy() for f, v in centers.items()}
    half = window // 2
    for chain in chains:
        arrs = [centers[f] for f in chain]
        for t, f in enumerate(chain):
            lo, hi = max(0, t - half), min(len(chain), t + half + 1)
            acc = [arrs[t]]
            for u in range(lo, hi):
                if u == t or len(arrs[u]) == 0 or len(arrs[t]) == 0:
                    continue
                d = np.linalg.norm(arrs[t][:, None] * (W, H) - arrs[u][None] * (W, H), axis=2)
                r, c = linear_sum_assignment(d)
                take = np.full_like(arrs[t], np.nan)
                for i, j in zip(r, c):
                    if d[i, j] < 25.0:
                        take[i] = arrs[u][j]
                acc.append(np.where(np.isnan(take), arrs[t], take))
            out[f] = np.mean(acc, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, _, centers = load_tables(args.data)
    chains = build_sequences(list(train["fr"]))
    print("training recordings:", [len(c) for c in chains])
    queries = [(r.fr, r.g) for r in train.itertuples()]
    targets = [r.ev for r in train.itertuples()]
    rng = np.random.default_rng(args.seed)

    base = run(queries, chains, centers, targets)
    print(f"\nCEILING with the supplied centres: {base:.4f}")
    print("  -> association and geometry are essentially exact; the task is detection\n")

    print("centre jitter (sigma px) -> score")
    for s in (0.5, 1.0, 2.0, 3.0):
        jit = {k: v + rng.normal(0, s, v.shape) / (W, H) for k, v in centers.items()}
        print(f"  {s:4.1f} px : {run(queries, chains, jit, targets):.4f}")

    print("\ndropped detections -> score")
    for p in (0.02, 0.05, 0.10, 0.20):
        drop = {k: v[rng.random(len(v)) > p] for k, v in centers.items()}
        print(f"  {p:4.0%} : {run(queries, chains, drop, targets):.4f}")

    print("\nfalse positives per frame -> score")
    for q in (0.5, 1.0, 2.0):
        fp = {}
        for k, v in centers.items():
            n = rng.poisson(q)
            fp[k] = np.vstack([v, rng.random((n, 2))]) if n else v
        print(f"  {q:4.1f} : {run(queries, chains, fp, targets):.4f}")

    print("\ntrajectory smoothing (exact centres) -> score")
    for win in (3, 5, 7):
        print(f"  window {win} : {run(queries, chains, smooth(centers, chains, win), targets):.4f}")
    print("  -> smoothing destroys real events: targets follow the raw annotated centres")

    print("\nassociation gate (px) -> score")
    for md in (10.0, 15.0, 20.0, 25.0, 30.0, 50.0):
        print(f"  {md:5.1f} : {run(queries, chains, centers, targets, max_dist=md):.4f}")


if __name__ == "__main__":
    main()
