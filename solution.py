"""Ant Colony Gate Crossings - offline solution.

Usage:
    python3 solution.py <public_dir> <output_csv>

The run trains an ant-centre detector from randomly initialised weights on CUDA using
only the public training frames and their supplied centres, then applies the trained
detector to the test frames, associates detections into tracks, and reads gate
crossings off those tracks with the interpolation rule from the task statement.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acg.bank import build_frame_bank
from acg.data import build_sequences, load_tables
from acg.detect import detect_stack
from acg.track import predict_events
from acg.train import Trainer, to_pixel_centers

# Fixed run constants. Sized for one A10G-class GPU inside the 90 minute budget;
# see tools/benchmark.py for the measurement these were derived from.
STEPS = 9000
N_MODELS = 2
WIDTH = 32
DEPTH = 4
BATCH_SIZE = 8
LR = 2e-3

# Decode and association settings, swept with tools/sweep_decode.py on held-out
# recordings. Chosen to be near-optimal across detector quality rather than fitted to
# one checkpoint: a tighter 15 px gate scored 0.007 better with a deliberately
# undertrained model, but costs 0.02 at the perfect-centre ceiling, so it only suits a
# weak detector. Re-sweep after a full-length run.
THRESHOLD = 0.25
MAX_DIST = 25.0
MAX_GAP = 2
FILL_GAP = 1
MAX_EVENTS = 200


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def format_events(events):
    trimmed = events[:MAX_EVENTS]
    return json.dumps([[round(float(t), 6), int(d), round(float(p), 6)]
                       for t, d, p in trimmed], separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--models", type=int, default=N_MODELS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device required: this solution trains on GPU.")

    log(f"loading tables from {args.public_dir}")
    train, test, centers = load_tables(args.public_dir)
    log(f"{len(train)} training queries, {len(test)} test queries")

    train_chains = build_sequences(list(train["fr"]))
    test_chains = build_sequences(list(test["fr"]))
    log(f"training recordings {[len(c) for c in train_chains]}")
    log(f"test recordings {[len(c) for c in test_chains]}")

    all_chains = train_chains + test_chains
    n_train_chains = len(train_chains)
    bank = build_frame_bank(args.public_dir, all_chains, log=log)
    is_train = bank.chain_of < n_train_chains
    train_idx = np.nonzero(is_train)[0]
    test_idx = np.nonzero(~is_train)[0]
    log(f"frame bank: {len(train_idx)} training frames, {len(test_idx)} test frames")

    pixel_centers = to_pixel_centers([centers.get(f, np.zeros((0, 2)))
                                      for f in bank.order])

    models = []
    for k in range(args.models):
        log(f"training model {k + 1}/{args.models} for {args.steps} steps")
        trainer = Trainer(bank.resid, pixel_centers, device=args.device, width=args.width,
                          depth=DEPTH, batch_size=args.batch_size, lr=LR, seed=k)
        models.append(trainer.fit(train_idx, bank.neighbours, steps=args.steps, log=log))
        log(f"model {k + 1} done")

    log(f"detecting centres on {len(test_idx)} test frames")
    det = detect_stack(models, bank.resid, bank.neighbours, test_idx,
                       device=args.device, threshold=args.threshold, tta=True)
    pred_centers = {bank.order[i]: d for i, d in zip(test_idx, det)}
    log(f"mean detections per frame {np.mean([len(d) for d in det]):.2f}")

    log("associating tracks and extracting crossings")
    queries = [(r.fr, r.g) for r in test.itertuples()]
    preds = predict_events(queries, test_chains, pred_centers,
                           max_dist=MAX_DIST, max_gap=MAX_GAP, fill_gap=FILL_GAP)
    log(f"mean predicted events per query {np.mean([len(p) for p in preds]):.2f}")

    out = pd.DataFrame({
        "id": test["id"].astype(str),
        "crossing_events": [format_events(p) for p in preds],
    })
    out.to_csv(args.output_csv, index=False)
    log(f"wrote {args.output_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
