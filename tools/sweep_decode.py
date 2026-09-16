"""Tune the decode and association settings against a saved model, without retraining.

    python3 tools/sweep_decode.py --data data --model m.pt --holdout 4 5

Training dominates the budget, so every decode parameter should be swept against a
checkpoint rather than by repeating the run. Detection is computed once per threshold
and the association grid is then explored on CPU.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acg.bank import build_frame_bank
from acg.data import build_sequences, load_tables
from acg.detect import detect_stack
from acg.metric import mean_score
from acg.model import CenterNet
from acg.track import predict_events

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import detection_report

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_models(paths, device):
    models = []
    for p in paths:
        blob = torch.load(p, map_location=device, weights_only=False)
        net = CenterNet(3, blob.get("width", 32), blob.get("depth", 4)).to(device)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        models.append(net)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--threshold", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--max-dist", type=float, nargs="+", default=[20.0, 25.0, 30.0])
    ap.add_argument("--max-gap", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--fill-gap", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train, _, centers = load_tables(args.data)
    chains = build_sequences(list(train["fr"]))
    bank = build_frame_bank(args.data, chains)
    hold = set(args.holdout)
    ho_idx = np.nonzero(np.isin(bank.chain_of, list(hold)))[0]
    hold_frames = {bank.order[i] for i in ho_idx}
    rows = [r for r in train.itertuples() if r.fr[0] in hold_frames]
    ho_chains = [c for ci, c in enumerate(chains) if ci in hold]
    queries = [(r.fr, r.g) for r in rows]
    targets = [r.ev for r in rows]
    print(f"holdout recordings {sorted(hold)}  {len(ho_idx)} frames  {len(rows)} queries")

    models = load_models(args.model, args.device)
    print(f"loaded {len(models)} model(s)")

    results = []
    for thr in args.threshold:
        t0 = time.time()
        det = detect_stack(models, bank.resid, bank.neighbours, ho_idx,
                           device=args.device, threshold=thr, tta=not args.no_tta)
        pred_centers = {bank.order[i]: d for i, d in zip(ho_idx, det)}
        rep = detection_report(pred_centers, centers)
        per_frame = float(np.mean([len(d) for d in det]))
        print(f"\nthreshold {thr}: recall {rep['recall']:.4f} precision {rep['precision']:.4f} "
              f"loc mean {rep['loc_mean']:.3f} p90 {rep['loc_p90']:.3f} "
              f"{per_frame:.1f} det/frame  ({time.time() - t0:.0f}s)")
        best_here = None
        for md, mg, fg in itertools.product(args.max_dist, args.max_gap, args.fill_gap):
            if fg > mg - 1:
                continue        # a hole of fg frames needs max_gap >= fg + 1 to survive
            preds = predict_events(queries, ho_chains, pred_centers,
                                   max_dist=md, max_gap=mg, fill_gap=fg)
            s = mean_score(preds, targets)
            results.append({"threshold": thr, "max_dist": md, "max_gap": mg,
                            "fill_gap": fg, "score": s, "det_per_frame": per_frame, **rep})
            if best_here is None or s > best_here["score"]:
                best_here = results[-1]
        print(f"  best at this threshold: {best_here['score']:.4f} "
              f"(max_dist={best_here['max_dist']} max_gap={best_here['max_gap']} "
              f"fill_gap={best_here['fill_gap']})")

    best = max(results, key=lambda r: r["score"])
    print("\n" + "=" * 70)
    print(f"BEST {best['score']:.4f}  threshold={best['threshold']} "
          f"max_dist={best['max_dist']} max_gap={best['max_gap']} fill_gap={best['fill_gap']}")
    print(f"  recall {best['recall']:.4f}  precision {best['precision']:.4f}  "
          f"localisation mean {best['loc_mean']:.3f} px")
    print("Put these into the constants at the top of solution.py.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"best": best, "all": results}, fh, indent=2)


if __name__ == "__main__":
    main()
