"""Grouped validation: hold out whole recordings, train on the rest, score end to end.

The public split hides two complete recordings, so held-out recordings are the only
honest estimate of test behaviour. Random frame splits leak across overlapping windows.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acg.bank import build_frame_bank
from acg.data import H, W, build_sequences, load_tables
from acg.detect import detect_stack
from acg.metric import mean_score
from acg.track import predict_events
from acg.train import Trainer, to_pixel_centers

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def detection_report(pred_centers, centers, tol=3.0):
    errs, tp, fp, fn = [], 0, 0, 0
    for f, d in pred_centers.items():
        g = centers[f]
        if len(g) and len(d):
            dist = np.linalg.norm(g[:, None] * (W, H) - d[None] * (W, H), axis=2)
            r, c = linear_sum_assignment(dist)
            ok = dist[r, c] <= tol
            errs.append(dist[r, c][ok])
            tp += int(ok.sum())
            fp += len(d) - int(ok.sum())
            fn += len(g) - int(ok.sum())
        else:
            fp += len(d)
            fn += len(g)
    e = np.concatenate(errs) if errs else np.zeros(1)
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "loc_mean": float(e.mean()),
        "loc_p50": float(np.percentile(e, 50)),
        "loc_p90": float(np.percentile(e, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.30])
    ap.add_argument("--fill-gap", type=int, nargs="+", default=[0])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--save-model", default="")
    ap.add_argument("--save-detections", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train, test, centers = load_tables(args.data)
    chains = build_sequences(list(train["fr"]))
    print("recordings:", [len(c) for c in chains])
    bank = build_frame_bank(args.data, chains)

    hold = set(args.holdout)
    is_hold = np.isin(bank.chain_of, list(hold))
    print(f"train frames {int((~is_hold).sum())}  holdout frames {int(is_hold.sum())}")

    pixel_centers = to_pixel_centers([centers[f] for f in bank.order])
    tr_idx = np.nonzero(~is_hold)[0]
    t0 = time.time()
    trainer = Trainer(bank.resid, pixel_centers, width=args.width, depth=args.depth,
                      batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    model = trainer.fit(tr_idx, bank.neighbours, steps=args.steps)
    train_time = time.time() - t0
    print(f"train time {train_time:.0f}s")
    if args.save_model:
        import torch
        torch.save({"state_dict": model.state_dict(), "width": args.width,
                    "depth": args.depth}, args.save_model)
        print(f"saved model to {args.save_model}")

    ho_idx = np.nonzero(is_hold)[0]
    hold_frames = {bank.order[i] for i in ho_idx}
    rows = [r for r in train.itertuples() if r.fr[0] in hold_frames]
    ho_chains = [c for ci, c in enumerate(chains) if ci in hold]
    queries = [(r.fr, r.g) for r in rows]
    targ = [r.ev for r in rows]
    print(f"holdout queries: {len(rows)}")

    results = []
    for thr in args.threshold:
        t0 = time.time()
        det = detect_stack(model, bank.resid, bank.neighbours, ho_idx,
                           threshold=thr, tta=not args.no_tta)
        pred_centers = {bank.order[i]: d for i, d in zip(ho_idx, det)}
        rep = detection_report(pred_centers, centers)
        print(f"thr {thr}: recall {rep['recall']:.4f} precision {rep['precision']:.4f} "
              f"loc mean {rep['loc_mean']:.3f} p90 {rep['loc_p90']:.3f} "
              f"({time.time() - t0:.0f}s)")
        if args.save_detections:
            np.savez_compressed(
                args.save_detections.replace(".npz", f"_thr{thr}.npz"),
                frames=np.array(list(pred_centers.keys())),
                counts=np.array([len(v) for v in pred_centers.values()]),
                points=np.concatenate(list(pred_centers.values())) if pred_centers else
                np.zeros((0, 2)))
        for md in (20.0, 25.0, 30.0):
            for mg in (1, 2):
                for fg in args.fill_gap:
                    preds = predict_events(queries, ho_chains, pred_centers,
                                           max_dist=md, max_gap=mg, fill_gap=fg)
                    s = mean_score(preds, targ)
                    print(f"    max_dist {md} max_gap {mg} fill_gap {fg}: "
                          f"END-TO-END {s:.4f}")
                    results.append({"threshold": thr, "max_dist": md, "max_gap": mg,
                                    "fill_gap": fg, "score": s, **rep})
    best = max(results, key=lambda r: r["score"])
    print(f"BEST {best['score']:.4f} thr={best['threshold']} "
          f"max_dist={best['max_dist']} max_gap={best['max_gap']} "
          f"fill_gap={best['fill_gap']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"best": best, "all": results, "train_time": train_time,
                       "args": vars(args)}, fh, indent=2)


if __name__ == "__main__":
    main()
