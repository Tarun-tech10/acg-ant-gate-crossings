"""Measure training throughput on the current GPU and size the step budget.

Run this once on the target machine before the graded run:

    python3 tools/benchmark.py --data data

It prints the measured seconds per step and the STEPS value that fills the training
share of the 90 minute budget. Put that number into the STEPS constant in solution.py
so the graded run stays a fixed, deterministic configuration with no wall-clock branch.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acg.model import CenterNet
from acg.train import PAD_H, W, focal_loss

BUDGET_MIN = 90.0
# Everything that is not the training loop: PNG decode, backgrounds, detection with
# flip averaging, association, CSV writing, plus headroom for a slower disk.
OVERHEAD_MIN = 12.0


def bench_steps(width, depth, batch_size, device, iters=15):
    torch.backends.cudnn.benchmark = True
    model = CenterNet(3, width, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device)
    x = torch.randn(batch_size, 3, PAD_H, W, device=device)
    hm = torch.zeros(batch_size, 1, PAD_H, W, device=device)
    off = torch.randn(batch_size, 2, PAD_H, W, device=device)
    msk = (torch.rand(batch_size, 1, PAD_H, W, device=device) > 0.995).float()

    def step():
        with torch.amp.autocast(device, dtype=torch.float16):
            phm, poff = model(x)
            loss = focal_loss(phm.float(), hm) + 2.0 * (
                torch.abs(poff.float() - off) * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--models", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device found")
    print("device:", torch.cuda.get_device_name(0))
    budget_s = (BUDGET_MIN - OVERHEAD_MIN) * 60.0
    print(f"training budget {budget_s / 60:.0f} min across {args.models} model(s)\n")
    for bs in args.batch_size:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sec, peak = bench_steps(args.width, args.depth, bs, args.device)
        steps = int(budget_s / args.models / sec / 100) * 100
        print(f"batch {bs:3d}: {sec * 1000:7.1f} ms/step  peak {peak:5.2f} GB  "
              f"{bs / sec:6.1f} img/s  ->  STEPS = {steps}")


if __name__ == "__main__":
    main()
