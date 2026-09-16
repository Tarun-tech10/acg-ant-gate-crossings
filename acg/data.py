"""Dataset loading, sequence reconstruction and per-sequence background estimation."""
import collections
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

W, H = 480, 270


def load_tables(public_dir):
    train = pd.read_csv(os.path.join(public_dir, "train.csv"))
    test = pd.read_csv(os.path.join(public_dir, "test.csv"))
    frame_tr = pd.read_csv(os.path.join(public_dir, "frame_training.csv"))
    for df in (train, test):
        df["fr"] = df["frames"].map(json.loads)
        df["g"] = df["gate"].map(json.loads)
    if "crossing_events" in train.columns:
        train["ev"] = train["crossing_events"].map(json.loads)
    centers = {
        r.image: np.asarray(json.loads(r.centers), dtype=np.float64).reshape(-1, 2)
        for r in frame_tr.itertuples()
    }
    return train, test, centers


def build_sequences(frame_lists):
    """Stitch overlapping 20-frame windows into full recording chains.

    Displayed frames inside a query are consecutive samples of one recording, so
    the successor relation is a function and each recording forms a simple path.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    succ, pred = {}, {}
    for fs in frame_lists:
        for i in range(len(fs) - 1):
            ra, rb = find(fs[i]), find(fs[i + 1])
            if ra != rb:
                parent[ra] = rb
            succ[fs[i]] = fs[i + 1]
            pred[fs[i + 1]] = fs[i]
    comps = collections.defaultdict(list)
    for f in parent:
        comps[find(f)].append(f)
    chains = []
    for members in comps.values():
        heads = [f for f in members if f not in pred]
        head = heads[0] if heads else sorted(members)[0]
        chain, seen = [head], {head}
        while chain[-1] in succ and succ[chain[-1]] not in seen:
            chain.append(succ[chain[-1]])
            seen.add(chain[-1])
        chains.append(chain)
    chains.sort(key=lambda c: (-len(c), c[0]))
    return chains


def read_gray(public_dir, rel):
    return np.asarray(Image.open(os.path.join(public_dir, rel)), dtype=np.float32) / 255.0


def load_chain_stack(public_dir, chain):
    return np.stack([read_gray(public_dir, f) for f in chain], 0)


def chain_background(stack, max_frames=120):
    """Median over the recording removes the static arena and leaves the ants out."""
    if len(stack) > max_frames:
        idx = np.linspace(0, len(stack) - 1, max_frames).astype(int)
        stack = stack[idx]
    return np.median(stack, axis=0)


def make_inputs(stack, bg):
    """Per-frame 3-channel tensor: background-removed frame plus its two neighbours."""
    n = len(stack)
    resid = (bg[None] - stack).astype(np.float32)
    prev = resid[np.maximum(np.arange(n) - 1, 0)]
    nxt = resid[np.minimum(np.arange(n) + 1, n - 1)]
    return np.stack([resid, prev, nxt], 1)
