"""Frame bank: decode every frame once, remove the per-recording background."""
import time

import numpy as np

from .data import H, W, chain_background, load_chain_stack


class FrameBank:
    """Background-removed frames for a set of recordings, plus neighbour indices."""

    def __init__(self, order, index, resid, neighbours, chain_of):
        self.order = order
        self.index = index
        self.resid = resid
        self.neighbours = neighbours
        self.chain_of = chain_of

    def indices_for(self, chain_ids):
        want = set(chain_ids)
        return np.nonzero(np.isin(self.chain_of, list(want)))[0]


def build_frame_bank(public_dir, chains, log=print):
    order, chain_of = [], []
    for ci, chain in enumerate(chains):
        for f in chain:
            order.append(f)
            chain_of.append(ci)
    index = {f: i for i, f in enumerate(order)}
    resid = np.zeros((len(order), H, W), dtype=np.float16)
    for ci, chain in enumerate(chains):
        t0 = time.time()
        stack = load_chain_stack(public_dir, chain)
        bg = chain_background(stack)
        r = (bg[None] - stack).astype(np.float16)
        for p, f in enumerate(chain):
            resid[index[f]] = r[p]
        log(f"  recording {ci}: {len(chain)} frames in {time.time() - t0:.1f}s")
    prev = np.zeros(len(order), dtype=np.int64)
    nxt = np.zeros(len(order), dtype=np.int64)
    for chain in chains:
        ids = np.array([index[f] for f in chain])
        prev[ids] = ids[np.maximum(np.arange(len(ids)) - 1, 0)]
        nxt[ids] = ids[np.minimum(np.arange(len(ids)) + 1, len(ids) - 1)]
    return FrameBank(order, index, resid, (prev, nxt), np.array(chain_of))
