"""Deterministic association across a recording and gate-crossing extraction."""
import collections

import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 480.0, 270.0


def track_chain(centers, max_dist=25.0, max_gap=1):
    """Link per-frame centre sets into tracks over one recording.

    centers: list (per chain frame) of (n,2) normalised centres.
    Returns list of (n,) int arrays with a track id per detection.
    """
    ids = [np.full(len(c), -1, dtype=np.int64) for c in centers]
    active = []  # (track_id, last_frame, last_xy_px)
    next_id = 0
    for t, c in enumerate(centers):
        pts = np.asarray(c, dtype=np.float64).reshape(-1, 2) * (W, H)
        alive = [a for a in active if t - a[1] <= max_gap]
        if len(alive) and len(pts):
            prev = np.stack([a[2] for a in alive])
            gaps = np.array([t - a[1] for a in alive], dtype=np.float64)
            dist = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=2)
            limit = max_dist * gaps[:, None]
            cost = np.where(dist <= limit, dist, 1e6)
            r, cc = linear_sum_assignment(cost)
            for i, j in zip(r, cc):
                if cost[i, j] < 1e6:
                    ids[t][j] = alive[i][0]
                    alive[i][1] = t
                    alive[i][2] = pts[j]
        for j in range(len(pts)):
            if ids[t][j] < 0:
                ids[t][j] = next_id
                active.append([next_id, t, pts[j]])
                next_id += 1
        active = [a for a in active if t - a[1] <= max_gap]
    return ids


def tracks_to_series(chain_ids, centers):
    """track id -> {chain frame index: (x, y) normalised}"""
    series = collections.defaultdict(dict)
    for t, (row, c) in enumerate(zip(chain_ids, centers)):
        arr = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        for j, tid in enumerate(row):
            series[int(tid)][t] = arr[j]
    return series


def gate_axis(gate):
    """0 -> vertical gate (crossings measured on x); 1 -> horizontal gate (on y)."""
    if abs(gate[0] - gate[2]) < 1e-9:
        return 0, float(gate[0])
    return 1, float(gate[1])


def window_events(series, start, length, axis, gval, fill_gap=0, spans=None):
    """Crossings for the 20-frame window beginning at chain index `start`."""
    out = []
    stop = start + length - 1
    for tid, pts in series.items():
        if spans is not None:
            lo, hi = spans[tid]
            if hi < start or lo > stop:
                continue
        local = {}
        for t, p in pts.items():
            i = t - start
            if 0 <= i < length:
                local[i] = p
        if len(local) < 2:
            continue
        if fill_gap > 0:
            local = _fill(local, fill_gap)
        for i in range(length - 1):
            if i not in local or (i + 1) not in local:
                continue
            p0, p1 = local[i], local[i + 1]
            d0, d1 = p0[axis] - gval, p1[axis] - gval
            if not ((d0 < 0 <= d1) or (d1 < 0 <= d0)):
                continue
            denom = d1 - d0
            alpha = 0.0 if denom == 0 else -d0 / denom
            alpha = min(1.0, max(0.0, alpha))
            other = 1 - axis
            pos = p0[other] + alpha * (p1[other] - p0[other])
            out.append([
                float(min(length - 1.0, max(0.0, i + alpha))),
                1 if d1 > d0 else -1,
                float(min(1.0, max(0.0, pos))),
            ])
    return out


def _fill(local, fill_gap):
    keys = sorted(local)
    filled = dict(local)
    for a, b in zip(keys, keys[1:]):
        gap = b - a
        if 1 < gap <= fill_gap + 1:
            for k in range(a + 1, b):
                w = (k - a) / gap
                filled[k] = local[a] * (1 - w) + local[b] * w
    return filled


def predict_events(queries, chains, centers_by_frame, max_dist=25.0, max_gap=1, fill_gap=0):
    """Run association once per recording, then read each query window off the tracks."""
    index = {}
    series_by_chain, spans_by_chain = [], []
    for ci, chain in enumerate(chains):
        cens = [centers_by_frame[f] for f in chain]
        ids = track_chain(cens, max_dist=max_dist, max_gap=max_gap)
        series = tracks_to_series(ids, cens)
        series_by_chain.append(series)
        spans_by_chain.append({tid: (min(p), max(p)) for tid, p in series.items()})
        for pos, f in enumerate(chain):
            index[f] = (ci, pos)
    preds = []
    for frames, gate in queries:
        ci, start = index[frames[0]]
        axis, gval = gate_axis(gate)
        preds.append(window_events(series_by_chain[ci], start, len(frames), axis, gval,
                                   fill_gap, spans_by_chain[ci]))
    return preds
