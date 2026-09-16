"""Official row metric: direction-gated similarity under one-to-one matching."""
import numpy as np
from scipy.optimize import linear_sum_assignment

TIME_TOL = 2.0
POS_TOL = 0.08


def row_score(pred, target):
    if len(pred) == 0 or len(target) == 0:
        return 0.0
    p = np.asarray(pred, dtype=float).reshape(-1, 3)
    t = np.asarray(target, dtype=float).reshape(-1, 3)
    sim = (
        np.maximum(0.0, 1.0 - np.abs(p[:, 0:1] - t[None, :, 0]) / TIME_TOL)
        * np.maximum(0.0, 1.0 - np.abs(p[:, 2:3] - t[None, :, 2]) / POS_TOL)
        * (p[:, 1:2] == t[None, :, 1])
    )
    r, c = linear_sum_assignment(-sim)
    return float(2.0 * sim[r, c].sum() / (len(p) + len(t)))


def mean_score(preds, targets):
    return float(np.mean([row_score(p, t) for p, t in zip(preds, targets)]))
