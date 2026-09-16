"""Heatmap decoding with sub-pixel offsets and flip test-time averaging."""
import numpy as np
import torch
import torch.nn.functional as F

from .train import H, PAD_TOP, W, pad_frames


def _forward_tta(model, x, tta):
    """Average heatmap probabilities and offsets over axis flips."""
    hm_sum, off_sum, n = None, None, 0
    combos = [(False, False)]
    if tta:
        combos = [(False, False), (True, False), (False, True), (True, True)]
    for fx, fy in combos:
        xi = x
        if fx:
            xi = xi.flip(-1)
        if fy:
            xi = xi.flip(-2)
        dev = x.device.type
        with torch.amp.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            hm, off = model(xi)
        hm = torch.sigmoid(hm.float())
        off = off.float()
        if fy:
            hm, off = hm.flip(-2), off.flip(-2)
            off[:, 1] = -off[:, 1]
        if fx:
            hm, off = hm.flip(-1), off.flip(-1)
            off[:, 0] = -off[:, 0]
        hm_sum = hm if hm_sum is None else hm_sum + hm
        off_sum = off if off_sum is None else off_sum + off
        n += 1
    return hm_sum / n, off_sum / n


@torch.no_grad()
def detect_stack(models, resid, neighbours, idx, device="cuda", threshold=0.30,
                 batch_size=8, tta=True, nms_kernel=5, max_per_frame=200):
    """Return a list of (n,2) normalised centres, one per index in `idx`.

    `models` may be a single module or a list whose heatmaps and offsets are averaged.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    resid_t = torch.from_numpy(resid)
    out = []
    for s in range(0, len(idx), batch_size):
        part = idx[s:s + batch_size]
        prev = neighbours[0][part]
        nxt = neighbours[1][part]
        x = torch.stack([resid_t[part], resid_t[prev], resid_t[nxt]], 1).float()
        x = pad_frames(x).to(device)
        hm, off = None, None
        for m in models:
            h, o = _forward_tta(m, x, tta)
            hm = h if hm is None else hm + h
            off = o if off is None else off + o
        hm, off = hm / len(models), off / len(models)
        keep = (hm == F.max_pool2d(hm, nms_kernel, 1, nms_kernel // 2)) & (hm >= threshold)
        keep[:, :, :PAD_TOP, :] = False
        keep[:, :, PAD_TOP + H:, :] = False
        b, _, ys, xs = torch.nonzero(keep, as_tuple=True)
        dx = off[b, 0, ys, xs]
        dy = off[b, 1, ys, xs]
        px = (xs.float() + dx).cpu().numpy()
        py = (ys.float() + dy - PAD_TOP).cpu().numpy()
        sc = hm[b, 0, ys, xs].cpu().numpy()
        bb = b.cpu().numpy()
        for k in range(len(part)):
            sel = bb == k
            pts = np.stack([np.clip(px[sel] / W, 0.0, 1.0),
                            np.clip(py[sel] / H, 0.0, 1.0)], 1)
            # Highest-scoring first, and capped: a frame never holds more than a few
            # dozen ants, so a larger count means a degenerate heatmap, and an
            # unbounded one would make the association cost matrix explode.
            order = np.argsort(-sc[sel])[:max_per_frame]
            out.append(pts[order].astype(np.float64))
    return out
