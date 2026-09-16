"""Supervised centre detection trained from randomly initialised weights on CUDA."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import CenterNet

W, H = 480, 270
PAD_H = 272          # 270 padded up to a multiple of 16
PAD_TOP = 1
SIGMA = 1.6          # heatmap spread in pixels
OFF_RADIUS = 2       # offset supervision radius in pixels


def pad_frames(x):
    """(n,c,270,480) -> (n,c,272,480) by edge replication."""
    return F.pad(x, (0, 0, PAD_TOP, PAD_H - H - PAD_TOP), mode="replicate")


def to_pixel_centers(centers_list):
    """Normalised centres -> padded-image pixel coordinates."""
    out = []
    for cs in centers_list:
        a = np.asarray(cs, dtype=np.float64).reshape(-1, 2) * (W, H)
        a[:, 1] += PAD_TOP
        out.append(a.astype(np.float32))
    return out


def make_targets(pixel_centers, sigma=SIGMA, radius=OFF_RADIUS):
    """Gaussian centre heatmaps plus dense sub-pixel offsets around each centre.

    `pixel_centers` is a sequence of (n,2) arrays already in padded pixel space.
    Cheap enough to run per batch, which keeps the whole target tensor out of RAM.
    """
    n = len(pixel_centers)
    hm = np.zeros((n, 1, PAD_H, W), dtype=np.float32)
    off = np.zeros((n, 2, PAD_H, W), dtype=np.float32)
    msk = np.zeros((n, 1, PAD_H, W), dtype=np.float32)
    rad = int(np.ceil(3 * sigma))
    for i, cs in enumerate(pixel_centers):
        for cx, cyp in cs:
            x0, x1 = max(0, int(cx) - rad), min(W, int(cx) + rad + 2)
            y0, y1 = max(0, int(cyp) - rad), min(PAD_H, int(cyp) + rad + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = cx - xx, cyp - yy
            g = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            np.maximum(hm[i, 0, y0:y1, x0:x1], g, out=hm[i, 0, y0:y1, x0:x1])
            near = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
            sel = near & (g >= hm[i, 0, y0:y1, x0:x1] - 1e-6)
            off[i, 0, y0:y1, x0:x1][sel] = dx[sel]
            off[i, 1, y0:y1, x0:x1][sel] = dy[sel]
            msk[i, 0, y0:y1, x0:x1][sel] = 1.0
    return hm, off, msk


def focal_loss(logit, gt, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(1.0 - 1e-6).float()
    neg = 1.0 - pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - gt) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def augment(x, hm, off, msk, gen):
    """Axis flips keep centres exact; gain and noise mimic the capture nuisances."""
    if torch.rand(1, generator=gen, device=x.device).item() < 0.5:
        x, hm, msk = x.flip(-1), hm.flip(-1), msk.flip(-1)
        off = off.flip(-1)
        off[:, 0] = -off[:, 0]
    if torch.rand(1, generator=gen, device=x.device).item() < 0.5:
        x, hm, msk = x.flip(-2), hm.flip(-2), msk.flip(-2)
        off = off.flip(-2)
        off[:, 1] = -off[:, 1]
    b = x.shape[0]
    gain = 1.0 + 0.25 * (torch.rand(b, 1, 1, 1, generator=gen, device=x.device) - 0.5)
    x = x * gain
    x = x + 0.02 * torch.randn(x.shape, generator=gen, device=x.device)
    return x, hm, off, msk


class Trainer:
    """Trains one detector. Targets are rasterised per batch, never stored whole."""

    def __init__(self, resid, pixel_centers, device="cuda", width=32, depth=4, seed=0,
                 lr=2e-3, batch_size=8, channels=3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.device = device
        self.resid = torch.from_numpy(resid)
        self.pixel_centers = pixel_centers
        self.model = CenterNet(cin=channels, width=width, depth=depth).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.cuda = device == "cuda"
        self.scaler = torch.amp.GradScaler(device, enabled=self.cuda)
        self.batch_size = batch_size
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 1)
        self.lr = lr
        self.seed = seed

    def _inputs(self, idx, neighbours):
        prev = torch.from_numpy(neighbours[0][idx])
        nxt = torch.from_numpy(neighbours[1][idx])
        x = torch.stack([self.resid[idx], self.resid[prev], self.resid[nxt]], 1).float()
        return pad_frames(x)

    def fit(self, train_idx, neighbours, steps, log_every=250, log=print):
        sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=self.lr, total_steps=steps, pct_start=0.25)
        self.model.train()
        rng = np.random.default_rng(1234 + self.seed)
        running, seen = 0.0, 0
        for step in range(steps):
            idx = rng.choice(train_idx, self.batch_size, replace=False)
            x = self._inputs(idx, neighbours).to(self.device, non_blocking=True)
            hm, off, msk = make_targets([self.pixel_centers[i] for i in idx])
            hm = torch.from_numpy(hm).to(self.device, non_blocking=True)
            off = torch.from_numpy(off).to(self.device, non_blocking=True)
            msk = torch.from_numpy(msk).to(self.device, non_blocking=True)
            x, hm, off, msk = augment(x, hm, off, msk, self.gen)
            with torch.amp.autocast(self.device, dtype=torch.float16, enabled=self.cuda):
                phm, poff = self.model(x)
                loss_hm = focal_loss(phm.float(), hm)
                denom = msk.sum().clamp(min=1.0)
                loss_off = (torch.abs(poff.float() - off) * msk).sum() / (2 * denom)
                loss = loss_hm + 2.0 * loss_off
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            sched.step()
            running += float(loss.detach())
            seen += 1
            if (step + 1) % log_every == 0:
                log(f"  step {step + 1}/{steps}  loss {running / seen:.4f}")
                running, seen = 0.0, 0
        return self.model
