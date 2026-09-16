# Ant Colony Gate Crossings

Predict every ant-centre crossing of a virtual gate — time, direction and along-gate
position — from 20 grayscale frames per query.

Single command, trains from randomly initialised weights on one CUDA GPU and writes the
submission:

```bash
python3 solution.py <public_dir> submission.csv
```

`<public_dir>` is the directory holding `frames/`, `train.csv`, `test.csv`,
`frame_training.csv`. A copy of the dataset ships in [`data/`](data/), so
`python3 solution.py data submission.csv` works straight after cloning.

---

## The one thing that matters

The task decomposes cleanly, and the decomposition was measured, not assumed:

**Feed the supplied ground-truth centres through the association + geometry stage and the
score is 0.9984.** Tracking and the crossing rule are effectively exact. Everything the
model has to earn is in one place: *recover each ant centre in a test frame as precisely as
the annotator placed it.*

Every number in this section is printed by [`tools/ceiling.py`](tools/ceiling.py), which is
CPU-only and takes about a minute — run it first.

That reframes the problem from "detect crossing events" to "detect points", which is a far
better-posed learning problem and is exactly what the extra `frame_training.csv`
supervision is for.

### Sub-pixel precision is the binding constraint

Perturbing the ground-truth centres with Gaussian jitter and re-running the pipeline:

| centre jitter | end-to-end score |
| ------------- | ---------------- |
| 0 px (exact)  | **0.9984**       |
| 0.5 px        | 0.8716           |
| 1.0 px        | 0.7767           |
| 2.0 px        | 0.6122           |
| 3.0 px        | 0.4979           |

Half a pixel of noise costs 13 points. The reason is structural: an ant loitering *on* the
gate has a near-zero signed distance, so tiny position noise flips the sign back and forth
and manufactures crossings that are not in the target list. Every spurious event also
inflates the metric's denominator.

**The pixel noise is not what stops you.** Treating each ant as a template plus the
described noise, the Cramér–Rao bound on centre localisation works out at **0.057 px**
median from real background-subtracted patches (0.097 px for the faintest decile), after
subtracting the gradient energy the noise itself contributes. That is an order of magnitude
finer than the 0.5 px that already costs 13 points. It is an idealised bound — it assumes a
known template and ignores overlapping ants — but the direction is unambiguous: the
information is present in the pixels, so accuracy is limited by the model and its training
budget, not by the sensor noise the task describes. Spending the budget on the detector pays.

Two consequences drive the design:

- The detector head regresses a **dense sub-pixel offset field**, not just a heatmap peak,
  and inference averages over the four axis flips. Integer-pixel peaks alone are not enough.
- Missing a detection is far worse than an extra *random* one. Dropping 10% of centres
  costs 16 points; two uniformly random false positives per frame cost 1. But see the
  warning below — this does **not** mean you should drive the threshold to zero.

| perturbation          | score  |
| --------------------- | ------ |
| drop 2% of centres    | 0.9757 |
| drop 5%               | 0.9305 |
| drop 10%              | 0.8352 |
| drop 20%              | 0.6979 |
| +0.5 false pos./frame | 0.9977 |
| +1.0 false pos./frame | 0.9938 |
| +2.0 false pos./frame | 0.9862 |

Random false positives are close to free for a structural reason: a spurious detection
uncorrelated between frames leaves a one-frame track, and a one-frame track has no
consecutive pair, so it emits no event. Only a *persistent* phantom costs anything.

**A real detector's false positives are not random, and this is where the simulation
misleads.** Sweeping the threshold on an actual trained model, lowering it buys recall and
*loses* score:

| threshold | recall | precision | detections/frame | end-to-end |
| --------- | ------ | --------- | ---------------- | ---------- |
| 0.05      | 0.643  | 0.497     | 35.0             | 0.5188     |
| 0.10      | 0.638  | 0.558     | 31.0             | 0.5453     |
| 0.15      | 0.628  | 0.587     | 29.0             | 0.5578     |
| 0.20      | 0.611  | 0.604     | 27.4             | 0.5675     |
| **0.25**  | 0.604  | 0.636     | 25.7             | **0.5713** |
| 0.30      | 0.590  | 0.667     | 24.0             | 0.5648     |
| 0.35      | 0.489  | 0.691     | —                | 0.4233     |

A weak peak sits on the *same* image structure frame after frame, so it is exactly the
persistent phantom the random model does not capture: it survives association, forms a
multi-frame track, and emits events. Tune the threshold empirically with
`tools/sweep_decode.py`; do not infer it from the perturbation table.

### Do not smooth the trajectories

The obvious defence against jitter — temporally smoothing each track before applying the
crossing rule — is actively harmful. With *exact* centres, a 3-frame moving average drops
the score from 0.9984 to 0.7737:

| smoothing window | score on exact centres |
| ---------------- | ---------------------- |
| none             | **0.9984**             |
| 3 frames         | 0.7737                 |
| 5 frames         | 0.6731                 |
| 7 frames         | 0.6130                 |

The targets are defined by linear interpolation between *raw annotated* centres, so the
rapid back-and-forth of an ant hovering at the gate produces genuine target events.
Smoothing erases real signal faster than it suppresses noise. The fix for jitter is a
better detector, not a filter.

---

## Structure recovered from the data

The 20-frame windows are overlapping views of a small number of continuous recordings.
Linking each window's consecutive frames and taking connected components gives a successor
relation that is a **function** — every component is a simple path, with exactly one head:

- **train:** 8 recordings of 500, 330, 300, 270, 260, 170, 170, 170 frames (2170 total)
- **test:** 2 recordings of 260 and 170 frames (430 total)

This matches the stated "two complete recording sequences are held out from ten". Three
things follow:

1. **Association runs once per recording, over its full chain** — not per query. Tracks are
   far more stable over 500 frames than over 20, and each query then just reads its
   20-frame window off the global tracks.
2. **Validation must hold out whole recordings.** Windows overlap heavily, so a random
   frame or row split leaks neighbouring frames of the same ants into training.
   [`tools/validate.py`](tools/validate.py) holds out recordings 4 and 5, which are 260 and
   170 frames — the same sizes as the two hidden test recordings.
3. **Per-recording background estimation is available at inference.** The arena is static,
   so a pixelwise median over a recording's frames removes it and leaves the ants. This is
   ordinary preprocessing of the supplied test images, and uses no labels.

Measured motion statistics, which set the association gate: matched frame-to-frame
displacement is 3.0 px median and 9.5 px at the 95th percentile, while the
nearest-neighbour spacing between ants in a frame is 27.8 px median. Association is
well-conditioned; `max_dist = 25 px` was the best of the values swept.

---

## Pipeline

```
PNG frames ─┬─> recording chains (union-find on window adjacency)
            └─> per-recording median background ─> residual = background − frame
                                                          │
                          ┌───────────────────────────────┘
                          v
            3-channel input [residual(t−1), residual(t), residual(t+1)]
                          │
                          v
         U-Net, random init, full-resolution output
              ├── centre heatmap   (penalty-reduced focal loss)
              └── sub-pixel offset (L1 on a ±2 px disc around each centre)
                          │
                          v
     5×5 NMS + threshold + offset, averaged over 4 axis flips  ─> centres
                          │
                          v
     Hungarian association along each recording (gate 25 px, 1-frame gap tolerance)
                          │
                          v
     For each query window and consecutive pair with the track present at both ends:
       d0 < 0 ≤ d1 or d1 < 0 ≤ d0  ->  α = −d0/(d1−d0)
       time = i + α,  position = lerp(along-gate coordinate, α),  direction = sign
```

The geometry stage is a direct transcription of the rule in the task statement; it is not
learned and has no tunable parameters.

### Why this input representation

- **Background residual** normalises appearance across recordings. With only 8 recordings
  to learn from and 2 unseen ones to predict, handing the model a background-invariant
  signal is the cheapest generalisation win available.
- **Neighbouring frames** let the model separate ants from the per-frame nuisances the task
  describes (smooth spot artifacts, row offsets, sensor outliers), which do not move
  coherently the way ants do.

---

## Layout

| path | role |
| ---- | ---- |
| `solution.py` | end-to-end entry point: train, detect, associate, write CSV |
| `acg/data.py` | table loading, recording reconstruction, background estimation |
| `acg/bank.py` | decode every frame once into a background-removed frame bank |
| `acg/model.py` | the U-Net (heatmap + offset heads), randomly initialised |
| `acg/train.py` | target rasterisation, focal loss, augmentation, training loop |
| `acg/detect.py` | flip-averaged decoding to sub-pixel centres |
| `acg/track.py` | association and the crossing-extraction geometry |
| `acg/metric.py` | the official Hungarian row metric |
| `tools/test_geometry.py` | asserts the crossing rule and metric against the statement |
| `tools/ceiling.py` | reproduces the evidence above (CPU only, ~1 min) |
| `tools/sweep_decode.py` | tunes decode/association against a saved model, no retraining |
| `tools/validate.py` | held-out-recording validation, end to end |
| `tools/benchmark.py` | GPU throughput probe used to size the step budget |
| `tools/check_submission.py` | validates a submission against every format rule |
| `data/` | the public dataset |

## Running

```bash
pip install -r requirements.txt          # install torch for your CUDA build first

python3 tools/test_geometry.py --data data   # CPU, seconds: geometry + metric checks
python3 tools/ceiling.py --data data     # CPU, ~1 min: reproduces the tables above
python3 tools/benchmark.py               # step time on this GPU -> recommended STEPS
python3 solution.py data submission.csv  # the graded run
python3 tools/check_submission.py data submission.csv

# held-out-recording validation (trains a model, reports the real metric)
python3 tools/validate.py --data data --holdout 4 5 --steps 2500 \
    --threshold 0.15 0.25 0.35 --fill-gap 0 1 --save-model m.pt
```

Before any GPU run, check the card is actually free:

```bash
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
```

A previous run that was interrupted can leave a CUDA context alive holding VRAM. The
symptom is not an error — training simply runs ~25× slower as it spills to host memory,
or the process dies with no traceback. Kill leftovers before timing anything.

### Budget

The run must finish offline within 90 minutes on one A10G-class GPU. Everything outside the
training loop — decoding 2600 PNGs, backgrounds, flip-averaged detection over 430 test
frames, association, CSV — is a few minutes; the rest is training.

`STEPS` and `N_MODELS` in [`solution.py`](solution.py) are **fixed constants**, deliberately
not derived from a wall-clock check, so the configuration is fully deterministic. Size them
on the target GPU with `tools/benchmark.py` and edit the constants before the graded run.

Peak GPU memory is ~2.2 GB at batch 8 and ~4.5 GB at batch 16, so an A10G can take a larger
batch than the default.

> Note: `channels_last` memory format was measured **5.2× slower** than contiguous for this
> model on this workload. Do not re-enable it without re-timing.

## Measured end to end

Held-out recordings 4 and 5 (260 and 170 frames — the sizes of the two hidden test
recordings), 330 queries, one model trained for 2500 steps at batch 4 on a **contended
laptop RTX 3050**:

| | |
| --- | --- |
| End-to-end score | **0.5713** |
| Detection recall / precision | 0.604 / 0.636 |
| Localisation error (matched, mean) | 1.83 px |
| Perfect-centre ceiling | 0.9984 |

**Treat this as a floor, not a forecast.** It is ~5.7 epochs; the shipped defaults are
9000 steps at batch 8 across 2 models, roughly 6× the training, and the loss was still
falling steeply when this run ended. The two gaps are plain in the numbers and both are
training-limited:

- **Localisation is 1.83 px** against a noise floor of 0.057 px. The jitter table says
  2 px alone caps you near 0.61, so this is the dominant loss. Nothing about the data
  forces it — the information is in the pixels.
- **Recall is 0.60 even at threshold 0.05**, so ~36% of ants are not found at any
  threshold. That is the model failing to fire, not a decode setting.

Both improve with training budget, model capacity and ensembling, which is where the A10G
run should spend its 90 minutes.

## If you want to push the score higher

Everything here moves detection quality, because that is the only thing that moves the
score. Ordered by expected value per unit of effort:

1. **More steps and a wider model**, sized by `tools/benchmark.py` — by far the biggest
   lever, because both measured gaps (1.83 px localisation, 0.60 recall) are training
   limited. An A10G has 24 GB, so `--batch-size 16` or more raises images/second.
2. **More models in the ensemble.** `detect_stack` averages heatmaps and offsets across
   models. Averaging is exactly the operation that suppresses localisation jitter, which is
   the dominant error term — the same mechanism that makes the 4-flip TTA worth its cost.
3. **Re-sweep `THRESHOLD` once trained**, with `tools/sweep_decode.py` against a saved
   model — it needs no retraining, one detection pass per threshold. The optimum moves with
   detector quality, so the shipped 0.25 is a starting point, not a constant of nature.
4. **Re-check the `max_gap` / `fill_gap` pair.** These two go as a pair. `max_gap` is the
   largest frame separation association will bridge, so the default `1` means
   consecutive-only and one missed detection ends the track; `2` lets the track survive a
   one-frame hole. Even then the crossing rule needs a position at *both* endpoints, so
   `fill_gap=1` interpolates across the hole. Partial credit beats none. Sweep with
   `--fill-gap 0 1`, which is already crossed against `max_gap` in `tools/validate.py`.

Measured dead ends, so you do not spend GPU time rediscovering them:

- **Trajectory smoothing** — see the table above. Costs 22 points on exact centres.
- **`channels_last`** — 5.2× slower here.
- **Per-query tracking** — association over a full 500-frame recording is strictly more
  stable than over a 20-frame window, and costs nothing because each query simply reads its
  window off the global tracks.
- **A classical dark-blob detector** — the annotated centre is the centroid of a soft,
  radially symmetric blob (peak contrast 0.27, radius ~8 px), and a matched-filter peak
  finder lands 2.7 px away on average. That is off the bottom of the jitter table.

## Constraints

Every weight starts from a random initialisation inside `acg/model.py` and is fitted
entirely on the public training frames and the centres in `frame_training.csv`. The run is
offline and self-contained: the only inputs it opens are the files under the public
directory. Test frames are used for forward inference and for their own per-recording
background statistic; test targets are never read, and no decode parameter is tuned on
them — `tools/validate.py` tunes on held-out *training* recordings.
