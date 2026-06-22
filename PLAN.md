# Improvement Plan

## Current state (epoch 29/30)

| Class       | Accuracy | Samples | Share  |
|-------------|----------|---------|--------|
| silence     | 50.3%    | 2,572   | 5.2%   |
| speech      | 73.5%    | 8,510   | 17.2%  |
| overlap     | 93.4%    | 25,000  | 50.5%  |
| vocalization| 71.7%    | 13,411  | 27.1%  |
| **overall** | **81.9%**|         |        |
| **bal_acc** | **72.2%**|         |        |

**Root cause of silence at 50.3%:** data imbalance. Silence has 10x fewer
samples than overlap. Class weights of 4.0 partially compensate but can't
overcome a 5% vs 51% split — the model sees ~10 overlap frames for every
silence frame per epoch.

---

## Priority 1 — Balanced batch sampling (no reprecompute, high impact)

Replace `shuffle=True` in the train loader with `WeightedRandomSampler`.
This makes every batch contain roughly equal class representation, regardless
of dataset size.

**Where:** `ml/train.py` → `make_split_loaders()`

```python
from torch.utils.data import WeightedRandomSampler

# Compute per-sample weights inversely proportional to class frequency
label_counts = np.bincount(full_train.labels[train_idx])
class_weights = 1.0 / label_counts
sample_weights = class_weights[full_train.labels[train_idx]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

# Pass sampler instead of shuffle=True
train_loader = DataLoader(..., sampler=sampler, shuffle=False)
```

Expected effect: silence goes from ~5% of each batch to ~25%.

---

## Priority 2 — Train longer (no reprecompute, easy)

Silence improved from 15.1% → 50.3% across 30 epochs — it's still on an
upward curve, not plateaued. Extend training.

**Where:** `ml/train.py`

```python
EPOCHS = 60  # was 30
```

Combined with the balanced sampler, the model will actually see enough silence
examples to converge.

---

## Priority 3 — Add more silence training data (reprecompute required)

2,572 silence samples is too few. Two cheap sources:

1. **Synthetic silence:** generate frames of zeros and very-low-amplitude
   Gaussian noise (simulating a quiet room). 5,000–10,000 frames takes seconds
   to create with numpy.
2. **MUSAN noise corpus** (free, ~11GB): contains music, speech, and noise
   clips. The noise subset provides realistic ambient/silence-like recordings.

Add generated samples to `train_data/audio/` and entries to `all_labels.csv`,
then re-run `python training/precompute_features.py`.

Target: bring silence up to at least 8,000–10,000 samples (≥15% of dataset).

---

## Priority 4 — Temporal context window (reprecompute required, medium effort)

The model currently sees one 30ms frame in isolation. Many confusions happen
at boundaries: a single frame of speech surrounded by silence looks ambiguous
without context. Feeding N consecutive frames gives the model the "before and
after" it needs.

**Approach:**
- In `precompute_features.py`, instead of one feature vector per frame, store
  a sliding window of 5 frames (5 × 127 = 635-dim vector).
- Update `INPUT_DIM = 635` in `ml/model.py`.
- No architecture change beyond the input dim.

This particularly helps overlap detection (two voices starting/stopping) and
silence boundaries.

---

## Priority 5 — CNN on 2D mel-spectrogram (bigger rewrite, highest ceiling)

The current MLP operates on averaged MFCC vectors — averaging collapses all
temporal structure within each frame. A small CNN operating on the 2D
mel-spectrogram retains spatial/frequency patterns that the MLP cannot see.

**Approach:**
- Change `extract_features` to return a 2D mel-spectrogram patch (e.g. 64
  mel bins × 3 time steps for a 30ms frame) instead of a 1D averaged vector.
- Replace `VADNet` with a small CNN: 2–3 conv layers → global avg pool →
  2-layer MLP head → 4 classes.
- Re-run precompute.

Expected gain: 3–6 points on bal_acc. This is the architectural ceiling of
the "train from scratch" approach.

---

## Priority 6 — Pre-trained audio encoder (highest effort, highest ceiling)

Fine-tune a frozen `wav2vec2-base` or `openai/whisper-tiny` encoder as a
feature extractor, then train only a small classifier head on top. These
models already understand speech structure deeply.

Expected gain: bal_acc likely jumps to 85–90%+ with minimal additional data.
Requires GPU and rewriting the feature pipeline to pass raw waveforms instead
of precomputed features.

---

## Recommended order

1. Balanced sampler + EPOCHS=60 — run overnight, free gains
2. Add silence data (synthetic) + reprecompute — one afternoon of work
3. Temporal context window — if silence still lags after 1 & 2
4. CNN architecture — if bal_acc plateaus below 80%
5. Pre-trained encoder — if target accuracy justifies the complexity
