# Improvement Plan

## Current State

| Metric | Value |
|---|---|
| Overall accuracy (eval) | 96.34% |
| Balanced accuracy (best epoch) | 92.5% |
| Best epoch | 51/60 |

| Class | Eval Accuracy |
|---|---|
| Silence | 92.65% |
| Speech | 96.79% |
| Overlap | 98.05% |
| non-vocal | 99.13% |

---

## Completed Work

| Fix | File(s) |
|---|---|
| Feature normalization (mean/std saved and applied at train + inference) | `precompute_features.py`, `dataset.py`, `demo.py` |
| Removed `f0_mean` (caused speech to be misclassified as non-vocal) | `dataset.py` |
| Added `spectral_entropy` and `harmonic_ratio` for overlap discrimination | `dataset.py` |
| Energy gate for near-silent frames at inference | `demo.py` |
| Synthetic silence rows (zeros + low-amplitude noise) in LibriSpeech labeling | `label_librispeech.py` |
| ESC-50 removed due to spectral ambiguity | `label_esc50.py` removed |
| AMI overlap fixed: per-speaker binary masks instead of combined counter | `label_ami.py` |
| AMI word boundary shrink (20ms) to prevent false overlap at turn edges | `label_ami.py` |
| non-vocal tightened to laughter, coughing, sneezing only | `label_ami.py` |
| Confusion matrix added to evaluate | `evaluate.py` |
| Noise, MP3 compression, reverb augmentation on LibriSpeech frames | `label_librispeech.py` |
| ResBlock architecture replacing flat MLP | `model.py` |
| Mixup, gradient clipping, label smoothing | `train.py` |
| EPOCHS extended to 60 | `train.py` |
| Sliding window LSTM considered, applied and rejected | N/A |

---

## Remaining Priorities

### Priority 1 — Energy gate during non-vocal labeling

- 190 non-vocal frames in the confusion matrix were predicted as silence
- These are silent gaps inside AMI vocalsound clips being stamped as non-vocal
- Fix: apply the same RMS threshold used in `demo.py` during labeling

```python
# In label_ami.py when emitting non-vocal rows:
rms = np.sqrt(np.mean(frame ** 2))
label = "non-vocal" if rms >= 0.02 else "silence"
```

- Rerun: `label_ami.py` -> `merge_labels.py` -> `precompute_features.py` -> retrain

---

### Priority 2 — Add VoxCeleb2 for real-world speech coverage

- LibriSpeech is studio-quality audiobook audio
- Real-world inference audio is conversational, compressed, and noisier
- VoxCeleb2 contains YouTube interview clips across many speakers, accents, and mic types
- Add `label_voxceleb.py` following the same pattern as `label_librispeech.py`
- Target: 10,000 to 15,000 additional speech frames

---

### Priority 3 — Synthesize overlap from speaker pairs

- All overlap currently comes from AMI (one acoustic environment)
- Mixing two VoxCeleb2 speakers programmatically adds diversity and volume control

```python
overlap_frame = speaker_a * alpha + speaker_b * (1 - alpha)
# alpha ~ uniform(0.4, 0.6) keeps both speakers audible
```

- Cleaner and more controllable than extracting more meeting recordings

---

### Priority 4 — CNN on mel-spectrogram

- Current MLP averages MFCC vectors, collapsing temporal structure within each frame
- A small CNN on a 2D mel-spectrogram patch retains frequency and time patterns
- Approach: 64 mel bins x 3 time steps per 30ms frame, 2 to 3 conv layers, global avg pool, MLP head
- Estimated gain: 3 to 6 points on balanced accuracy
- Requires rewriting `extract_features` and `VADNet`, and re-running precompute

---

### Priority 5 — Pre-trained audio encoder

- Fine-tune a frozen `wav2vec2-base` or `openai/whisper-tiny` as a feature extractor
- Train only a small classifier head on top
- Estimated balanced accuracy: 85 to 90%+
- Requires GPU and rewriting the feature pipeline to pass raw waveforms

---

## Recommended Next Step

Run **Priority 1** first. It is a one-afternoon fix with no new data required and directly targets the largest remaining confusion in the matrix.