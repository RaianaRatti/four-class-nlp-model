# VADNet: Four-Class Conversational Audio Classifier

A lightweight PyTorch model that classifies short audio frames into four conversational categories: **silence**, **speech**, **overlap**, and **vocalization**. Built from scratch as part of a real-time speaker diarization pipeline.

---

## What It Does

Most voice activity detectors (VADs) answer a single binary question: *is someone speaking?* VADNet goes further, distinguishing between:

| Class | Description |
|---|---|
| `silence` | No speech activity, background noise, pauses |
| `speech` | A single speaker talking |
| `overlap` | Two or more speakers talking simultaneously |
| `vocalization` | Non-linguistic sounds, laughter, coughing, breathing |

This richer labeling feeds directly into a diarization pipeline, where knowing *when* speakers overlap is as important as knowing *who* is speaking.

---

## Results

### Final Model (Run C)

```
Overall accuracy:   83.4%
Balanced accuracy:  82.4%

Per-class accuracy:
  Silence        93.7%
  Speech         83.5%
  Overlap        73.8%
  Vocalization   78.7%
```

Balanced accuracy is reported alongside overall accuracy to penalise models that ignore minority classes. A model that predicts "speech" for everything would score ~40% on overall accuracy but near 25% balanced. This distinction was tracked throughout training.

### Experiment History

| Run | Key Change | Balanced Accuracy |
|---|---|---|
| A | Baseline, equal class weights | 75.7% |
| B | WeightedRandomSampler + weight tuning | 82.0% |
| C | Gradient clipping, label smoothing, mixup | 82.4% |

---

## Architecture

`VADNet` is a feedforward neural network operating on 127-dimensional feature vectors extracted from 30ms audio frames.

```
Input (127)
    │
    ▼
Linear(127 → 256) → LayerNorm → ReLU → Dropout(0.3)
    │
    ▼
Linear(256 → 256) → LayerNorm → ReLU → Dropout(0.3)
    │
    ▼
Linear(256 → 128) → ReLU
    │
    ▼
Linear(128 → 4)   ← raw logits
```

Roughly 100K parameters, small enough to run in real time on CPU as part of a live pipeline.

---

## Features (127-dim vector per frame)

Each audio frame is converted to a fixed-length feature vector before training or inference:

| Feature | Dim | Purpose |
|---|---|---|
| MFCC means | 40 | Timbral/spectral content |
| MFCC delta means | 40 | First-order temporal dynamics |
| MFCC delta-delta means | 40 | Second-order temporal dynamics |
| Log energy | 1 | Frame loudness |
| Zero-crossing rate | 1 | Noisiness / unvoiced content |
| Spectral flatness | 1 | Tonal vs. noise-like signal |
| Spectral centroid | 1 | Perceived brightness |
| Spectral rolloff | 1 | Energy distribution |
| Voiced fraction | 1 | Proportion of voiced frames |
| F0 mean | 1 | Fundamental frequency (pitch) |

Pitch features (`voiced_frac`, `f0_mean`) were key for separating vocalization from speech, as laughter and non-speech sounds occupy a different F0 range than conversational speech.

---

## Training Details

### Dataset

Training data was sourced and labelled from three corpora:

- **LibriSpeech** — clean read speech (`speech`)
- **AMI Meeting Corpus** — multi-speaker meetings (`speech`, `overlap`)
- **ESC-50** — environmental sounds and vocalizations (`silence`, `vocalization`)

Labels were assigned per frame using energy thresholds and speaker turn annotations. Final dataset: ~50K labelled frames after capping per-source contribution to avoid skew.

### Class Imbalance Strategy

The raw dataset was heavily skewed toward speech and silence. Two strategies were combined:

1. **WeightedRandomSampler** — each training batch is resampled so every class appears roughly equally, regardless of dataset frequency
2. **Class-weighted cross-entropy loss** — minority classes (overlap, vocalization) incur higher loss penalties when misclassified

```python
# Loss weights: [silence, speech, overlap, vocalization]
weights = torch.tensor([1.0, 1.0, 1.3, 1.2])
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
```

### Data Augmentation

Applied to training frames only (never validation):

| Augmentation | Probability | Purpose |
|---|---|---|
| Gaussian noise | 50% | Microphone/environment variation |
| MFCC frequency masking | 50% | SpecAugment-style frequency dropout |
| Delta block zeroing | 30% | SpecAugment-style time masking |
| Delta-delta masking | 20% | Second-order dropout |
| Volume scale jitter | 40% | Recording level variation |

### Other Training Choices

- **Mixup** (beta=0.2): blends pairs of training samples to smooth decision boundaries
- **Gradient clipping** (max norm=1.0): prevents loss spikes during early training
- **ReduceLROnPlateau** scheduler: halves LR after 3 epochs without validation improvement
- **Best checkpoint** saved on balanced accuracy, not validation loss, ensuring the saved model performs well across all classes and not just the majority

---

## Project Structure

```
voice-detection/
├── ml/
│   ├── dataset.py          # VADDataset, feature extraction
│   ├── model.py            # VADNet architecture
│   └── labeling/
│       ├── label_librispeech.py
│       ├── label_ami.py
│       ├── label_esc50.py
│       └── merge_labels.py
├── training/
│   ├── train.py            # Training loop with checkpointing
│   └── evaluate.py         # Per-class evaluation on held-out set
├── config.py               # Shared constants (SAMPLE_RATE, paths)
├── models/
│   └── custom_vad.pt       # Best saved checkpoint
└── preprocessed_features/
    ├── features.npy        # Precomputed 127-dim feature vectors
    └── labels.npy          # Corresponding string labels
```

Features are precomputed and saved to `.npy` files before training. This avoids re-extracting 50K frames on every epoch and speeds up the training loop significantly.

---

## Running It

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch librosa numpy pandas
```

### Creating Training Data

```bash
python ml/labeling/label_librispeech.py
python ml/labeling/label_ami.py
python ml/labeling/label_esc50.py
python ml/labeling/merge_labels.py
```

### Precompute Features

```bash
python training/precompute_features.py
```

### Train

```bash
python training/train.py
```

Training automatically saves a checkpoint after every epoch and resumes from it if interrupted. The best model (by balanced accuracy) is saved to `models/custom_vad.pt`.

### Evaluate

```bash
python training/evaluate.py
```

Reports overall accuracy and per-class accuracy on a held-out validation split.

---

## Future Work

- [ ] Add a confusion matrix to evaluate class-level confusion patterns (e.g. overlap misclassified as speech)
- [ ] Experiment with a sliding-window LSTM or Transformer to capture temporal context across frames
- [ ] Add spectral entropy and harmonic ratio features to better discriminate overlap
- [ ] Build an audio demo: drop in a `.wav` file and visualise a timeline of predicted classes
- [ ] Upload best checkpoint to Hugging Face Hub for reproducibility

---

## Repository

[github.com/RaianaRatti/voice_detection](https://github.com/RaianaRatti/voice_detection)