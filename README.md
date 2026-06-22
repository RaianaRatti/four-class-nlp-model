# VADNet — 4-Class Voice Activity Detector

A custom-trained voice activity detection model that classifies audio frames into four classes: **silence**, **speech**, **overlap**, and **vocalization**. Built with PyTorch and trained on a ~627K sample dataset drawn from LibriSpeech, AMI, and ESC-50.

---

## Why four classes?

Most off-the-shelf VAD models output a binary signal (speech / no speech). This model was built to support a real-time speaker diarization pipeline that needs finer-grained awareness:

| Class | Description |
|---|---|
| `silence` | No audio activity |
| `speech` | Single speaker |
| `overlap` | Two or more speakers talking simultaneously |
| `vocalization` | Non-speech vocal sounds (laughter, coughs, etc.) |

Distinguishing overlap from clean speech is particularly important for diarization — sending overlapping audio to a speaker encoder produces unreliable embeddings, so overlap frames are handled separately downstream.

---

## Architecture

`VADNet` is a lightweight convolutional classifier that operates on log-mel spectrogram features.

- Input: log-mel spectrogram frames (extracted with `torchaudio`)
- Backbone: stacked Conv1D + BatchNorm + ReLU blocks
- Output: 4-class logits (softmax at inference)
- Framework: PyTorch

The model is intentionally small — it needs to run in real time alongside audio capture, VAD, speaker encoding, and clustering without becoming a bottleneck.

---

## Dataset

Training data was drawn from three sources and merged into a single `all_labels.csv`:

| Source | Content | Contribution |
|---|---|---|
| [LibriSpeech](https://www.openslr.org/12) | Clean read speech | `silence`, `speech` |
| [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Meeting recordings | `speech`, `overlap` |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Environmental sounds | `vocalization`, `silence` |

Total samples after labeling and merging: ~627K. Features were precomputed to `.npy` files before training to avoid repeated extraction overhead.

### Label scripts

- `label_librispeech.py`
- `label_ami.py`
- `label_esc50.py`
- `merge_labels.py` — combines outputs into `all_labels.csv`

---

## Training

### Requirements

```
torch
torchaudio
numpy
```

### Precompute features

```bash
python precompute_features.py
```

This writes `preprocessed_features/features.npy` and `preprocessed_features/labels.npy`.

### Run training

```bash
python train.py
```

Training runs for 30 epochs with an 85/15 train/val split. The best checkpoint (by validation loss) is saved to `models/custom_vad.pt`. A rolling `models/checkpoint_latest.pt` is also saved every epoch to support resumption if training is interrupted.

### Key hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 256 |
| Learning rate | 1e-3 (Adam) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Class weights | `[1.0, 1.0, 2.0, 2.0]` |
| Label smoothing | 0.05 |
| Gradient clipping | max norm 1.0 |

Class weights upweight `overlap` and `vocalization` since they are underrepresented in the training data. The validation set is kept augmentation-free to give a clean loss signal.

### Best checkpoint

The best model was saved at **epoch 13** with balanced class weights. Earlier training runs used imbalanced weights (`[0.5, 1.0, 2.0, 2.0]`) which caused silence to be over-suppressed at inference — this was corrected in the final training run.

---

## Inference

```python
import torch
from ml.model import VADNet

CLASS_NAMES = ["silence", "speech", "overlap", "vocalization"]

model = VADNet()
model.load_state_dict(torch.load("models/custom_vad.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(features)          # features: your log-mel tensor
    label  = logits.argmax(dim=1)
    print(CLASS_NAMES[label.item()])
```

---

## Repo structure

```
.
├── ml/
│   ├── model.py          # VADNet architecture
│   └── dataset.py        # VADDataset (loads .npy features, optional augmentation)
├── train_data/
│   ├── audio/            # raw audio files
│   └── labels/
│       └── all_labels.csv
├── preprocessed_features/
│   ├── features.npy
│   └── labels.npy
├── models/
│   ├── custom_vad.pt          # best model weights
│   └── checkpoint_latest.pt   # rolling checkpoint for resumption
├── label_librispeech.py
├── label_ami.py
├── label_esc50.py
├── merge_labels.py
├── precompute_features.py
└── train.py
```

---

## Context

VADNet was built as a component of a larger real-time speaker diarization system. The diarization pipeline feeds VADNet's output into a class-aware state machine that routes audio frames to different downstream handlers — clean speech frames go to speaker encoding and clustering, overlap frames trigger speech separation (SepFormer), and silence/vocalization frames are used to manage segment boundaries.