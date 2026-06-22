import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import SAMPLE_RATE

LABEL_MAP = {"silence": 0, "speech": 1, "overlap": 2, "vocalization": 3}
N_MFCC = 40

def extract_features(frame: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32) / 32768.0

    mfcc    = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC, n_fft=480, hop_length=160)
    delta   = librosa.feature.delta(mfcc, mode="nearest")
    delta2  = librosa.feature.delta(mfcc, order=2, mode="nearest")
    energy  = np.array([np.log(np.sum(frame ** 2) + 1e-8)])

    # Discriminative spectral features
    zcr               = np.array([librosa.feature.zero_crossing_rate(frame).mean()])
    spectral_flatness = np.array([librosa.feature.spectral_flatness(y=frame).mean()])
    spectral_centroid = np.array([librosa.feature.spectral_centroid(y=frame, sr=sr).mean() / sr])
    spectral_rolloff  = np.array([librosa.feature.spectral_rolloff(y=frame, sr=sr).mean() / sr])

    # Pitch (F0): voiced_fraction separates speech/vocalization from silence/noise;
    # f0_mean distinguishes vocalization pitch range from normal speech
    f0           = librosa.yin(frame, fmin=75, fmax=400, sr=sr,
                               frame_length=480, hop_length=160)
    voiced       = f0 < 400                                          # yin returns fmax when unvoiced
    voiced_frac  = np.array([voiced.mean()])
    f0_mean      = np.array([f0[voiced].mean() / sr if voiced.any() else 0.0])

    features = np.concatenate([
        mfcc.mean(axis=1),        # 40
        delta.mean(axis=1),       # 40
        delta2.mean(axis=1),      # 40
        energy,                   # 1
        zcr,                      # 1
        spectral_flatness,        # 1
        spectral_centroid,        # 1
        spectral_rolloff,         # 1
        voiced_frac,              # 1
        f0_mean,                  # 1
    ])                            # total: 127

    return features.astype(np.float32)

class VADDataset(Dataset):
    def __init__(self, features_npy: str, labels_npy: str, augment: bool = False):
        """Load precomputed features instead of extracting on-the-fly"""
        self.features = np.load(features_npy).astype(np.float32)  # (N, 127)
        self.labels_str = np.load(labels_npy, allow_pickle=True)   # (N,)
        self.labels = np.array([LABEL_MAP[l] for l in self.labels_str])
        self.augment = augment

    def _augment(self, features: np.ndarray) -> np.ndarray:
        # Feature vector layout:
        #   [0:40]   MFCC means        ← frequency axis
        #   [40:80]  delta means       ← 1st-order temporal dynamics
        #   [80:120] delta2 means      ← 2nd-order temporal dynamics
        #   [120:]   energy, ZCR, flatness, centroid, rolloff, voiced_frac, f0_mean

        features = features.copy()

        # 1. Gaussian noise
        if np.random.rand() < 0.5:
            features += (np.random.randn(*features.shape) * 0.025).astype(np.float32)

        # 2. SpecAugment frequency masking: zero a contiguous band of MFCC
        #    coefficients — equivalent to masking a frequency range on a spectrogram
        if np.random.rand() < 0.5:
            start = np.random.randint(0, 33)
            width = np.random.randint(2, 8)
            features[start:start + width] = 0.0

        # 3. Time masking analog: zero the delta block entirely — simulates a
        #    frame with no detectable temporal change (SpecAugment T-mask equiv.)
        if np.random.rand() < 0.3:
            features[40:80] = 0.0

        # 4. Delta-delta masking: independently drop second-order dynamics
        if np.random.rand() < 0.2:
            features[80:120] = 0.0

        # 5. Scale jitter: simulates recording volume variation
        if np.random.rand() < 0.4:
            features *= np.random.uniform(0.8, 1.2)

        return features.astype(np.float32)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        if self.augment:
            features = self._augment(features)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)