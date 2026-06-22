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

    # New: discriminative spectral features
    zcr               = np.array([librosa.feature.zero_crossing_rate(frame).mean()])
    spectral_flatness = np.array([librosa.feature.spectral_flatness(y=frame).mean()])
    spectral_centroid = np.array([librosa.feature.spectral_centroid(y=frame, sr=sr).mean() / sr])
    spectral_rolloff  = np.array([librosa.feature.spectral_rolloff(y=frame, sr=sr).mean() / sr])

    features = np.concatenate([
        mfcc.mean(axis=1),        # 40
        delta.mean(axis=1),       # 40
        delta2.mean(axis=1),      # 40
        energy,                   # 1
        zcr,                      # 1
        spectral_flatness,        # 1
        spectral_centroid,        # 1
        spectral_rolloff,         # 1
    ])                            # total: 125

    return features.astype(np.float32)

class VADDataset(Dataset):
    def __init__(self, features_npy: str, labels_npy: str, augment: bool = False):
        """Load precomputed features instead of extracting on-the-fly"""
        self.features = np.load(features_npy).astype(np.float32)  # (N, 121)
        self.labels_str = np.load(labels_npy, allow_pickle=True)   # (N,)
        self.labels = np.array([LABEL_MAP[l] for l in self.labels_str])
        self.augment = augment

    def _augment(self, features: np.ndarray) -> np.ndarray:
        # Gaussian noise — raise probability to 40%
        if np.random.rand() < 0.4:
            noise = np.random.randn(*features.shape).astype(np.float32) * 0.02
            features = features + noise

        # SpecAugment-style: zero out a band of MFCC coefficients (indices 0–39)
        if np.random.rand() < 0.35:
            start = np.random.randint(0, 35)
            width = np.random.randint(1, 6)
            features[start:start + width] = 0.0

        # Scale jitter: simulate volume variation
        if np.random.rand() < 0.3:
            features = features * np.random.uniform(0.85, 1.15)

        return features.astype(np.float32)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        if self.augment:
            features = self._augment(features)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)