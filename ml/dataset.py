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
    """Extract MFCC + delta + delta-delta + energy from one frame"""
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32) / 32768.0

    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC, n_fft=480, hop_length=160)
    delta = librosa.feature.delta(mfcc, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    energy = np.array([[np.log(np.sum(frame ** 2) + 1e-8)]])

    features = np.concatenate([
        mfcc.mean(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
        energy.flatten()
    ])

    return features.astype(np.float32)


class VADDataset(Dataset):
    def __init__(self, features_npy: str, labels_npy: str, augment: bool = False):
        """Load precomputed features instead of extracting on-the-fly"""
        self.features = np.load(features_npy).astype(np.float32)  # (N, 121)
        self.labels_str = np.load(labels_npy, allow_pickle=True)   # (N,)
        self.labels = np.array([LABEL_MAP[l] for l in self.labels_str])
        self.augment = augment

    def _augment(self, features: np.ndarray) -> np.ndarray:
        # Augment features instead of raw audio (simpler)
        if np.random.rand() < 0.1:
            noise = np.random.randn(*features.shape).astype(np.float32) * 0.02
            features = np.clip(features + noise, -1, 1).astype(np.float32)
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx].copy()
        if self.augment:
            features = self._augment(features)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)