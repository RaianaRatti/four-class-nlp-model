# precompute_features.py
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from ml.dataset import extract_features
import librosa
from config import SAMPLE_RATE

LABEL_CSV = "training/train_data/labels/all_labels.csv"
AUDIO_DIR = "training/train_data/audio"
OUTPUT_DIR = "preprocessed_features"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

df = pd.read_csv(LABEL_CSV)
features_list = []
labels_list = []

print(f"Processing {len(df)} samples...")

for i, row in df.iterrows():
    if i % 1000 == 0:
        print(f"  {i}/{len(df)}...")
    
    audio_path = Path(AUDIO_DIR) / row["filename"]
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = (audio * 32768).astype(np.int16)
    
    start = int(row["start_ms"] * SAMPLE_RATE / 1000)
    end = int(row["end_ms"] * SAMPLE_RATE / 1000)
    frame = audio[start:end]
    
    if len(frame) != 480:  # FRAME_SIZE
        continue
    
    features = extract_features(frame)
    features_list.append(features)
    labels_list.append(row["label"])

# Saving as single numpy file
features_array = np.array(features_list, dtype=np.float32)
labels_array = np.array(labels_list, dtype=str)

np.save(f"{OUTPUT_DIR}/features.npy", features_array)
np.save(f"{OUTPUT_DIR}/labels.npy", labels_array)

mean = features_array.mean(axis=0)
std  = features_array.std(axis=0) + 1e-8
np.save(f"{OUTPUT_DIR}/mean.npy", mean)
np.save(f"{OUTPUT_DIR}/std.npy", std)

print(f"✓ Saved {len(features_list)} features")
print(f"✓ Saved normalization stats (mean/std)")