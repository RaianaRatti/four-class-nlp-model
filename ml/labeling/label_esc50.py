# IMPORTS
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

from config import SAMPLE_RATE, FRAME_MS

# GLOBAL VARIABLES
FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

# DIRECTORIES
ESC50_AUDIO_DIR = "train_data/audio/ESC-50-master/audio"
ESC50_CSV       = "train_data/audio/ESC-50-master/meta/esc50.csv"
OUTPUT_CSV      = "train_data/labels/esc50_labels.csv"

# map ESC-50 category names to your VAD label names
CATEGORY_MAP = {
    "laughing":  "vocalization",
    "coughing":  "vocalization",
}

# FUNCTIONS

# Loads a wav file and slices into 30ms int 16 frames
def extract_frames(wav_path: Path) -> list[np.ndarray]:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    audio_int16 = (audio * 32768).astype(np.int16)

    frames = []
    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i:i + FRAME_SIZE]
        if len(frame) == FRAME_SIZE:
            frames.append(frame)
    return frames


def run():
    max_clips = 50
    df_meta = pd.read_csv(ESC50_CSV)

    # only keeps categories we care about 
    df_meta = df_meta[df_meta["category"].isin(CATEGORY_MAP.keys())]
    df_meta = df_meta.head(max_clips)
    
    print(f"Found {len(df_meta)} clips across categories: {df_meta['category'].value_counts().to_dict()}")

    rows = []
    audio_dir = Path(ESC50_AUDIO_DIR)

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        wav_path = audio_dir / row["filename"]
        if not wav_path.exists():
            print(f"  Missing: {wav_path}")
            continue

        label = CATEGORY_MAP[row["category"]]
        frames = extract_frames(wav_path)

        for i, frame in enumerate(frames):
            start_ms = i * FRAME_MS
            end_ms   = start_ms + FRAME_MS
            rows.append({
                "filename": f"ESC-50-master/audio/{row['filename']}",
                "start_ms": start_ms,
                "end_ms":   end_ms,
                "label":    label
            })

    df_out = pd.DataFrame(rows)

    print(f"\nLabel counts:")
    print(df_out["label"].value_counts())

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df_out)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()