import librosa
import numpy as np
import pandas as pd
import webrtcvad
from pathlib import Path
import sys
from tqdm import tqdm
import soundfile as sf
from config import SAMPLE_RATE, FRAME_MS

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

FRAME_SIZE    = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

LIBRISPEECH_DIR = "train_data/audio/librispeech"
OUTPUT_CSV      = "train_data/labels/librispeech_labels.csv"

vad = webrtcvad.Vad(0)  # aggressiveness 0 — good for clean audio

# --------------------------------------------------------------------------------

def label_file(wav_path: Path) -> list[dict]:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True) # [-1,1]
    audio_int16 = (audio * 32768).astype(np.int16)

    rows = []
    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i : i + FRAME_SIZE]

        # size check
        if len(frame) != FRAME_SIZE:
            continue

        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        label = "speech" if is_speech else "silence"

        rows.append({
            "filename": f"librispeech_flat/{wav_path.name}",
            "start_ms": int(i / SAMPLE_RATE * 1000),
            "end_ms":   int((i + FRAME_SIZE) / SAMPLE_RATE * 1000),
            "label":    label
        })

    return rows


def run():
    # only process ~2 hours
    wav_files = list(Path(LIBRISPEECH_DIR).rglob("*.flac"))
    wav_files = wav_files[:15]

    # copy flac files flat into a single folder so dataset.py can find them
    flat_dir = Path("train_data/audio/librispeech_flat")
    flat_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # copy to flat dir as .wav
    for wav_path in tqdm(wav_files): # progress bar
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        flat_path = flat_dir / (wav_path.stem + ".wav")

        if not flat_path.exists():
            sf.write(flat_path, audio, SAMPLE_RATE)

        rows = label_file(flat_path) # rows.= [("filename", "start_ms", "end_ms", "label")] -> label = silence / speech
        all_rows.extend(rows) # Ex: ([1,2]).extend([3,4]) = [1,2,3,4]

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)

    # cap silence frames so they don't dwarf speech frames - no change if silence_count < speech_count
    speech_count  = (df["label"] == "speech").sum()
    silence_count = (df["label"] == "silence").sum()

    silence_df = df[df["label"] == "silence"].sample(
        n=min(silence_count, speech_count), random_state=42
    )
    speech_df  = df[df["label"] == "speech"]
    df_balanced = pd.concat([speech_df, silence_df]).sample(frac=1, random_state=42)

    df_balanced.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    run()