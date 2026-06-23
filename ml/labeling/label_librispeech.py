import io
import librosa
import numpy as np
import pandas as pd
import webrtcvad
from pathlib import Path
import sys
from tqdm import tqdm
import soundfile as sf
from scipy.signal import fftconvolve
from config import SAMPLE_RATE, FRAME_MS

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

FRAME_SIZE      = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
LIBRISPEECH_DIR = "train_data/audio/librispeech_flat"
OUTPUT_CSV      = "train_data/labels/librispeech_labels.csv"

vad = webrtcvad.Vad(0)  # aggressiveness 0 — good for clean audio


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def simulate_codec(audio: np.ndarray, sr: int) -> np.ndarray:
    """Round-trip through OGG Vorbis to create lossy compression artifacts."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="OGG", subtype="VORBIS")
    buf.seek(0)
    out, _ = sf.read(buf, dtype="float32")
    n = len(audio)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]


def add_reverb(audio: np.ndarray, decay: float = 0.3, delay_samples: int = 800) -> np.ndarray:
    ir        = np.zeros(delay_samples + 1, dtype=np.float32)
    ir[0]     = 1.0
    ir[-1]    = decay
    convolved = fftconvolve(audio, ir)[:len(audio)]
    return np.clip(convolved, -1.0, 1.0).astype(np.float32)


def get_frame_labels(audio: np.ndarray) -> list[tuple[int, int, str]]:
    """Run WebRTC VAD on clean audio; return (start_ms, end_ms, label) per frame."""
    audio_int16 = (audio * 32768).astype(np.int16)
    frames = []
    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i : i + FRAME_SIZE]
        if len(frame) != FRAME_SIZE:
            continue
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        label     = "speech" if is_speech else "silence"
        start_ms  = int(i / SAMPLE_RATE * 1000)
        end_ms    = int((i + FRAME_SIZE) / SAMPLE_RATE * 1000)
        frames.append((start_ms, end_ms, label))
    return frames


def make_rows(filename: str, frame_labels: list[tuple]) -> list[dict]:
    return [
        {"filename": filename, "start_ms": s, "end_ms": e, "label": l}
        for s, e, l in frame_labels
    ]


def run():
    wav_files = list(Path(LIBRISPEECH_DIR).glob("*.wav"))

    all_rows = []

    for wav_path in tqdm(wav_files):
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        stem      = wav_path.stem

        # VAD runs on clean audio only — augmented variants share the same labels
        frame_labels = get_frame_labels(audio)

        # Original already exists in librispeech_flat/
        all_rows.extend(make_rows(f"librispeech_flat/{stem}.wav", frame_labels))

        augmented_variants = [
            (add_noise(audio, snr_db=15),        f"librispeech_flat_aug/{stem}_noise15.wav"),
            (add_noise(audio, snr_db=5),         f"librispeech_flat_aug/{stem}_noise5.wav"),
            (simulate_codec(audio, SAMPLE_RATE), f"librispeech_flat_aug/{stem}_codec.wav"),
            (add_reverb(audio),                  f"librispeech_flat_aug/{stem}_reverb.wav"),
        ]

        for aug_audio, csv_filename in augmented_variants:
            aug_path = Path("train_data/audio") / csv_filename
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            if not aug_path.exists():
                sf.write(str(aug_path), aug_audio, SAMPLE_RATE)
            all_rows.extend(make_rows(csv_filename, frame_labels))

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)

    # Cap silence so it doesn't dwarf speech frames
    speech_count  = (df["label"] == "speech").sum()
    silence_count = (df["label"] == "silence").sum()

    silence_df  = df[df["label"] == "silence"].sample(
        n=min(silence_count, speech_count), random_state=42
    )
    speech_df   = df[df["label"] == "speech"]
    df_balanced = pd.concat([speech_df, silence_df]).sample(frac=1, random_state=42)

    df_balanced.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    run()
