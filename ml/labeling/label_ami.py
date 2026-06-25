import xml.etree.ElementTree as ET
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import SAMPLE_RATE, FRAME_MS

FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

AMI_AUDIO_DIR = "training/train_data/audio/ami/amicorpus"
AMI_WORDS_DIR = "training/train_data/audio/ami/ami_public_manual_1.6.2/words"
OUTPUT_CSV    = "training/train_data/labels/ami_labels.csv"

MEETINGS = [
    "ES2008a", "ES2008b", "ES2008c", "ES2008d",
    "ES2009a", "ES2009b", "ES2009c", "ES2009d",
    "ES2010a", "ES2010b", "ES2010c", "ES2010d",
]
SPEAKERS = ["A", "B", "C", "D"]

# Minimum gap between consecutive words to NOT bridge them.
# Words within the same speaker turn are often separated by short pauses
# that the XML still covers — we shrink word segments slightly so we don't
# accidentally mark inter-word silence as speech.
WORD_SHRINK_SEC = 0.02  # shave 20ms off each word end to avoid bleed


def parse_words_xml(xml_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Returns:
        speech_segments  — list of (start_sec, end_sec) for individual words
        vocal_segments   — list of (start_sec, end_sec) for vocalizations
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    speech_segments = []
    vocal_segments  = []

    for elem in root:
        tag   = elem.tag.split("}")[-1]
        start = elem.get("starttime")
        end   = elem.get("endtime")

        if start is None or end is None:
            continue

        start, end = float(start), float(end)

        if end <= start:
            continue

        if tag == "w":
            # Shrink slightly so consecutive words in the same turn
            # don't bleed into each other and create false overlap
            shrunk_end = max(start + 0.001, end - WORD_SHRINK_SEC)
            speech_segments.append((start, shrunk_end))
        elif tag == "vocalsound":
            vocal_segments.append((start, end))

    return speech_segments, vocal_segments


def label_meeting(meeting_id: str) -> list[dict]:
    """
    Build per-frame labels for one meeting.

    Label priority:
        overlap      — 2+ speakers have a word active in this frame
        speech       — exactly 1 speaker has a word active
        non-vocal    — 1 speaker or background has a vocalsound, no words active for anyone
        silence      — nothing active
    """

    # --- load per-speaker segments separately so we can count by speaker ---
    # Each entry is a list of (start, end) for that speaker
    speaker_speech = {}   # speaker -> [(start, end), ...]
    speaker_vocal  = {}

    for speaker in SPEAKERS:
        xml_path = Path(AMI_WORDS_DIR) / f"{meeting_id}.{speaker}.words.xml"
        if not xml_path.exists():
            continue
        speech_segs, vocal_segs = parse_words_xml(xml_path)
        if speech_segs:
            speaker_speech[speaker] = speech_segs
        if vocal_segs:
            speaker_vocal[speaker] = vocal_segs

    if not speaker_speech and not speaker_vocal:
        print(f"  No annotations found for {meeting_id}, skipping.")
        return []

    # --- find audio file and duration ---
    audio_path = None
    for h in range(4):
        candidate = Path(AMI_AUDIO_DIR) / meeting_id / "audio" / f"{meeting_id}.Headset-{h}.wav"
        if candidate.exists():
            audio_path = candidate
            break

    if audio_path is None:
        print(f"  No audio found for {meeting_id}, skipping.")
        return []

    duration_sec = librosa.get_duration(path=str(audio_path))
    total_frames = int(duration_sec * 1000 / FRAME_MS)

    # --- build per-SPEAKER frame arrays (not combined) ---
    # This is the key fix: count active speakers per frame, not total word
    # events. Two words from the same speaker in the same frame = 1 active
    # speaker, not 2.
    def build_speaker_activity(segments_by_speaker: dict) -> np.ndarray:
        """Returns (total_frames,) array counting how many distinct speakers
        are active in each frame."""
        active = np.zeros(total_frames, dtype=np.int16)
        for speaker, segs in segments_by_speaker.items():
            # binary mask for this speaker — 1 if they have any word active
            mask = np.zeros(total_frames, dtype=np.int16)
            for start, end in segs:
                f_start = int(start * 1000 / FRAME_MS)
                f_end   = min(int(end * 1000 / FRAME_MS) + 1, total_frames)
                mask[f_start:f_end] = 1
            active += mask  # each speaker contributes at most 1 per frame
        return active

    speech_active = build_speaker_activity(speaker_speech)

    # For vocal: just need to know if any speaker has a vocalsound active
    vocal_active = np.zeros(total_frames, dtype=np.int16)
    for speaker, segs in speaker_vocal.items():
        for start, end in segs:
            f_start = int(start * 1000 / FRAME_MS)
            f_end   = min(int(end * 1000 / FRAME_MS) + 1, total_frames)
            vocal_active[f_start:f_end] = 1

    # --- assign label per frame ---
    rows = []
    filename = f"ami/amicorpus/{meeting_id}/audio/{meeting_id}.Headset-0.wav"

    for i in range(total_frames):
        start_ms = i * FRAME_MS
        end_ms   = start_ms + FRAME_MS
        sc = speech_active[i]
        vc = vocal_active[i]

        if sc >= 2:
            label = "overlap"
        elif sc == 1:
            label = "speech"
        elif vc >= 1:
            label = "non-vocal"
        else:
            label = "silence"

        rows.append({
            "filename": filename,
            "start_ms": start_ms,
            "end_ms":   end_ms,
            "label":    label,
        })

    return rows


def run():
    all_rows = []

    for meeting_id in tqdm(MEETINGS, desc="Processing meetings"):
        rows = label_meeting(meeting_id)
        all_rows.extend(rows)
        if rows:
            counts = pd.DataFrame(rows)["label"].value_counts().to_dict()
            print(f"  {meeting_id}: {counts}")

    if not all_rows:
        print("No rows generated — check your paths.")
        return

    df = pd.DataFrame(all_rows)

    print("\nRaw label counts:")
    print(df["label"].value_counts())

    # Cap each label — overlap still dominates raw counts in meeting data
    # so cap it firmly to keep the class balanced with speech
    LABEL_CAPS = {
        "silence":       3_000,
        "speech":        8_000,
        "overlap":       8_000,   # capped to match speech
        "non-vocal":     3_000,
    }

    balanced_parts = []
    for label in ["silence", "speech", "overlap", "non-vocal"]:
        subset = df[df["label"] == label]
        if len(subset) == 0:
            print(f"  WARNING: no rows for label '{label}'")
            continue
        cap = LABEL_CAPS.get(label)
        if cap and len(subset) > cap:
            subset = subset.sample(n=cap, random_state=42)
            print(f"  {label:<14}: {len(df[df['label']==label]):>6} → {cap} (downsampled)")
        else:
            print(f"  {label:<14}: {len(subset):>6} (kept all)")
        balanced_parts.append(subset)

    df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)

    print("\nFinal label counts:")
    print(df_balanced["label"].value_counts())
    print(f"\nSaved {len(df_balanced)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()