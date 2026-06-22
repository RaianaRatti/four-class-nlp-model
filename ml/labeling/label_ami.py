import xml.etree.ElementTree as ET
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import SAMPLE_RATE, FRAME_MS

FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

# paths relative to project root — adjust if your ami folder is elsewhere
AMI_AUDIO_DIR = "train_data/audio/ami/amicorpus"
AMI_WORDS_DIR = "train_data/audio/ami/ami_public_manual_1.6.2/words"
OUTPUT_CSV    = "train_data/labels/ami_labels.csv"

# only process the meetings we actually downloaded audio for
MEETINGS = [
    "ES2008a", "ES2008b", "ES2008c", "ES2008d",
    "ES2009a", "ES2009b", "ES2009c", "ES2009d",
    "ES2010a", "ES2010b", "ES2010c", "ES2010d",
]
SPEAKERS = ["A", "B", "C", "D"]


def parse_words_xml(xml_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Returns:
        speech_segments    — list of (start_sec, end_sec) for words
        vocal_segments     — list of (start_sec, end_sec) for vocalizations
    """
    tree = ET.parse(xml_path) # stores parsed XML as tree
    root = tree.getroot() # gets topmost node

    speech_segments = []
    vocal_segments  = []

    for elem in root: # loops through every direct child element inside XML root
        tag = elem.tag.split("}")[-1]  # strips namespace if present (Ex: {namespace}tagName -> tagName)
        start = elem.get("starttime")
        end   = elem.get("endtime")

        if start is None or end is None: # check if either attribute is missing (then ignore)
            continue

        start, end = float(start), float(end)

        if end <= start:  # check if segment duration is 0 or negative (then ignore)
            continue

        if tag == "w": # check if XML element represents a word
            speech_segments.append((start, end))
        elif tag == "vocalsound": # check if XML element represents a sound
            vocal_segments.append((start, end))

    return speech_segments, vocal_segments


def label_meeting(meeting_id: str) -> list[dict]:
    """
    For a given meeting (e.g. ES2008a), load all 4 speaker XML files,
    build a per-frame label array over the full recording duration,
    then extract features from the MIXED headset audio.

    Label priority per frame:
        overlap       — 2+ speakers active simultaneously
        vocalization  — 1 speaker has a vocalsound, none have a word
        speech        — exactly 1 speaker active
        silence       — no speaker active
    """

    # --- load all speaker segments ---
    all_speech = []   # list of (start, end) across all speakers
    all_vocal  = []

    for speaker in SPEAKERS: # [A, B, C, D]
        xml_path = Path(AMI_WORDS_DIR) / f"{meeting_id}.{speaker}.words.xml"
        if not xml_path.exists():
            continue
        speech_segs, vocal_segs = parse_words_xml(xml_path)
        all_speech.extend(speech_segs)
        all_vocal.extend(vocal_segs)

    if not all_speech and not all_vocal:
        print(f"No annotations found for {meeting_id}, skipping.")
        return []

    # --- find total duration from any available headset wav ---
    audio_path = None
    for h in range(4):
        candidate = Path(AMI_AUDIO_DIR) / meeting_id / "audio" / f"{meeting_id}.Headset-{h}.wav"
        if candidate.exists():
            audio_path = candidate
            break

    if audio_path is None:
        print(f"  No audio found for {meeting_id}, skipping.")
        return []

    # load just enough to get duration — no need to load full file into RAM
    duration_sec = librosa.get_duration(path=str(audio_path))
    total_frames = int(duration_sec * 1000 / FRAME_MS)

    # --- build per-frame active speaker counts ---
    speech_count = np.zeros(total_frames, dtype=np.int16) # how many speakers speaking during each frame
    vocal_count  = np.zeros(total_frames, dtype=np.int16) # how many speakers are vocalizing during each frame

    def mark(segments, counter):
        for start, end in segments:
            f_start = int(start * 1000 / FRAME_MS) # start frame
            f_end = min(int(end * 1000 / FRAME_MS) + 1, total_frames) # end frame
            counter[f_start:f_end] += 1 # increases all frames in that range by 1

    mark(all_speech, speech_count)
    mark(all_vocal,  vocal_count)

    # --- assign label per frame ---
    rows = []
    filename = f"ami/amicorpus/{meeting_id}/audio/{meeting_id}.Headset-0.wav"

    for i in range(total_frames):
        start_ms = i * FRAME_MS
        end_ms = start_ms + FRAME_MS

        sc = speech_count[i]
        vc = vocal_count[i]

        if sc >= 2:
            label = "overlap"
        elif sc == 1:
            label = "speech"
        elif vc >= 1:
            label = "vocalization"
        else:
            label = "silence"

        rows.append({
            "filename": filename,
            "start_ms": start_ms,
            "end_ms":   end_ms,
            "label":    label
        })

    return rows


def run():
    all_rows = []

    for meeting_id in tqdm(MEETINGS, desc="Processing meetings"):
        rows = label_meeting(meeting_id)
        all_rows.extend(rows)
        if rows:
            # converts rows into dataframe, gets "label", counts values of each category, returns them as dict
            counts = pd.DataFrame(rows)["label"].value_counts().to_dict()
            print(f"  {meeting_id}: {counts}")

    if not all_rows:
        print("No rows generated — check your paths.")
        return

    df = pd.DataFrame(all_rows)

    print("\nRaw label counts:")
    print(df["label"].value_counts()) # values of each label category, prints them as dict

    # per-label caps — silence/speech are rare in AMI so keep small;
    # overlap dominates (100k+) so cap it to keep total AMI ~34k
    LABEL_CAPS = {
        "silence": 2_000,
        "speech":  2_000,
        "overlap": 25_000,
        # vocalization: no cap (~5k available, keep all)
    }

    balanced_parts = []

    for label in ["silence", "speech", "overlap", "vocalization"]:
        subset = df[df["label"] == label]
        if label in LABEL_CAPS and len(subset) > LABEL_CAPS[label]:
            subset = subset.sample(n=LABEL_CAPS[label], random_state=42)
        balanced_parts.append(subset)

    df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False) # prevents from adding row numbers as extra column

    print("\nBalanced label counts:")
    print(df_balanced["label"].value_counts())
    print(f"\nSaved {len(df_balanced)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()