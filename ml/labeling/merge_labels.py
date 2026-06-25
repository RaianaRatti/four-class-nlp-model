# Combines all three CSV files into one CSV files ready for training

import pandas as pd
from pathlib import Path

INPUT_CSVS = [
    "training/train_data/labels/librispeech_labels.csv",
    "training/train_data/labels/ami_labels.csv",
]

OUTPUT_CSV = "training/train_data/labels/all_labels.csv"

# Target class distribution for training
TARGET_FRACTIONS = {
    "silence":       0.25,
    "speech":        0.40,
    "overlap":       0.20,
    "non-vocal":  0.15,
}

def run():
    dfs = []
    for csv_path in INPUT_CSVS:
        p = Path(csv_path)
        if not p.exists():
            print(f"WARNING: missing {csv_path} - skipping")
            continue
        df = pd.read_csv(p)
        print(f"{p.name}: {len(df)} rows - {df['label'].value_counts().to_dict()}")
        dfs.append(df)

    combined = pd.concat(dfs).reset_index(drop=True)

    print(f"\nRaw combined label counts:")
    print(combined["label"].value_counts())
    print(f"Total: {len(combined)} frames")

    total_target = 50_000

    balanced_parts = []
    for label, frac in TARGET_FRACTIONS.items():
        n_target = int(total_target * frac)
        subset = combined[combined["label"] == label]
        replace = len(subset) < n_target
        sampled = subset.sample(n=n_target, replace=replace, random_state=42)
        balanced_parts.append(sampled)
        status = "oversampled" if replace else "downsampled"
        print(f"  {label:<14}: {len(subset):>6} → {n_target} ({status})")

    combined = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal balanced label counts:")
    print(combined["label"].value_counts())
    print(f"Total: {len(combined)} frames")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()