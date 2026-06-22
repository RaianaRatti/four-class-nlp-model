# Combines all three CSV files into one CSV files ready for training

import pandas as pd
from pathlib import Path

INPUT_CSVS = [
    "train_data/labels/librispeech_labels.csv",
    "train_data/labels/ami_labels.csv",
    "train_data/labels/esc50_labels.csv"
]

OUTPUT_CSV = "train_data/labels/all_labels.csv"

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

    combined = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop = True)

    print(f"\nFinal combined label counts:")
    print(combined["label"].value_counts())
    print(f"Total: {len(combined)} frames")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()