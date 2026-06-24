import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split

from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_NAMES = ["silence", "speech", "overlap", "vocalization"]

DEFAULT_MODEL = "models/custom_vad.pt"
DEFAULT_FEATURES = "preprocessed_features/features.npy"
DEFAULT_LABELS = "preprocessed_features/labels.npy"


def load_model(model_path: str, device: torch.device):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model = VADNet().to(device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a VAD model on precomputed features."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Path to model weights."
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="Path to feature numpy file."
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Path to label numpy file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction."
    )

    args = parser.parse_args()

    dataset = VADDataset(args.features, args.labels, augment=False)

    val_size = int(len(dataset) * args.val_split)

    _, val_ds = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(args.model_path, device)
    model.eval()

    class_correct = [0] * len(LABEL_NAMES)
    class_total = [0] * len(LABEL_NAMES)

    total_correct = 0
    total_samples = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            if logits.ndim == 1:
                raise ValueError(
                    "Model output must have shape (batch, num_classes)"
                )

            if logits.shape[1] != len(LABEL_NAMES):
                raise ValueError(
                    f"Model output dimension {logits.shape[1]} "
                    f"does not match expected number of classes "
                    f"{len(LABEL_NAMES)}."
                )

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for cls in range(len(LABEL_NAMES)):
                mask = labels == cls
                class_total[cls] += mask.sum().item()
                class_correct[cls] += (preds[mask] == cls).sum().item()

    print(f"Overall accuracy = {total_correct / total_samples * 100:.2f}%")

    for cls_idx, name in enumerate(LABEL_NAMES):
        if class_total[cls_idx] == 0:
            continue

        accuracy = (
            class_correct[cls_idx]
            / class_total[cls_idx]
            * 100
        )

        print(
            f"  {name:12s}: "
            f"{accuracy:6.2f}% "
            f"({class_correct[cls_idx]}/{class_total[cls_idx]})"
        )

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(colorbar=False)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("✓ Saved confusion_matrix.png")


if __name__ == "__main__":
    main()