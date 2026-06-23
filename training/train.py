import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_CSV       = "train_data/labels/all_labels.csv"
AUDIO_DIR       = "train_data/audio"
MODEL_OUT       = "models/custom_vad.pt"
CHECKPOINT_PATH = "models/checkpoint_latest.pt"
EPOCHS          = 60
BATCH_SIZE      = 256
LR              = 1e-3
VAL_SPLIT       = 0.15

CLASS_NAMES = ["silence", "speech", "overlap", "vocalization"]


def make_split_loaders(features_path, labels_path, val_split, batch_size):
    full_train = VADDataset(features_path, labels_path, augment=True)
    full_val   = VADDataset(features_path, labels_path, augment=False)

    n         = len(full_train)
    val_size  = int(n * val_split)
    indices   = torch.randperm(n).tolist()
    val_idx   = indices[:val_size]
    train_idx = indices[val_size:]

    num_workers = 0 if sys.platform == "darwin" else 2

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train():
    print("Creating dataset splits...")
    train_loader, val_loader = make_split_loaders(
        "preprocessed_features/features.npy",
        "preprocessed_features/labels.npy",
        VAL_SPLIT,
        BATCH_SIZE,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = VADNet().to(device)

    # [silence, speech, overlap, vocalization]
    weights   = torch.tensor([1.0, 1.0, 1.3, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_bal_acc = 0.0
    start_epoch  = 0

    # Resume from checkpoint if available and run is incomplete
    if Path(CHECKPOINT_PATH).exists():
        ckpt        = torch.load(CHECKPOINT_PATH, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        if start_epoch >= EPOCHS:
            print(f"Checkpoint is from a completed run (epoch {ckpt['epoch']}) — starting fresh")
            Path(CHECKPOINT_PATH).unlink()
            start_epoch = 0
        else:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            best_bal_acc = ckpt.get("best_bal_acc", 0.0)
            print(f"Resumed from epoch {start_epoch} (best bal_acc: {best_bal_acc:.1f}%)")
    else:
        print("No checkpoint found — starting from scratch")

    print("Starting training loop...")
    for epoch in range(start_epoch, EPOCHS):

        # ── Training ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixup: blend pairs of samples; forces smoother decision boundaries
            lam = max(float(np.random.beta(0.1, 0.1)), 0.7)
            idx = torch.randperm(features.size(0), device=device)
            mixed = lam * features + (1 - lam) * features[idx]
            logits = model(mixed)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss      = 0.0
        class_correct = torch.zeros(4, device=device)
        class_total   = torch.zeros(4, device=device)

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits    = model(features)
                val_loss += criterion(logits, labels).item()
                preds     = logits.argmax(dim=1)

                for c in range(4):
                    mask = labels == c
                    class_correct[c] += (preds[mask] == c).sum()
                    class_total[c]   += mask.sum()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        overall   = class_correct.sum() / class_total.sum() * 100
        # Mean per-class recall — penalises ignoring any single class
        bal_acc   = (class_correct / class_total.clamp(min=1)).mean().item() * 100

        print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f} | acc={overall:.1f}% | bal_acc={bal_acc:.1f}%")
        for c in range(4):
            if class_total[c] > 0:
                pct = class_correct[c] / class_total[c] * 100
                print(f"  {CLASS_NAMES[c]:<14} {pct:.1f}%")

        scheduler.step(avg_val)

        # ── Save latest checkpoint every epoch ──────────────────────────────
        Path(CHECKPOINT_PATH).parent.mkdir(exist_ok=True)
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_bal_acc": best_bal_acc,
        }, CHECKPOINT_PATH)

        # ── Save best model on balanced accuracy, not val loss ───────────────
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            Path(MODEL_OUT).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✓ saved best model → {MODEL_OUT}  (bal_acc={bal_acc:.1f}%)")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
