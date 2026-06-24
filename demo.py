# Use: python demo.py {filename}
import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import collections

from ml.model import VADNet
from ml.dataset import extract_features
from config import CONTEXT_FRAMES

# Class colors for visualization
CLASS_COLORS = {
    0: "#1f77b4",  # blue: silence
    1: "#2ca02c",  # green: speech
    2: "#d62728",  # red: overlap
    3: "#ff7f0e",  # orange: vocalization
}

CLASS_NAMES = ["silence", "speech", "overlap", "vocalization"]


def load_model(model_path, device):
    """Load VADNet checkpoint from disk."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = VADNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both raw state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def run_inference(audio_path, model, device, sample_rate,
                  mean_path="preprocessed_features/mean.npy",
                  std_path="preprocessed_features/std.npy"):
    """
    Load audio and run inference on all frames.

    Returns:
        predictions: (num_frames,) array of class indices
        confidences: (num_frames, 4) array of softmax probabilities
        frame_times: (num_frames,) array of frame start times in seconds
    """
    # Load normalization stats (must match training)
    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)

    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Frame parameters (same as training)
    frame_length = int(0.03 * sample_rate)  # 30ms
    hop_length = frame_length  # non-overlapping frames
    
    # Extract features for all frames
    predictions = []
    confidences = []
    frame_times = []
    
    print(f"Processing {len(y) / sample_rate:.2f}s of audio ({len(y)} samples)...")

    context = collections.deque(
        [np.zeros(128, dtype=np.float32)] * CONTEXT_FRAMES,
        maxlen=CONTEXT_FRAMES
    )
    
    with torch.no_grad():
        for start in range(0, len(y) - frame_length, hop_length):
            frame = y[start:start + frame_length]

            if np.sqrt(np.mean(frame ** 2)) < 0.001:
                context.append(np.zeros(128, dtype=np.float32))
                predictions.append(0)
                confidences.append(np.array([1.0, 0.0, 0.0, 0.0]))
                frame_times.append(start / sample_rate)
                continue

            features = (extract_features(frame, sr=sample_rate) - mean) / std
            context.append(features)

            # Stack context window into sequence tensor
            seq = np.stack(list(context))                              # (CONTEXT_FRAMES, 128)
            seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)    # (1, CONTEXT_FRAMES, 128)

            logits = model(seq_tensor)    # (1, CONTEXT_FRAMES, NUM_CLASSES)
            last   = logits[0, -1, :]     # classify the current frame
            probs  = torch.softmax(last, dim=0).cpu().numpy()
            pred   = last.argmax().item()

            predictions.append(pred)
            confidences.append(probs)
            frame_times.append(start / sample_rate)

        return (
            np.array(predictions),
            np.array(confidences),
            np.array(frame_times),
        )


def print_predictions(predictions, confidences, frame_times, class_names):
    """Print frame-by-frame predictions to console with confidence scores."""
    print("\nFrame-by-frame predictions:")
    print("-" * 80)
    print(f"{'Time':>8s} | {'Predicted':>12s} | Confidence per class")
    print("-" * 80)
    
    # Show every Nth frame to avoid spam (or first/last if fewer than 10 frames)
    stride = max(1, len(predictions) // 20)
    
    for i in range(0, len(predictions), stride):
        pred_class = predictions[i]
        pred_name = class_names[pred_class]
        conf = confidences[i]
        time_str = f"{frame_times[i]:.2f}s"
        conf_str = " | ".join([f"{class_names[j]}: {conf[j]:.2f}" for j in range(len(class_names))])
        
        print(f"{time_str:>8s} | {pred_name:>12s} | {conf_str}")
    
    print("-" * 80)


def create_visualization(predictions, confidences, frame_times, class_names, output_path):
    """Create and save a timeline visualization of predictions."""
    num_frames = len(predictions)
    duration = frame_times[-1] if len(frame_times) > 0 else 1.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [1, 3]})
    
    # Top plot: Stacked confidence scores
    for class_idx in range(len(class_names)):
        ax1.fill_between(
            frame_times,
            confidences[:, class_idx],
            alpha=0.7,
            label=class_names[class_idx],
            color=CLASS_COLORS[class_idx]
        )
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Confidence")
    ax1.set_title("Per-class confidence over time")
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper right", ncol=4)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Predicted class timeline
    colors = [CLASS_COLORS[p] for p in predictions]
    ax2.bar(frame_times, height=1, width=frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.03, 
            color=colors, edgecolor="none", alpha=0.8)
    
    # Add class labels on the left
    for class_idx, class_name in enumerate(class_names):
        ax2.text(-0.15, 0.5, class_name.capitalize(), transform=ax2.get_yaxis_transform(), 
                 fontsize=10, fontweight="bold", va="center")
    
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, duration])
    ax2.set_ylim([0, 1])
    ax2.set_yticks([])
    ax2.set_title("Predicted class timeline")
    ax2.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    plt.close()


def compute_statistics(predictions, confidences, class_names):
    """Compute and print high-level statistics."""
    total_frames = len(predictions)
    class_counts = np.bincount(predictions, minlength=len(class_names))
    
    print("\nPrediction statistics:")
    print("-" * 60)
    for class_idx, class_name in enumerate(class_names):
        count = class_counts[class_idx]
        percentage = 100 * count / total_frames
        avg_conf = confidences[predictions == class_idx, class_idx].mean() if count > 0 else 0
        print(f"{class_name.capitalize():12s}: {count:5d} frames ({percentage:5.1f}%) | avg confidence: {avg_conf:.3f}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run VADNet inference on audio and visualize predictions."
    )
    parser.add_argument("audio", type=str, help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--model", type=str, default="models/custom_vad.pt", 
                        help="Path to model checkpoint (default: models/custom_vad.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save visualization PNG (default: audio_name_prediction.png)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate for audio processing (default: 16000)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization, print predictions only")
    
    args = parser.parse_args()
    
    # Validate inputs
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        print(f"Have you trained a model yet? See README for training instructions.", file=sys.stderr)
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        output_dir = Path("testing/test_results")
        output_dir.mkdir(exist_ok=True)

        args.output = str(
            output_dir / f"{audio_path.stem}_prediction.png"
        )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    # Run inference
    print(f"Loading audio from {audio_path}...")
    predictions, confidences, frame_times = run_inference(
        str(audio_path), model, device, args.sample_rate
    )
    
    # Print results
    print_predictions(predictions, confidences, frame_times, CLASS_NAMES)
    compute_statistics(predictions, confidences, CLASS_NAMES)
    
    # Visualize
    if not args.no_viz:
        create_visualization(predictions, confidences, frame_times, CLASS_NAMES, args.output)


if __name__ == "__main__":
    main()
