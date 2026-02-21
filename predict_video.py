"""
predict_video.py
Classify a single video using trained CNN+RNN model.

Usage:
    python predict_video.py --video path/to/video.avi
"""

import argparse
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from model import build_model
from dataloader import ACTIONS


# ───────────────────────── CONFIG ─────────────────────────
NUM_FRAMES = 16
IMG_SIZE = (112, 112)
CKPT_PATH = "checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────


def load_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Failed to read video.")

    # Uniform frame sampling
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = [frames[i] for i in indices]

    processed = []
    for frame in sampled:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC → CHW
        processed.append(frame)

    tensor = torch.tensor(processed)  # (T, C, H, W)
    return tensor


def main(video_path):
    device = torch.device(DEVICE)

    print(f"\nLoading model on {device}...")
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    print(f"Reading video: {video_path}")
    frames = load_video_frames(video_path, NUM_FRAMES)
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        logits = model(frames)
        probs = F.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()

    predicted_class = ACTIONS[pred_idx]
    confidence = probs[0][pred_idx].item() * 100

    print("\n==============================")
    print(f"Predicted Action : {predicted_class}")
    print(f"Confidence       : {confidence:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    main(Path(args.video))