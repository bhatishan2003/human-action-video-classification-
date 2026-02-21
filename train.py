"""
train.py
Full training pipeline for the Simple CNN + Simple RNN action classifier.
Now includes:
  • tqdm progress bars
  • Training curves plot
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import get_dataloaders, ACTIONS
from model import build_model, count_parameters


# ───────────────────────── CONFIG ─────────────────────────
DEFAULT_CONFIG = dict(
    data_root="data/kth_actions",
    epochs=5,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-4,
    num_frames=16,
    img_size=(112, 112),
    num_workers=4,
    ckpt_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
# ───────────────────────────────────────────────────────────


# ─────────────────────── TRAIN LOOP ────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0

    loop = tqdm(loader, leave=False)

    for frames, labels in loop:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += frames.size(0)

        loop.set_postfix(
            loss=f"{running_loss/total:.4f}",
            acc=f"{100*correct/total:.1f}%"
        )

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, leave=False)

    for frames, labels in loop:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
        loss = criterion(logits, labels)

        running_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += frames.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        loop.set_postfix(
            val_loss=f"{running_loss/total:.4f}",
            val_acc=f"{100*correct/total:.1f}%"
        )

    return running_loss / total, correct / total, all_preds, all_labels


# ───────────────────── TRAINING PLOT ────────────────────────
def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ───────────────────────── MAIN ─────────────────────────────
def main(cfg: dict):
    device = torch.device(cfg["device"])

    print(f"\n{'='*60}")
    print("  KTH Human Action Classification — Simple CNN + Simple RNN")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    print("[1/4] Building dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=Path(cfg["data_root"]),
        num_frames=cfg["num_frames"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    print("[2/4] Building model...")
    model = build_model().to(device)
    print(f"Trainable parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("[3/4] Training...\n")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.1f}%\n"
            f"Val Loss  : {val_loss:.4f} | Val Acc  : {100*val_acc:.1f}%"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print("✓ Best model saved.")

    # ── TEST ────────────────────────────────────────────────
    print("\n[4/4] Testing best model...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Accuracy: {100*test_acc:.2f}%")


    # ── PLOT ────────────────────────────────────────────────
    plot_path = ckpt_dir / "training_curves.png"
    plot_history(history, plot_path)
    print(f"\nTraining curves saved to: {plot_path}")

    print("\nTraining complete.\n")


# ───────────────────────── CLI ──────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--num_frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--ckpt_dir", default=DEFAULT_CONFIG["ckpt_dir"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    return vars(parser.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    cfg["img_size"] = DEFAULT_CONFIG["img_size"]
    main(cfg)