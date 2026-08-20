from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .config import (
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    PATIENCE,
    REPORTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from .data import load_datasets
from .model import build_model


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += images.size(0)

    return total_loss / total_examples, total_correct / total_examples


def save_history(history: dict[str, list[float]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EcoSort AI Training History")
    plt.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "training_history.png", dpi=160)
    plt.close(fig)


def train(architecture: str = "baseline", pretrained: bool = False, epochs: int = DEFAULT_EPOCHS):
    seed_everything()
    train_loader, val_loader, _, class_names = load_datasets()
    device = choose_device()
    model = build_model(len(class_names), architecture=architecture, pretrained=pretrained).to(device)

    class_weights = compute_class_weights(train_loader.dataset, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=2
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    stale_epochs = 0

    print(f"Device: {device}")
    print(f"Architecture: {architecture} | pretrained={pretrained}")
    print(f"Classes: {class_names}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "architecture": architecture,
                    "pretrained": pretrained,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epochs_completed": epoch,
                },
                MODEL_PATH,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    save_history(history)
    (REPORTS_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint: {MODEL_PATH}")
    return MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train EcoSort AI")
    parser.add_argument("--architecture", choices=["baseline", "mobilenet"], default="baseline")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights (mobilenet only)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    if args.pretrained and args.architecture != "mobilenet":
        parser.error("--pretrained is only valid with --architecture mobilenet")
    train(args.architecture, args.pretrained, args.epochs)


if __name__ == "__main__":
    main()
