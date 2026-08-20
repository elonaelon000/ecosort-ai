from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from .config import MODEL_PATH, REPORTS_DIR
from .data import load_datasets
from .model import build_model


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it before evaluation.")

    _, _, test_loader, class_names = load_datasets()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint classes do not match the prepared dataset.")

    model = build_model(
        len(class_names),
        architecture=checkpoint["architecture"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "architecture": checkpoint["architecture"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_test_examples": len(y_true),
        "class_names": class_names,
        "per_class": {name: report_dict[name] for name in class_names},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, xticks_rotation=45, colorbar=False, ax=ax
    )
    ax.set_title("EcoSort AI - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    print("\n" + report_text)


if __name__ == "__main__":
    main()
