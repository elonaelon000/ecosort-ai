from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

from .config import RAW_DIR, REPORTS_DIR, SUPPORTED_EXTENSIONS


def class_image_paths():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {RAW_DIR}")
    result = {}
    for class_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        result[class_dir.name] = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return result


def main():
    paths = class_image_paths()
    counts = Counter({name: len(items) for name, items in paths.items()})
    if not counts or sum(counts.values()) == 0:
        raise ValueError("No images found. Run `python scripts/download_trashnet.py` first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Class distribution:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    fig = plt.figure(figsize=(9, 5))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("EcoSort AI - Raw Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    names = [name for name, items in paths.items() if items]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, name in zip(axes.flat, names[:6]):
        with Image.open(paths[name][0]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(name.title())
        ax.axis("off")
    for ax in axes.flat[len(names[:6]):]:
        ax.axis("off")
    fig.suptitle("EcoSort AI - One Example per Class")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "sample_grid.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
