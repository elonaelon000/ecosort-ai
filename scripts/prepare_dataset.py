"""Create deterministic train/validation/test folders from data/raw."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import shutil

from src.ecosort.config import PROCESSED_DIR, RAW_DIR, SEED, SUPPORTED_EXTENSIONS


def image_files(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def split_items(items: list[Path], train_ratio: float, val_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def copy_split(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths):
        shutil.copy2(source, destination / f"{index:05d}_{source.name}")


def prepare_dataset(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
):
    class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.exists() else []
    if len(class_dirs) < 2:
        raise ValueError("Add at least two class folders with images under data/raw/.")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) < 7:
            raise ValueError(
                f"Class '{class_dir.name}' has only {len(images)} images. "
                "At least 7 are needed for non-empty 70/15/15 splits."
            )
        train, val, test = split_items(images, train_ratio, val_ratio, seed)
        if not train or not val or not test:
            raise ValueError(f"Class '{class_dir.name}' is too small for the requested split.")
        copy_split(train, processed_dir / "train" / class_dir.name)
        copy_split(val, processed_dir / "val" / class_dir.name)
        copy_split(test, processed_dir / "test" / class_dir.name)
        summary[class_dir.name] = {"train": len(train), "val": len(val), "test": len(test)}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Split EcoSort image data")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    summary = prepare_dataset(train_ratio=args.train, val_ratio=args.val, seed=args.seed)
    print("Prepared dataset:")
    for class_name, counts in summary.items():
        print(f"  {class_name}: {counts}")


if __name__ == "__main__":
    main()
