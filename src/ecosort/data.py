from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, IMAGE_SIZE, SEED, TEST_DIR, TRAIN_DIR, VAL_DIR


def _ensure_directory(path: Path) -> None:
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"Dataset directory is missing or empty: {path}. "
            "Run `python scripts/prepare_dataset.py` first."
        )


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_datasets(batch_size: int = BATCH_SIZE):
    for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        _ensure_directory(directory)

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform())
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_transform())
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transform())

    if val_ds.classes != train_ds.classes or test_ds.classes != train_ds.classes:
        raise ValueError("Train, validation, and test sets do not have the same classes.")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, train_ds.classes
