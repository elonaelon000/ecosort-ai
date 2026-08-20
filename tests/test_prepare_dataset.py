from pathlib import Path

from scripts.prepare_dataset import prepare_dataset, split_items


def test_split_is_deterministic():
    items = [Path(f"image_{i}.jpg") for i in range(100)]
    assert split_items(items, 0.7, 0.15, 42) == split_items(items, 0.7, 0.15, 42)


def test_split_sizes():
    items = [Path(f"image_{i}.jpg") for i in range(100)]
    train, val, test = split_items(items, 0.7, 0.15, 42)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
