# Data Directory

## Automatic setup

Run:

```bash
python scripts/download_trashnet.py
```

The script downloads the 42.8 MB resized TrashNet archive from the current Hugging Face mirror, verifies its SHA-256 checksum, and copies the six class folders into `data/raw/`.

Then run:

```bash
python scripts/prepare_dataset.py
```

## Expected raw layout

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Processed layout

```text
data/processed/
├── train/
├── val/
└── test/
```

The split is deterministic by default (seed 42). The test set should remain untouched during model development.

## Attribution

See `../DATA_SOURCES.md`. Do not remove dataset attribution when publishing the project.
