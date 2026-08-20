"""Download and verify the resized TrashNet dataset.

The data is downloaded from the current Hugging Face mirror of the original
TrashNet project. The archive checksum comes from the source file metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile

from src.ecosort.config import EXPECTED_CLASSES, RAW_DIR, ROOT_DIR

URLS = [
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true",
    "https://github.com/garythung/trashnet/raw/refs/heads/master/data/dataset-resized.zip",
]
SHA256 = "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c"
ARCHIVE = ROOT_DIR / "data" / "trashnet-dataset-resized.zip"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(urls=URLS, destination: Path = ARCHIVE) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        try:
            print(f"Downloading TrashNet from: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "EcoSort-AI/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256sum(destination) != SHA256:
                destination.unlink(missing_ok=True)
                raise ValueError("Downloaded archive checksum did not match the published SHA-256.")
            return destination
        except Exception as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            print(f"Source failed: {exc}")
    raise RuntimeError(f"Could not download TrashNet from the configured sources: {last_error}")


def extract_to_raw(archive: Path = ARCHIVE, raw_dir: Path = RAW_DIR) -> None:
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)
        candidates = [tmp_path / "dataset-resized", tmp_path]
        source_root = next(
            (p for p in candidates if all((p / c).exists() for c in EXPECTED_CLASSES)),
            None,
        )
        if source_root is None:
            raise ValueError("Archive did not contain the expected TrashNet class folders.")
        for class_name in EXPECTED_CLASSES:
            shutil.copytree(source_root / class_name, raw_dir / class_name)


def main():
    parser = argparse.ArgumentParser(description="Download the resized TrashNet dataset")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    existing = [p for p in RAW_DIR.iterdir() if p.is_dir()] if RAW_DIR.exists() else []
    if existing and not args.force:
        print("data/raw already contains class folders. Use --force to replace them.")
        return

    archive = download()
    print("Checksum verified.")
    extract_to_raw(archive)
    print(f"Dataset extracted to: {RAW_DIR}")
    if not args.keep_archive:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
