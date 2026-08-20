"""Run the complete reproducible baseline workflow."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import subprocess


def run(*args: str) -> None:
    print("\n$", " ".join(args))
    subprocess.run(args, check=True)


def main():
    python = sys.executable
    run(python, "scripts/download_trashnet.py")
    run(python, "scripts/prepare_dataset.py")
    run(python, "-m", "src.ecosort.eda")
    run(python, "-m", "src.ecosort.train", "--architecture", "baseline")
    run(python, "-m", "src.ecosort.evaluate")
    print("\nEcoSort AI baseline pipeline completed. Run `python app.py` for the demo.")


if __name__ == "__main__":
    main()
