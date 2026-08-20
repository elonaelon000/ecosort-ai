# EcoSort AI

## 🚀 Live Demo

### [Try EcoSort AI in your browser →](https://ecosort-ai-elonagit.streamlit.app)

Upload a photo of a waste item and the live model will classify it as **cardboard, glass, metal, paper, plastic, or trash**, show its confidence, and provide disposal guidance.

**Current model:** MobileNetV3 transfer learning · **90.4% test accuracy** · **0.908 macro F1** on 384 held-out TrashNet images.

---

**EcoSort AI** is an environmental computer-vision project that classifies a photo of a waste item into one of six broad categories and gives a simple disposal suggestion.

The project is intentionally built as a full machine-learning workflow rather than a single notebook: data acquisition, dataset splitting, exploratory analysis, model training, evaluation, error analysis, and an interactive browser demo.

## Problem

People often place waste in the wrong bin because material categories are not always obvious. EcoSort AI explores a narrow question:

> Can a computer-vision model recognize the broad material category of a single waste item from an image?

This is an educational prototype. It is **not** an official recycling authority. Real disposal rules depend on local facilities, contamination, coatings, mixed materials, and regulations that may not be visible from a photo.

## Classes

The first version uses the six TrashNet classes:

- cardboard
- glass
- metal
- paper
- plastic
- trash

## Dataset

The project uses **TrashNet**, created by Gary Thung and Mindy Yang for a Stanford CS 229 project. The original repository describes 2,527 resized images across the six classes and asks dataset users to cite the repository. The repository is MIT licensed.

Dataset source and license notes are recorded in [`DATA_SOURCES.md`](DATA_SOURCES.md).

The dataset itself is **not committed to this repository**. Download it with:

```bash
python scripts/download_trashnet.py
```

## Project structure

```text
ecosort-ai/
├── app.py
├── streamlit_app.py
├── README.md
├── DATA_SOURCES.md
├── MODEL_CARD.md
├── PROJECT_PLAN.md
├── LEARNING_NOTES.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── download_trashnet.py
│   ├── prepare_dataset.py
│   └── run_pipeline.py
├── src/
│   └── ecosort/
│       ├── config.py
│       ├── data.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── guidance.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
└── tests/
```

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Option A: run the complete baseline pipeline

```bash
python scripts/run_pipeline.py
```

That command downloads TrashNet, creates deterministic train/validation/test splits, generates EDA, trains the baseline CNN, and evaluates it on the untouched test set.

### Option B: run one step at a time

```bash
python scripts/download_trashnet.py
python scripts/prepare_dataset.py
python -m src.ecosort.eda
python -m src.ecosort.train --architecture baseline
python -m src.ecosort.evaluate
```

## Models

EcoSort AI contains two architectures.

### 1. Baseline CNN

A small convolutional neural network trained from scratch. This is the best model to start with because every layer and training decision is visible and explainable.

```bash
python -m src.ecosort.train --architecture baseline
```

### 2. MobileNetV3 transfer learning

After understanding the baseline, train a transfer-learning model:

```bash
python -m src.ecosort.train --architecture mobilenet --pretrained
```

The first run may download ImageNet weights through torchvision.

## Evaluation

```bash
python -m src.ecosort.evaluate
```

Generated files include:

- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/class_distribution.png`
- `reports/sample_grid.png`

Do not judge the model from accuracy alone. The `trash` class is much smaller than several other classes, so macro F1, per-class recall, and the confusion matrix matter.

## Interactive demo

### Live browser demo

Open the deployed Streamlit application:

**[https://ecosort-ai-elonagit.streamlit.app](https://ecosort-ai-elonagit.streamlit.app)**

A reviewer can upload a waste image and see the predicted class, confidence score, top alternatives, and disposal guidance.

### Local demo

After training locally:

```bash
streamlit run streamlit_app.py
```

The original Gradio interface is also available with:

```bash
python app.py
```

## Tests

```bash
pytest
```

The tests verify deterministic data splitting, model output shape, prediction utilities, and disposal guidance. They do not download the dataset or train the full model.

## What makes this a machine-learning project

This repository demonstrates:

- supervised image classification
- image preprocessing and augmentation
- train / validation / test separation
- class imbalance awareness
- convolutional neural networks
- transfer learning
- early stopping
- learning-rate scheduling
- precision, recall, F1, and confusion matrices
- model checkpointing and reproducibility
- interactive inference
- model limitations and responsible-use documentation

## Portfolio rule

Do not invent performance numbers. After each real training run, record the model, settings, metrics, mistakes, and lessons in [`MODEL_CARD.md`](MODEL_CARD.md). The reasoning is more valuable than a suspiciously perfect score.
