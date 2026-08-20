# EcoSort AI

EcoSort AI is a computer-vision project for classifying common household waste into six categories: **cardboard, glass, metal, paper, plastic, and trash**.

**Live demo:** https://ecosort-ai-elonagit.streamlit.app  
**Model:** MobileNetV3 transfer learning  
**Test accuracy:** 90.4%  
**Macro F1:** 0.908

## Overview

The project covers the full image-classification workflow: dataset preparation, augmentation, model training, evaluation, inference, and deployment.

A user can upload a photo through the Streamlit app and receive:

- the predicted waste category
- a confidence score
- the top three predictions
- basic disposal guidance

The repository also includes the training pipeline, evaluation reports, tests, and model documentation.

## Results

The deployed model was evaluated on **384 held-out TrashNet images**.

| Metric | Result |
| --- | ---: |
| Accuracy | 90.4% |
| Macro F1 | 0.908 |
| Test images | 384 |

Per-class results and the confusion matrix are available in the [`reports/`](reports/) directory.

## Dataset

EcoSort AI uses the **TrashNet** dataset created by Gary Thung and Mindy Yang. The dataset contains 2,527 resized images across six waste categories.

The images are not stored in this repository. The download script retrieves the dataset from the documented source and verifies the archive checksum before extraction.

Dataset attribution and licensing details are in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Model

Two architectures are included:

### Baseline CNN

A small convolutional neural network trained from scratch. It is useful as a simple baseline and for understanding the effect of the training pipeline without pretrained features.

```bash
python -m src.ecosort.train --architecture baseline
```

### MobileNetV3

The deployed version uses **MobileNetV3 Small** with ImageNet pretrained weights and transfer learning.

```bash
python -m src.ecosort.train --architecture mobilenet --pretrained
```

Training includes image augmentation, class weighting, early stopping, and learning-rate scheduling.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and prepare TrashNet:

```bash
python scripts/download_trashnet.py
python scripts/prepare_dataset.py
```

To run the existing trained model with Streamlit:

```bash
streamlit run streamlit_app.py
```

To reproduce the full baseline workflow:

```bash
python scripts/run_pipeline.py
```

## Project structure

```text
ecosort-ai/
├── streamlit_app.py        # deployed web interface
├── app.py                  # Gradio interface
├── src/ecosort/            # training, evaluation and inference code
├── scripts/                # dataset and pipeline utilities
├── models/                 # trained model checkpoint
├── reports/                # metrics and evaluation figures
├── tests/                  # automated tests
├── DATA_SOURCES.md
├── MODEL_CARD.md
└── requirements.txt
```

## Evaluation

Run evaluation against the prepared test split with:

```bash
python -m src.ecosort.evaluate
```

The evaluation step produces:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`

The project reports both overall accuracy and macro F1 because the dataset is not perfectly balanced across classes.

## Tests

```bash
pytest
```

The test suite covers deterministic dataset splitting, model output shape, prediction utilities, and disposal guidance.

## Limitations

TrashNet was photographed mostly against simple backgrounds, so performance can drop on cluttered scenes, poor lighting, damaged items, or objects made from multiple materials.

The disposal guidance in the demo is general information. Local recycling rules can differ by municipality and facility.

## References

- TrashNet dataset and original project: documented in [`DATA_SOURCES.md`](DATA_SOURCES.md)
- Model details and intended use: [`MODEL_CARD.md`](MODEL_CARD.md)
