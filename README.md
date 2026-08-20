# EcoSort AI

Image classification for everyday waste sorting.

**Live demo:** https://ecosort-ai-elonagit.streamlit.app  
**Model:** MobileNetV3 Small (transfer learning)  
**Test accuracy:** 90.4%  
**Macro F1:** 0.908

EcoSort AI takes a photo of a single waste item and predicts one of six classes: **cardboard, glass, metal, paper, plastic, or trash**. The Streamlit demo also shows prediction confidence, the top three classes, and a basic disposal suggestion.

## Results

The deployed model was evaluated on a held-out split of **384 TrashNet images**.

| Metric | Result |
| --- | ---: |
| Accuracy | 90.4% |
| Macro F1 | 0.908 |
| Test images | 384 |

Detailed results are stored in [`reports/`](reports/), including the classification report and confusion matrix.

## Dataset

This project uses **TrashNet**, created by Gary Thung and Mindy Yang. It contains 2,527 images across the six waste categories used here.

The dataset is not committed to this repository. `scripts/download_trashnet.py` downloads it from the documented source and verifies the archive checksum before extraction.

Source and license information: [`DATA_SOURCES.md`](DATA_SOURCES.md)

## Model

The repository includes two model options:

- **Baseline CNN** — a small convolutional network trained from scratch.
- **MobileNetV3 Small** — transfer learning with ImageNet pretrained weights. This is the model used by the deployed app.

Training uses image augmentation, class weighting, early stopping, and learning-rate scheduling.

Train the baseline:

```bash
python -m src.ecosort.train --architecture baseline
```

Train MobileNetV3:

```bash
python -m src.ecosort.train --architecture mobilenet --pretrained
```

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Download and prepare the dataset:

```bash
python scripts/download_trashnet.py
python scripts/prepare_dataset.py
```

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Run the baseline pipeline from download through evaluation:

```bash
python scripts/run_pipeline.py
```

## Evaluation

```bash
python -m src.ecosort.evaluate
```

This writes the main evaluation artifacts to `reports/`:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`

Accuracy is reported together with macro F1 because the class distribution is uneven, especially for the `trash` class.

## Tests

```bash
pytest
```

Tests cover dataset splitting, model output shape, prediction helpers, and disposal guidance.

## Repository layout

```text
ecosort-ai/
├── streamlit_app.py        # deployed Streamlit interface
├── app.py                  # Gradio interface
├── src/ecosort/            # model, training, evaluation and inference code
├── scripts/                # dataset and pipeline scripts
├── models/                 # trained checkpoint
├── reports/                # metrics and evaluation figures
├── tests/
├── DATA_SOURCES.md
├── MODEL_CARD.md
└── requirements.txt
```

## Limitations

TrashNet images are mostly photographed against simple backgrounds. Real-world photos can be harder: clutter, poor lighting, damaged objects, mixed materials, and unusual angles can all reduce prediction quality.

The disposal suggestion is intentionally general. Recycling rules vary by location, so the app should not be treated as a local recycling authority.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — model details, evaluation, intended use, and limitations
- [`DATA_SOURCES.md`](DATA_SOURCES.md) — dataset source, attribution, and license notes
- [`LEARNING_NOTES.md`](LEARNING_NOTES.md) — project notes and implementation decisions
