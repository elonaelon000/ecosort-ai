# EcoSort AI Project Plan

## Goal

Build and understand an end-to-end environmental computer-vision system rather than only calling an existing AI API.

## Phase 1 — Baseline

- Download and verify TrashNet.
- Create deterministic train / validation / test splits.
- Inspect class imbalance and example images.
- Train a custom CNN from scratch.
- Evaluate on the untouched test set.
- Record accuracy, macro F1, per-class precision/recall/F1, and confusion matrix.

## Phase 2 — Improvement

- Train MobileNetV3 with transfer learning.
- Compare it against the baseline using the same test set.
- Inspect which classes are confused most often.
- Decide whether augmentation, class weighting, or more data is justified.

## Phase 3 — Real-world test

- Photograph at least 30–60 original waste examples in different lighting/backgrounds.
- Keep them separate from training.
- Compare performance on TrashNet test data vs. original real-world photos.
- Document failure cases honestly.

## Phase 4 — Demo

- Use the Gradio app for image uploads.
- Show confidence and top predictions.
- Provide generic disposal guidance and a local-rules disclaimer.

## Stretch ideas

- add organic waste only after obtaining enough licensed data
- add e-waste as a separate model because it has distinct safety implications
- move from single-image classification to object detection for multiple items
- add location-specific recycling rules from verified municipal sources
- quantify uncertainty and reject low-confidence inputs
- compare MobileNetV3 with EfficientNet or ResNet

## Interview questions to master

1. Why did I separate training, validation, and test data?
2. What is data leakage?
3. Why can class imbalance make accuracy misleading?
4. What does recall mean for the `trash` class?
5. What does a confusion matrix show?
6. What is overfitting and how can I detect it?
7. Why does transfer learning help on a relatively small dataset?
8. Why can a 95% confidence prediction still be wrong?
