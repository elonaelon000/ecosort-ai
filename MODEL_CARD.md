# EcoSort AI Model Card

## Intended use

Educational classification of a **single clearly visible waste item** into one of six broad categories: cardboard, glass, metal, paper, plastic, or trash.

## Not intended for

- official municipal recycling decisions
- hazardous-material identification
- medical, food-safety, or chemical decisions
- images containing several mixed objects without a clear primary item
- determining contamination that cannot be seen reliably from the image
- deciding whether an item is accepted by a specific city recycling program

## Dataset

**TrashNet** — Gary Thung and Mindy Yang.

See `DATA_SOURCES.md` for source, license, counts, checksum, and limitations.

Known imbalance: the original source reports only 137 `trash` images compared with 594 `paper` images. Accuracy alone can therefore hide poor minority-class performance.

## Experiment log

Do not fill in metrics until the corresponding command has actually been run.

### Experiment 1 — Baseline CNN

- Date: _TBD_
- Architecture: custom CNN trained from scratch
- Input size: 160 × 160
- Split: 70% train / 15% validation / 15% test, deterministic seed 42
- Epochs completed: _TBD_
- Best validation loss: _TBD_
- Test accuracy: _TBD_
- Macro F1: _TBD_
- Weakest class: _TBD_
- Most common confusion: _TBD_
- What I learned: _TBD_

### Experiment 2 — MobileNetV3 transfer learning

- Date: _TBD_
- Architecture: MobileNetV3 Small
- Pretrained ImageNet weights: yes
- Change from baseline: transfer learning + feature extractor
- Epochs completed: _TBD_
- Test accuracy: _TBD_
- Macro F1: _TBD_
- Did it improve over baseline?: _TBD_
- Failure cases: _TBD_
- What I learned: _TBD_

## Practical limitations

TrashNet images were collected in relatively controlled conditions. Real waste varies substantially in color, shape, branding, damage, lighting, camera angle, and background. A model can also be confidently wrong because softmax confidence is not a guarantee of correctness.

## Responsible interpretation

EcoSort AI should be described as a **material-classification experiment**, not a universal recycling decision engine. A useful next step would be collecting a small, independently photographed test set from real household conditions and reporting how much performance changes.
