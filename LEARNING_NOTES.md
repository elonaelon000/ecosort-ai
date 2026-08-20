# EcoSort AI Learning Notes

These are the concepts to understand well enough to explain without memorizing code.

## Image classification

The model receives an image and returns probabilities for a fixed set of labels. In EcoSort AI, exactly one broad class is assumed to be the primary class in each image.

## Convolutional neural network (CNN)

A CNN learns small visual filters and combines them into increasingly complex features. Early layers often respond to edges and textures; deeper layers can represent larger visual patterns.

## Training / validation / test

- **Training:** used to update model weights.
- **Validation:** used while developing the model and deciding when to stop training.
- **Test:** kept untouched until evaluation so it estimates performance on unseen examples.

Using the test set repeatedly to tune decisions indirectly turns it into another validation set and makes the final score less trustworthy.

## Data augmentation

Random flips, rotations, crops, and color changes create plausible variations during training. Augmentation can reduce overfitting, but unrealistic transformations can hurt performance.

## Overfitting

Overfitting happens when training performance keeps improving while validation performance stops improving or becomes worse. The model is learning training-specific details instead of general visual patterns.

## Precision

Of the images the model predicted as a class, how many were correct?

## Recall

Of the images that truly belong to a class, how many did the model find?

For an imbalanced class like `trash`, recall can reveal problems that overall accuracy hides.

## F1 score

F1 combines precision and recall. **Macro F1** calculates F1 for each class and averages them equally, so larger classes cannot dominate the metric.

## Transfer learning

A pretrained model has already learned useful visual features from a large dataset. We replace its final classifier and adapt it to our waste classes. This often works better than training a large network from scratch on a small dataset.

## Softmax confidence

Softmax turns model scores into values that sum to 1. A value such as 0.90 is not a guarantee that the prediction is truly correct 90% of the time. Neural networks can be overconfident, especially on unfamiliar images.
