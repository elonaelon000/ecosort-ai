from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image

from .config import MODEL_PATH
from .data import eval_transform
from .model import build_model


@lru_cache(maxsize=1)
def load_artifacts(model_path: str = str(MODEL_PATH)):
    path = MODEL_PATH if model_path == str(MODEL_PATH) else __import__('pathlib').Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.ecosort.train` first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(
        len(class_names),
        architecture=checkpoint["architecture"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names


def prepare_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    return eval_transform()(image).unsqueeze(0)


def top_predictions(probabilities: torch.Tensor, class_names: list[str], top_k: int = 3):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    k = min(top_k, len(class_names))
    values, indices = torch.topk(probabilities, k=k)
    return [
        {"class_name": class_names[index], "confidence": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]


def predict_image(image: Image.Image, top_k: int = 3):
    model, class_names = load_artifacts()
    tensor = prepare_image(image)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor)[0], dim=0)
    return top_predictions(probabilities, class_names, top_k=top_k)
