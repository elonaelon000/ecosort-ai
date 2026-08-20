import torch

from src.ecosort.predict import top_predictions


def test_top_predictions_sorted_and_named():
    probs = torch.tensor([0.05, 0.11, 0.50, 0.20, 0.09, 0.05])
    names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    result = top_predictions(probs, names, top_k=3)
    assert [item["class_name"] for item in result] == ["metal", "paper", "glass"]
    assert result[0]["confidence"] == 0.5


def test_top_k_must_be_positive():
    try:
        top_predictions(torch.tensor([0.5, 0.5]), ["a", "b"], top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
