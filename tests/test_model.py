import torch

from src.ecosort.model import build_model


def test_baseline_output_shape():
    model = build_model(6, architecture="baseline")
    batch = torch.randn(2, 3, 160, 160)
    output = model(batch)
    assert output.shape == (2, 6)


def test_invalid_architecture():
    try:
        build_model(6, architecture="unknown")
    except ValueError as exc:
        assert "architecture" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
