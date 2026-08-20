from __future__ import annotations

import gradio as gr
from PIL import Image

from src.ecosort.guidance import disposal_guidance
from src.ecosort.predict import predict_image


def classify(image: Image.Image | None):
    if image is None:
        return {}, "Upload an image to classify it."
    try:
        predictions = predict_image(image, top_k=3)
    except FileNotFoundError as exc:
        return {}, str(exc)

    scores = {item["class_name"]: item["confidence"] for item in predictions}
    best = predictions[0]
    message = (
        f"Predicted class: **{best['class_name'].title()}**\n\n"
        f"Confidence: **{best['confidence']:.1%}**\n\n"
        f"{disposal_guidance(best['class_name'])}\n\n"
        "_Educational prototype: always check local recycling rules._"
    )
    return scores, message


with gr.Blocks(title="EcoSort AI") as demo:
    gr.Markdown("# ♻️ EcoSort AI")
    gr.Markdown(
        "Upload a clear photo of one waste item. The model estimates its broad material category."
    )
    with gr.Row():
        image = gr.Image(type="pil", label="Waste item")
        with gr.Column():
            labels = gr.Label(num_top_classes=3, label="Model predictions")
            guidance = gr.Markdown()
    button = gr.Button("Classify")
    button.click(classify, inputs=image, outputs=[labels, guidance])


if __name__ == "__main__":
    demo.launch()
