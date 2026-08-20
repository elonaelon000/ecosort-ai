from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.ecosort.guidance import disposal_guidance
from src.ecosort.predict import predict_image

MODEL_PATH = Path("models/ecosort.pt")

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ EcoSort AI")
st.caption("Computer vision for smarter waste sorting")

st.markdown(
    "EcoSort AI classifies a photo into one of six broad waste categories: "
    "**cardboard, glass, metal, paper, plastic, or trash**. The model was built "
    "as an environmental machine-learning project using the TrashNet dataset."
)

uploaded_file = st.file_uploader(
    "Upload a clear photo of one waste item",
    type=["jpg", "jpeg", "png", "webp"],
)

if not MODEL_PATH.exists():
    st.info(
        "The interface is ready, but the trained model is not in this branch yet. "
        "Once training finishes, predictions will appear here automatically."
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if MODEL_PATH.exists():
        with st.spinner("Classifying image..."):
            predictions = predict_image(image, top_k=3)

        best = predictions[0]
        st.subheader(f"Prediction: {best['class_name'].title()}")
        st.metric("Confidence", f"{best['confidence']:.1%}")

        st.markdown("#### Top predictions")
        for item in predictions:
            label = item["class_name"].title()
            confidence = item["confidence"]
            st.write(f"**{label}** — {confidence:.1%}")
            st.progress(float(confidence))

        st.markdown("#### Disposal guidance")
        st.info(disposal_guidance(best["class_name"]))

st.divider()
st.markdown(
    "**How it works:** a convolutional neural network receives a resized image, "
    "extracts visual features, and returns probabilities for the six classes. "
    "The final prediction is the class with the highest probability."
)
st.caption(
    "Educational prototype. Recycling rules vary by location, and the model can be wrong—"
    "especially with mixed materials, unusual lighting, or cluttered backgrounds."
)
