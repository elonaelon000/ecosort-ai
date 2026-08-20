from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from src.ecosort.guidance import disposal_guidance
from src.ecosort.predict import predict_image

MODEL_PATH = Path("models/ecosort.pt")
METRICS_PATH = Path("reports/metrics.json")


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def as_percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4fbf6 0%, #ffffff 36%);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            background: linear-gradient(135deg, #0b5d36 0%, #198754 55%, #4ab477 100%);
            border-radius: 26px;
            padding: 2.2rem 2.35rem;
            color: white;
            box-shadow: 0 22px 50px rgba(24, 135, 84, 0.18);
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.4rem, 6vw, 4.1rem);
            line-height: 1;
            letter-spacing: -0.04em;
        }

        .hero p {
            max-width: 760px;
            margin: 0.85rem 0 0;
            font-size: 1.08rem;
            line-height: 1.65;
            color: rgba(255, 255, 255, 0.92);
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.2rem;
        }

        .badge {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.86rem;
            color: white;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #dfeee4;
            border-radius: 22px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 14px 32px rgba(22, 82, 49, 0.07);
            min-height: 100%;
        }

        .eyebrow {
            color: #16834f;
            font-weight: 800;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.35rem;
        }

        .prediction-card {
            background: linear-gradient(135deg, #f4fff7 0%, #e9f8ee 100%);
            border: 1px solid #cfe9d8;
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            margin: 0.5rem 0 1rem;
        }

        .prediction-label {
            color: #607168;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .prediction-class {
            color: #0b5d36;
            font-size: 2rem;
            font-weight: 900;
            line-height: 1.05;
            margin-bottom: 0.55rem;
        }

        .confidence-chip {
            display: inline-block;
            color: #0b5d36;
            background: #d9f3e3;
            border: 1px solid #bde5cc;
            border-radius: 999px;
            padding: 0.3rem 0.72rem;
            font-weight: 800;
            font-size: 0.92rem;
        }

        .guidance-card {
            background: #f7fbff;
            border: 1px solid #dce9f2;
            border-left: 5px solid #198754;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            margin-top: 1rem;
        }

        .stats-title {
            margin-top: 1.9rem;
            margin-bottom: 0.25rem;
            font-size: 1.35rem;
            font-weight: 850;
            color: #173b29;
        }

        .stat-card {
            background: #ffffff;
            border: 1px solid #dfeee4;
            border-radius: 17px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(22, 82, 49, 0.055);
        }

        .stat-label {
            color: #66786e;
            font-size: 0.84rem;
            margin-bottom: 0.2rem;
        }

        .stat-value {
            color: #153b28;
            font-size: 1.55rem;
            font-weight: 900;
        }

        .footer-note {
            margin-top: 1.7rem;
            padding: 1rem 1.15rem;
            background: #f8fbf9;
            border: 1px solid #e5eee8;
            border-radius: 16px;
            color: #586b61;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        div[data-testid="stFileUploader"] section {
            border-radius: 16px;
            border: 1.5px dashed #9bc9ac;
            background: #f7fcf8;
        }

        div[data-testid="stProgress"] > div > div > div > div {
            background-color: #198754;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero {
                padding: 1.55rem 1.35rem;
                border-radius: 20px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()
accuracy = metrics.get("accuracy") if metrics else None
macro_f1 = metrics.get("macro_f1") if metrics else None
num_examples = metrics.get("num_test_examples") if metrics else None

st.markdown(
    """
    <div class="hero">
        <h1>♻️ EcoSort AI</h1>
        <p>
            Upload waste. Get an instant classification. Sort smarter.
            EcoSort AI uses computer vision to recognize six common waste categories
            and turn a photo into practical recycling guidance.
        </p>
        <div class="badge-row">
            <span class="badge">6 waste categories</span>
            <span class="badge">Computer vision</span>
            <span class="badge">MobileNetV3</span>
            <span class="badge">Environmental ML</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="eyebrow">Upload image</div>', unsafe_allow_html=True)
    st.markdown("### Show EcoSort one waste item")
    st.write(
        "Use a clear photo of a single bottle, can, cardboard package, glass jar, "
        "paper product, or general waste item."
    )

    uploaded_file = st.file_uploader(
        "Choose a waste photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    else:
        image = None
        st.info("Upload an image to start the classification.")

with right:
    st.markdown('<div class="eyebrow">Model output</div>', unsafe_allow_html=True)
    st.markdown("### What does the model see?")

    if not MODEL_PATH.exists():
        st.warning("The trained model is not available in this deployment yet.")
    elif image is None:
        st.markdown(
            """
            <div class="prediction-card">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-class">Waiting for image</div>
                <div style="color:#64766d;">Your result will appear here after upload.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Classifying image..."):
            predictions = predict_image(image, top_k=3)

        best = predictions[0]
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-class">{best['class_name'].title()}</div>
                <span class="confidence-chip">Confidence {best['confidence']:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Top predictions")
        for item in predictions:
            label = item["class_name"].title()
            confidence = float(item["confidence"])
            st.write(f"**{label}** · {confidence:.1%}")
            st.progress(confidence)

        st.markdown(
            f"""
            <div class="guidance-card">
                <div class="eyebrow">What should I do with this?</div>
                <div>{disposal_guidance(best['class_name'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="stats-title">Model snapshot</div>', unsafe_allow_html=True)
st.caption("Performance measured on the held-out TrashNet test split used by this project.")

stat1, stat2, stat3 = st.columns(3, gap="medium")
with stat1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Test accuracy</div>
            <div class="stat-value">{as_percent(accuracy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Macro F1</div>
            <div class="stat-value">{as_percent(macro_f1)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Held-out test images</div>
            <div class="stat-value">{num_examples if num_examples is not None else '—'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-note">
        <strong>How it works:</strong> the uploaded image is resized and passed through a
        convolutional neural network. The model extracts visual features and returns a
        probability for each of the six waste categories; the highest probability becomes
        the prediction.<br><br>
        <strong>Important:</strong> EcoSort AI is an educational prototype. Recycling rules
        vary by location, and the model can still make mistakes, especially with mixed
        materials, cluttered backgrounds, unusual lighting, or damaged objects.
    </div>
    """,
    unsafe_allow_html=True,
)
