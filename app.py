"""Launch from the repository root: python -m streamlit run app.py."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from apple_disease.preprocessing import read_image
from apple_disease.features import glcm_features, FEATURE_NAMES
from apple_disease.segmentation import segment
from apple_disease.inference import available_models, available_detector, load_classifier, predict_classifier

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Apple Disease Explorer", page_icon="🍎", layout="centered")
st.title("Apple Disease Explorer")
st.caption("Four-class fruit image experiment · Blotch / Rot / Scab / Normal")
st.info("Trained on a small recovered fruit-photo dataset. Predictions are uncalibrated and may be wrong, especially for leaves, unfamiliar images, or non-apples.")


@st.cache_resource
def get_classifier(directory, checkpoint_hash):
    return load_classifier(directory)


models = available_models(os.environ.get("APPLE_ARTIFACTS_DIR", str(ROOT / "artifacts")))
if not models:
    st.warning("No trained classification checkpoints are available. Image texture and clustering still work.")
    selected = None
else:
    default = "resnet50" if "resnet50" in models else next(iter(models))
    selected = st.selectbox("Classification model", list(models), index=list(models).index(default),
                            format_func=lambda x: {"ann": "ANN · GLCM texture", "cnn": "Custom CNN", "resnet50": "ResNet50 · transfer learning"}[x])

upload = st.file_uploader("Upload an apple fruit image", type=["jpg", "jpeg", "png", "webp"])
detector = available_detector(ROOT / "models/detection.json")
if detector is None:
    st.caption("Custom disease detection is unavailable: no verified trained detector is registered.")

if upload is not None:
    if upload.size > 10 * 1024**2:
        st.error("Choose an image file smaller than 10 MB.")
        st.stop()
    try:
        image = read_image(upload)
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    st.image(image, caption="Uploaded image", width="stretch")
    classification, clustering, texture = st.tabs(["Classification", "K-means clusters", "Texture features"])
    with classification:
        if selected:
            try:
                info = json.loads((models[selected] / "metadata.json").read_text())
                with st.spinner("Loading model and classifying…"):
                    model, info = get_classifier(str(models[selected]), info["checkpoint_sha256"])
                    result = predict_classifier(model, info, image)
                st.subheader(result["class"])
                st.metric("Highest model probability", f"{result['confidence']:.1%}")
                if result["low_confidence"]:
                    st.warning("Low confidence (below 60%). Try a clear image of one apple; this threshold is a display rule, not calibrated certainty.")
                st.bar_chart(pd.Series(result["probabilities"], name="Probability"))
                st.dataframe(pd.DataFrame({"Class": result["probabilities"].keys(), "Probability": result["probabilities"].values()}), hide_index=True)
                st.caption(f"Selected training phase: {info['selected_phase']}. Confidence is not a verified disease diagnosis.")
            except Exception as exc:
                st.error(f"This classifier could not be loaded or used: {exc}")
        else:
            st.write("Add a trained checkpoint to enable classification.")
    with clustering:
        k = st.slider("Number of color clusters", 2, 6, 3)
        if st.button("Show K-means segmentation"):
            try:
                resized = np.asarray(Image.fromarray(image).resize((192, 192), Image.Resampling.BILINEAR))
                segmented, labels, _ = segment(resized, k)
                left, right = st.columns(2)
                left.image(segmented, caption="Cluster color reconstruction", width="stretch")
                palette = np.array([[31,119,180], [255,127,14], [44,160,44], [214,39,40], [148,103,189], [140,86,75]], dtype=np.uint8)
                right.image(palette[labels], caption="Cluster IDs", width="stretch")
                st.write("Clusters group similar Lab colors. A cluster can mix shadows, background and disease marks; it is not a biological lesion label.")
            except ValueError as exc:
                st.warning(str(exc))
    with texture:
        resized = np.asarray(Image.fromarray(image).resize((96, 96), Image.Resampling.BILINEAR))
        values = glcm_features(resized)
        st.dataframe(pd.DataFrame({"Feature": FEATURE_NAMES, "Value": values}), hide_index=True)
        st.caption("32 gray levels; distances 1, 2, 4; angles 0°, 45°, 90°, 135°. Mean/std describe variation across these settings.")
    if detector:
        if st.button("Detect and draw disease bounding boxes"):
            try:
                from apple_disease.detection import predict_yolo
                rendered, boxes = predict_yolo(detector["weights"], image, expected_names=detector["names"])
                st.image(rendered, caption="Detected boxes and classes", width="stretch")
                st.dataframe(boxes) if boxes else st.info("No boxes exceed the confidence threshold.")
            except Exception as exc:
                st.error(f"Detection is unavailable: {exc}")
