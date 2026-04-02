import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")


@st.cache_data(ttl=60)
def fetch_health(base_url: str):
    response = requests.get(f"{base_url}/health", timeout=8)
    if response.status_code == 429:
        return None, "Health endpoint is rate-limited (429). Wait a few seconds and refresh."
    if response.status_code != 200:
        return None, f"Health endpoint returned {response.status_code}: {response.text[:200]}"
    try:
        return response.json(), None
    except ValueError:
        return None, f"Health endpoint returned non-JSON response: {response.text[:200]}"


@st.cache_data(ttl=60)
def fetch_metrics(base_url: str):
    response = requests.get(f"{base_url}/metrics", timeout=8)
    if response.status_code == 429:
        return None, "Metrics endpoint is rate-limited (429). Wait a few seconds and refresh."
    if response.status_code != 200:
        return None, f"Metrics endpoint returned {response.status_code}: {response.text[:200]}"
    return response.text, None


def parse_api_json(response: requests.Response, action_name: str):
    if response.status_code == 429:
        return None, f"{action_name} is temporarily rate-limited (429). Please wait a few seconds and try again."
    if response.status_code >= 400:
        return None, f"{action_name} failed ({response.status_code}): {response.text[:200]}"
    try:
        return response.json(), None
    except ValueError:
        return None, f"{action_name} returned a non-JSON response. Please retry in a few seconds."


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.hero-box {
    border: 1px solid #dbe7f3;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    background: linear-gradient(120deg, #f5fbff 0%, #f8fff9 100%);
}

.section-note {
    color: #3f4f5f;
    margin-top: -0.2rem;
    margin-bottom: 0.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-box">
    <h1 style="margin:0; color:#0f3554;">Image Classification Monitoring and Control</h1>
    <p style="margin:0.45rem 0 0 0; color:#294a66;">
        Use this dashboard to monitor model health, review data insights, run predictions, upload new images,
        and trigger retraining.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.title("Image Classification Monitoring and Control")

st.write(f"API URL: {API_URL}")

st.subheader("Model Up-time")
st.caption("Check if the API is reachable and whether the trained model is currently loaded.")
if "health_data" not in st.session_state:
    st.session_state.health_data = None
if "health_error" not in st.session_state:
    st.session_state.health_error = None

if st.button("Refresh Health"):
    try:
        health, health_error = fetch_health(API_URL)
        st.session_state.health_data = health
        st.session_state.health_error = health_error
    except (requests.RequestException, ValueError) as ex:
        st.session_state.health_data = None
        st.session_state.health_error = f"Health endpoint unavailable: {ex}"

if st.session_state.health_error:
    st.warning(st.session_state.health_error)
elif st.session_state.health_data:
    st.json(st.session_state.health_data)
else:
    st.info("Click 'Refresh Health' to fetch current API/model status.")

st.subheader("Production Metrics")
st.caption("Live Prometheus metrics for request count, latency, and retraining activity.")
if "metrics_text" not in st.session_state:
    st.session_state.metrics_text = None
if "metrics_error" not in st.session_state:
    st.session_state.metrics_error = None

if st.button("Refresh Metrics"):
    try:
        metrics_text, metrics_error = fetch_metrics(API_URL)
        st.session_state.metrics_text = metrics_text
        st.session_state.metrics_error = metrics_error
    except (requests.RequestException, ValueError) as ex:
        st.session_state.metrics_text = None
        st.session_state.metrics_error = f"Metrics endpoint unavailable: {ex}"

if st.session_state.metrics_error:
    st.warning(st.session_state.metrics_error)
elif st.session_state.metrics_text:
    st.text(st.session_state.metrics_text[:4000])
else:
    st.info("Click 'Refresh Metrics' to fetch current Prometheus metrics.")

st.divider()
st.subheader("Visualizations")
st.caption("Simple insights from your training data: class balance, image dimensions, and brightness patterns.")

train_path = Path("data/train")
class_counts = {}
widths = []
heights = []
brightness = []

if train_path.exists():
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            files = [p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}]
            class_counts[class_dir.name] = len(files)
            for f in files[:40]:
                try:
                    img = Image.open(f).convert("RGB")
                    arr = np.array(img)
                    h, w, _ = arr.shape
                    widths.append(w)
                    heights.append(h)
                    brightness.append(float(arr.mean()))
                except (UnidentifiedImageError, OSError, ValueError):
                    continue

if class_counts:
    counts_df = pd.DataFrame({"class": list(class_counts.keys()), "count": list(class_counts.values())})
    st.write("Feature 1: Class distribution")
    st.caption("Shows how many images exist per class. Big imbalance can affect model quality.")
    st.bar_chart(counts_df.set_index("class"))

    shape_df = pd.DataFrame({"width": widths, "height": heights})
    st.write("Feature 2: Image width/height distribution")
    st.caption("Shows whether image sizes are consistent. Very different sizes can hurt performance.")
    st.scatter_chart(shape_df)

    bright_df = pd.DataFrame({"brightness": brightness})
    st.write("Feature 3: Brightness distribution")
    st.caption("Shows if images are mostly dark or bright. Lighting variation can change predictions.")
    st.line_chart(bright_df)
else:
    st.info("Add class folders and images under data/train to see visualizations.")

st.divider()
st.subheader("Production Evaluation")
st.caption("Run evaluation on the current production model and review performance metrics.")
if st.button("Run Production Evaluation"):
    try:
        response = requests.get(f"{API_URL}/evaluate", timeout=60)
        body, error = parse_api_json(response, "Evaluation")
        if error:
            st.warning(error)
        else:
            st.json(body)
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Evaluation failed: {ex}")

st.divider()
st.subheader("Predict Single Image")
st.caption("Upload one image to get its predicted class from the deployed model.")
pred_file = st.file_uploader("Upload one image for prediction", type=["png", "jpg", "jpeg", "bmp", "gif"], key="predict")
if pred_file and st.button("Predict"):
    files = {"file": (pred_file.name, pred_file.getvalue(), pred_file.type)}
    try:
        response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
        body, error = parse_api_json(response, "Prediction")
        if error:
            st.warning(error)
        else:
            st.json(body)
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Prediction failed: {ex}")

st.divider()
st.subheader("Upload Bulk Data for Retraining")
st.caption("Upload multiple images for one class. These files are saved and used in future retraining.")
class_name = st.text_input("Class name for uploaded files")
bulk_files = st.file_uploader(
    "Upload multiple images for retraining",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    accept_multiple_files=True,
    key="bulk",
)

if bulk_files and class_name and st.button("Upload Bulk"):
    files = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
    try:
        response = requests.post(f"{API_URL}/upload-bulk", params={"class_name": class_name}, files=files, timeout=120)
        body, error = parse_api_json(response, "Bulk upload")
        if error:
            st.warning(error)
        else:
            st.json(body)
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Upload failed: {ex}")

st.subheader("Trigger Retraining")
st.caption("Start a new training run using current dataset and uploaded images.")
epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)
if st.button("Retrain Model"):
    try:
        response = requests.post(f"{API_URL}/retrain", params={"epochs": int(epochs)}, timeout=3600)
        body, error = parse_api_json(response, "Retraining")
        if error:
            st.warning(error)
        else:
            st.json(body)
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Retraining failed: {ex}")
