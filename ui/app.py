import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError


API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=20)
def fetch_health(base_url: str):
    response = requests.get(f"{base_url}/health", timeout=8)
    if response.status_code != 200:
        return None, f"Health endpoint returned {response.status_code}: {response.text[:200]}"
    try:
        return response.json(), None
    except ValueError:
        return None, f"Health endpoint returned non-JSON response: {response.text[:200]}"


@st.cache_data(ttl=20)
def fetch_metrics(base_url: str):
    response = requests.get(f"{base_url}/metrics", timeout=8)
    if response.status_code == 429:
        return None, "Metrics endpoint is rate-limited (429). Wait a few seconds and refresh."
    if response.status_code != 200:
        return None, f"Metrics endpoint returned {response.status_code}: {response.text[:200]}"
    return response.text, None

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")
st.title("Image Classification Monitoring and Control")

st.write(f"API URL: {API_URL}")

st.subheader("Model Up-time")
try:
    health, health_error = fetch_health(API_URL)
    if health_error:
        st.error(f"Health endpoint unavailable: {health_error}")
    else:
        st.json(health)
except (requests.RequestException, ValueError) as ex:
    st.error(f"Health endpoint unavailable: {ex}")

st.subheader("Production Metrics")
try:
    metrics_text, metrics_error = fetch_metrics(API_URL)
    if metrics_error:
        st.warning(metrics_error)
    else:
        st.text(metrics_text[:4000])
except (requests.RequestException, ValueError) as ex:
    st.error(f"Metrics endpoint unavailable: {ex}")

st.divider()
st.subheader("Visualizations")

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
    st.bar_chart(counts_df.set_index("class"))

    shape_df = pd.DataFrame({"width": widths, "height": heights})
    st.write("Feature 2: Image width/height distribution")
    st.scatter_chart(shape_df)

    bright_df = pd.DataFrame({"brightness": brightness})
    st.write("Feature 3: Brightness distribution")
    st.line_chart(bright_df)
else:
    st.info("Add class folders and images under data/train to see visualizations.")

st.divider()
st.subheader("Production Evaluation")
if st.button("Run Production Evaluation"):
    try:
        response = requests.get(f"{API_URL}/evaluate", timeout=60)
        st.json(response.json())
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Evaluation failed: {ex}")

st.divider()
st.subheader("Predict Single Image")
pred_file = st.file_uploader("Upload one image for prediction", type=["png", "jpg", "jpeg", "bmp", "gif"], key="predict")
if pred_file and st.button("Predict"):
    files = {"file": (pred_file.name, pred_file.getvalue(), pred_file.type)}
    try:
        response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
        st.json(response.json())
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Prediction failed: {ex}")

st.divider()
st.subheader("Upload Bulk Data for Retraining")
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
        st.json(response.json())
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Upload failed: {ex}")

st.subheader("Trigger Retraining")
epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)
if st.button("Retrain Model"):
    try:
        response = requests.post(f"{API_URL}/retrain", params={"epochs": int(epochs)}, timeout=3600)
        st.json(response.json())
    except (requests.RequestException, ValueError) as ex:
        st.error(f"Retraining failed: {ex}")
