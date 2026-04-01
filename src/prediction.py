from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from src.model import load_trained_model


def preprocess_single_image(image_path, image_size=(64, 64)):
    img = Image.open(image_path).convert("RGB").resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def predict_image(image_path: str) -> Dict[str, float | str]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")

    model, labels = load_trained_model()
    image_batch = preprocess_single_image(str(path), image_size=(64, 64))
    probs = model.predict_proba(image_batch)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return {
        "predicted_class": labels[idx],
        "confidence": confidence,
    }
