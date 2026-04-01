from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_path(path_value):
    path = Path(path_value)
    if path.exists():
        return path

    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative

    return path


def load_folder_data(base_dir, class_to_index, image_size=(64, 64)):
    x_data = []
    y_data = []

    for class_name in class_to_index:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            continue

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue

            # Basic image preprocessing: RGB, resize, normalize, flatten.
            image = Image.open(image_path).convert("RGB").resize(image_size)
            image_array = np.array(image, dtype=np.float32) / 255.0
            x_data.append(image_array.reshape(-1))
            y_data.append(class_to_index[class_name])

    if len(x_data) == 0:
        feature_count = image_size[0] * image_size[1] * 3
        return np.empty((0, feature_count), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int64)


def get_image_datasets(train_dir="data/train", test_dir="data/test", image_size=(64, 64)):
    train_path = _resolve_data_path(train_dir)
    test_path = _resolve_data_path(test_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    class_names = sorted([folder.name for folder in train_path.iterdir() if folder.is_dir()])
    if len(class_names) == 0:
        raise ValueError("No class folders found in data/train")

    class_to_index = {name: i for i, name in enumerate(class_names)}

    x_train, y_train = load_folder_data(train_path, class_to_index, image_size=image_size)
    x_test, y_test = load_folder_data(test_path, class_to_index, image_size=image_size)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Training or test data is empty. Add image files to data/train and data/test.")

    return x_train, y_train, x_test, y_test, class_names
