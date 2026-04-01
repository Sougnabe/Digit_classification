from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import get_image_datasets


MODEL_PATH = Path("models/image_classifier.pkl")
LABELS_PATH = Path("models/labels.txt")


def build_model(epochs=1):
    max_iter = max(500, epochs * 500)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def train_and_evaluate(train_dir="data/train", test_dir="data/test", epochs=1, image_size=(64, 64)):
    x_train, y_train, x_test, y_test, class_names = get_image_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
    )

    model = build_model(epochs=epochs)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "loss": float(log_loss(y_test, y_proba, labels=list(range(y_proba.shape[1])))),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    LABELS_PATH.write_text("\n".join(class_names), encoding="utf-8")

    return results


def evaluate_trained_model(test_dir="data/test", train_dir="data/train", image_size=(64, 64)):
    model, _ = load_trained_model()
    _, _, x_test, y_test, _ = get_image_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
    )

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "loss": float(log_loss(y_test, y_proba, labels=list(range(y_proba.shape[1])))),
    }


def load_trained_model() -> Tuple[Pipeline, List[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Train the model first.")
    if not LABELS_PATH.exists():
        raise FileNotFoundError("No labels file found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    labels = [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return model, labels


if __name__ == "__main__":
    output = train_and_evaluate()
    print("Training and evaluation complete:")
    for k, v in output.items():
        print(f"{k}: {v:.4f}")
