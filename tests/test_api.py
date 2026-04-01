from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_evaluate_returns_metrics(monkeypatch):
    def fake_evaluate_trained_model():
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "loss": 0.0,
        }

    monkeypatch.setattr("src.api.evaluate_trained_model", fake_evaluate_trained_model)
    response = client.get("/evaluate")
    assert response.status_code == 200
    assert "accuracy" in response.json()


def test_predict_rejects_non_image_file():
    files = {"file": ("bad.txt", b"not-image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_upload_bulk_stores_and_triggers_retrain(monkeypatch):
    saved_calls = []

    def fake_save_uploaded_file(class_name, filename, content, content_type=None):
        saved_calls.append((class_name, filename, content, content_type))
        return 1

    def fake_retrain_model(epochs=3):
        return {"accuracy": 1.0, "moved_files": 0, "copied_files": 0, "epochs": epochs}

    monkeypatch.setattr("src.api.save_uploaded_file", fake_save_uploaded_file)
    monkeypatch.setattr("src.api.retrain_model", fake_retrain_model)

    files = [
        ("files", ("sample1.png", b"fake-image-1", "image/png")),
        ("files", ("sample2.png", b"fake-image-2", "image/png")),
    ]
    response = client.post("/upload-bulk", params={"class_name": "1", "auto_retrain": "true"}, files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["stored_files"] == 2
    assert body["retrain_triggered"] is True
    assert len(saved_calls) == 2
