import os
from pathlib import Path

from locust import HttpUser, between, task


SAMPLE_IMAGE_PATH = os.getenv("SAMPLE_IMAGE", "data/test/sample.jpg")


class MLApiUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def predict(self):
        path = Path(SAMPLE_IMAGE_PATH)
        if not path.exists():
            self.client.get("/health", name="health-fallback")
            return

        with path.open("rb") as f:
            files = {"file": (path.name, f, "image/jpeg")}
            self.client.post("/predict", files=files, name="predict")
