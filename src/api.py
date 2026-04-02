import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import generate_latest

from src.monitoring import REQUEST_COUNT, REQUEST_LATENCY, RETRAIN_COUNT
from src.upload_store import save_uploaded_file


app = FastAPI(title="Image Classification API", version="1.0.0")
MODEL_PATH = Path("models/image_classifier.pkl")


def evaluate_trained_model(*args, **kwargs):
    from src.model import evaluate_trained_model as _evaluate_trained_model

    return _evaluate_trained_model(*args, **kwargs)


def retrain_model(*args, **kwargs):
    from src.retrain import retrain_model as _retrain_model

    return _retrain_model(*args, **kwargs)


def predict_image(*args, **kwargs):
    from src.prediction import predict_image as _predict_image

    return _predict_image(*args, **kwargs)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "uptime_hint": "Use /metrics for request counters and latency histograms.",
    }


@app.get("/metrics")
def get_metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    REQUEST_COUNT.inc()

    filename = file.filename or "uploaded_image"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_dir = Path("data/uploads/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    with temp_path.open("wb") as f:
        f.write(await file.read())

    try:
        Image.open(temp_path).verify()
        result = predict_image(str(temp_path))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {ex}") from ex
    finally:
        if temp_path.exists():
            temp_path.unlink()

    REQUEST_LATENCY.observe(time.perf_counter() - start)
    return JSONResponse(result)


@app.post("/upload-bulk")
async def upload_bulk(class_name: str, auto_retrain: bool = False, files: List[UploadFile] = File(...)):
    if not class_name.strip():
        raise HTTPException(status_code=400, detail="class_name is required")

    target = Path("data/uploads") / class_name
    target.mkdir(parents=True, exist_ok=True)

    stored = 0
    for file in files:
        filename = file.filename or "uploaded_image"
        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            continue
        content = await file.read()
        out = target / filename
        with out.open("wb") as f:
            f.write(content)
        save_uploaded_file(class_name=class_name, filename=filename, content=content, content_type=file.content_type)
        stored += 1

    response = {"stored_files": stored, "class_name": class_name, "auto_retrain": auto_retrain}

    if auto_retrain and stored > 0:
        retrain_metrics = retrain_model(epochs=3)
        RETRAIN_COUNT.inc()
        response["retrain_triggered"] = True
        response["retrain_metrics"] = retrain_metrics
    else:
        response["retrain_triggered"] = False

    return response


@app.post("/retrain")
def retrain(epochs: int = 3):
    try:
        output = retrain_model(epochs=epochs)
        RETRAIN_COUNT.inc()
        return output
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {ex}") from ex


@app.get("/evaluate")
def evaluate():
    try:
        return evaluate_trained_model()
    except FileNotFoundError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {ex}") from ex
