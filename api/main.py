
import io
import os
from pathlib import Path

import zipfile
import shutil

from pathlib import PurePosixPath
import time
from datetime import datetime, timezone

from monitoring.logger import (
    log_prediction,
    read_prediction_logs
)
from fastapi import BackgroundTasks
from src.retraining import retrain_model, DATA_DIR

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from src.prediction import (
    load_prediction_model,
    predict_image,
    load_metadata
)

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = os.getenv("GEOVISION_MODEL", "mobilenet")

# Load metadata and model once when the API starts
metadata = load_metadata()
class_names = metadata["image_classes"]

model = load_prediction_model(MODEL_NAME)

app = FastAPI(
    title="GeoVision ML API",
    description="Satellite image land-use classification API",
    version="1.0.0"
)
# Monitoring information
START_TIME = time.time()

METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0
}

REQUEST_TIMES = []

@app.middleware("http")
async def monitor_requests(request, call_next):

    start = time.perf_counter()

    METRICS["total_requests"] += 1

    try:
        response = await call_next(request)

        if response.status_code < 400:
            METRICS["successful_requests"] += 1
        else:
            METRICS["failed_requests"] += 1

        return response

    except Exception:
        METRICS["failed_requests"] += 1
        raise

    finally:
        elapsed = (time.perf_counter() - start) * 1000

        REQUEST_TIMES.append(elapsed)

        # Keep only the most recent 1,000 measurements
        if len(REQUEST_TIMES) > 1000:
            REQUEST_TIMES.pop(0)
            
            
retraining_status = {
    "status": "idle",
    "result": None,
    "error": None
}


@app.get("/")
def home():
    return {
        "message": "Welcome to GeoVision ML API",
        "project": "Satellite Image Classification",
        "available_classes": class_names
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check whether the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # Limit uploads to 10 MB
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prediction_start = time.perf_counter()

        result = predict_image(
            image=image,
            model=model,
            model_name=MODEL_NAME,
            class_names=class_names
        )

        prediction_time = (
            time.perf_counter() - prediction_start
        ) * 1000

        log_prediction(
            filename=file.filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            response_time=prediction_time
        )

        return {
            "filename": file.filename,
            "model": MODEL_NAME,
            **result
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


@app.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file."
        )

    contents = await file.read()

    if len(contents) > 200 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="ZIP file must not exceed 200 MB."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract into a temporary directory first
    temp_dir = DATA_DIR.parent / "upload_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    try:
        import io

        with zipfile.ZipFile(io.BytesIO(contents)) as archive:

            # Basic protection against unsafe ZIP paths
            for item in archive.infolist():
                path = PurePosixPath(item.filename)

                if path.is_absolute() or ".." in path.parts:
                    raise HTTPException(
                        status_code=400,
                        detail="ZIP contains an unsafe file path."
                    )

                if item.file_size > 20 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail="An individual file is too large."
                    )

            archive.extractall(temp_dir)

        # Support ZIPs containing either the class folders
        # directly or inside one parent folder.
        source_dir = temp_dir

        entries = list(temp_dir.iterdir())

        if (
            len(entries) == 1
            and entries[0].is_dir()
        ):
            source_dir = entries[0]

        metadata = load_metadata()
        expected_classes = metadata["image_classes"]

        found_classes = {
            folder.name
            for folder in source_dir.iterdir()
            if folder.is_dir()
        }

        missing = set(expected_classes) - found_classes

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing class folders: {sorted(missing)}"
            )

        # Replace previous uploaded data
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

        shutil.copytree(source_dir, DATA_DIR)

        return {
            "message": "Training data uploaded successfully",
            "classes": expected_classes
        }

    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid ZIP archive."
        )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def run_retraining():
    global retraining_status

    try:
        retraining_status = {
            "status": "running",
            "result": None,
            "error": None
        }

        result = retrain_model(epochs=5)

        retraining_status = {
            "status": "completed",
            "result": result,
            "error": None
        }

    except Exception as error:
        retraining_status = {
            "status": "failed",
            "result": None,
            "error": str(error)
        }


@app.post("/retrain")
def start_retraining(background_tasks: BackgroundTasks):

    if retraining_status["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Retraining is already running."
        )

    if not DATA_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Upload labelled training data first."
        )

    background_tasks.add_task(run_retraining)

    return {
        "message": "Retraining started",
        "status": "running"
    }


@app.get("/retrain/status")
def get_retraining_status():
    return retraining_status

@app.get("/metrics")
def get_metrics():

    uptime_seconds = time.time() - START_TIME

    average_response_time = (
        sum(REQUEST_TIMES) / len(REQUEST_TIMES)
        if REQUEST_TIMES else 0
    )

    prediction_logs = read_prediction_logs()

    prediction_times = [
        item["response_time_ms"]
        for item in prediction_logs
    ]

    average_prediction_time = (
        sum(prediction_times) / len(prediction_times)
        if prediction_times else 0
    )

    return {
        "uptime_seconds": round(uptime_seconds, 2),
        "total_requests": METRICS["total_requests"],
        "successful_requests": METRICS["successful_requests"],
        "failed_requests": METRICS["failed_requests"],
        "average_request_time_ms": round(
            average_response_time, 2
        ),
        "total_predictions": len(prediction_logs),
        "average_prediction_time_ms": round(
            average_prediction_time, 2
        ),
        "model_name": MODEL_NAME
    }