"""FastAPI service: prediction, bulk upload, train / retrain triggers, status, insights.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000      (Swagger UI at /docs)
"""
import os
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import model as mdl
from src import prediction
from src import preprocessing as pp

START_TIME = time.time()
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 30))   # new clips that auto-trigger retraining
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"

app = FastAPI(title="Spoken-Word Classifier - MLOps API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pp.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    """Model up-time + model / training information for the dashboard."""
    metrics = mdl.load_metrics() or {}
    return {
        "status": "up",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": prediction.model_loaded(),
        "model_version": metrics.get("version"),
        "trained_at": metrics.get("trained_at"),
        "metrics": {k: metrics.get(k) for k in ("accuracy", "loss", "precision", "recall", "f1",
                                                "macro_f1", "n_test_clips")},
        "confusion_matrix": metrics.get("confusion_matrix"),
        "history": metrics.get("history"),
        "classes": pp.get_class_names(),
        "training": mdl.get_state(),
        "pending_uploads": pp.pending_upload_count(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "auto_retrain": AUTO_RETRAIN,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload ONE audio clip (wav / flac / ogg) of a spoken word, get the predicted word."""
    content = await file.read()
    if not content or len(content) > pp.MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Empty file or larger than 5 MB")
    try:
        return prediction.predict_audio(content)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/upload")
async def upload(label: str = Form(...), speaker: str = Form("anon"), files: List[UploadFile] = File(...)):
    """Bulk upload of labelled recordings of ONE word. Saved to data/train/<label>/ and logged in the
    database. If enough new clips are pending, retraining is triggered automatically."""
    payload = [(f.filename or "clip.wav", await f.read()) for f in files]
    try:
        result = pp.save_uploaded_files(payload, label, speaker)
    except ValueError as e:
        raise HTTPException(400, str(e))
    pending = pp.pending_upload_count()
    triggered = False
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD and mdl.MODEL_PATH.exists():
        triggered = mdl.start_job("retrain")
    return {**result, "pending_uploads": pending, "auto_retrain_triggered": triggered}


@app.post("/train")
def train(epochs: int = 15):
    """Train the first model from data/train (runs in the background)."""
    if not mdl.start_job("train", epochs):
        raise HTTPException(409, "A training job is already running")
    return {"message": "Training started", "state": mdl.get_state()}


@app.post("/retrain")
def retrain(epochs: int = 8):
    """Manual retraining trigger. Uses the saved model as the pre-trained model."""
    if not mdl.MODEL_PATH.exists():
        raise HTTPException(400, "No model to retrain. Call /train first.")
    if not mdl.start_job("retrain", epochs):
        raise HTTPException(409, "A training job is already running")
    return {"message": "Retraining started", "state": mdl.get_state()}


@app.get("/insights")
def insights():
    """Data for the visualisation page (class balance, 3 acoustic features, spectrograms, interpretations)."""
    if not pp.get_class_names():
        raise HTTPException(404, "No dataset found in data/train/")
    return pp.dataset_insights()


@app.get("/runs")
def runs():
    """History of training / retraining runs stored in the database."""
    return pp.list_runs()
