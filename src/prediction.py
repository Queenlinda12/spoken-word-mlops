"""Single-clip prediction. The model reloads automatically whenever the .h5 file changes,
so every API container picks up a retrained model without a restart."""
import json
import threading

import numpy as np

from src import preprocessing as pp
from src.model import CLASSES_PATH, MODEL_PATH

_lock = threading.Lock()
_cache = {"model": None, "mtime": None, "classes": None}


def reload_model():
    import tensorflow as tf
    with _lock:
        if not MODEL_PATH.exists():
            _cache.update(model=None, mtime=None, classes=None)
            return False
        _cache["model"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _cache["classes"] = json.loads(CLASSES_PATH.read_text())
        _cache["mtime"] = MODEL_PATH.stat().st_mtime
        return True


def _ensure_loaded():
    if not MODEL_PATH.exists():
        return False
    if _cache["model"] is None or _cache["mtime"] != MODEL_PATH.stat().st_mtime:
        return reload_model()
    return True


def model_loaded():
    return _ensure_loaded()


def predict_audio(audio_bytes):
    """bytes of a wav/flac/ogg clip -> {"label", "confidence", "probabilities"}"""
    if not _ensure_loaded():
        raise RuntimeError("No trained model available yet. Train one first.")
    pp.validate_audio(audio_bytes)                       # raises ValueError for unusable audio
    x = pp.audio_to_spec(audio_bytes)[None, ...]
    probs = _cache["model"].predict(x, verbose=0)[0]
    classes = _cache["classes"]
    i = int(np.argmax(probs))
    return {"label": classes[i], "confidence": float(probs[i]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)}}


if __name__ == "__main__":  # python -m src.prediction path/to/word.wav
    import sys
    print(predict_audio(open(sys.argv[1], "rb").read()))
