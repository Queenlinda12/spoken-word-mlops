"""Flood the API with audio prediction requests.

Headless example (60 s, 100 users, spawn 10/s):
    locust -f locustfile.py --headless -u 100 -r 10 -t 60s --host http://localhost:8000 --csv results/containers_1
"""
import glob
import io
import random

import numpy as np
import soundfile as sf
from locust import HttpUser, between, task


def _fallback_clip():
    sr = 16000
    t = np.arange(int(sr * 0.8)) / sr
    y = (0.5 * np.sin(2 * np.pi * 500 * t) * np.hanning(len(t))).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


SAMPLES = [open(p, "rb").read() for p in glob.glob("data/test/*/*.wav")[:30]] or [_fallback_clip()]


class ApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        self.client.post("/predict", files={"file": ("word.wav", random.choice(SAMPLES), "audio/wav")},
                         name="POST /predict")

    @task(2)
    def status(self):
        self.client.get("/status", name="GET /status")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
