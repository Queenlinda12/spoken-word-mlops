"""Streamlit dashboard: uptime, voice prediction, data visualisations, record / upload data, train / retrain.

Run:  API_URL=http://localhost:8000 streamlit run ui/app.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Spoken Word Recognition", page_icon="🎙️", layout="wide")


def api_get(path, timeout=15):
    t0 = time.time()
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json(), (time.time() - t0) * 1000


def fmt_uptime(sec):
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{d}d {h}h {m}m {s}s" if d else f"{h}h {m}m {s}s"


st.sidebar.title("🎙️ Spoken Words")
st.sidebar.caption("Voice recognition for accessible learning")
page = st.sidebar.radio("Go to", ["Dashboard & uptime", "Predict a word", "Data visualisations",
                                  "Record / upload data", "Train & retrain"])
st.sidebar.caption(f"API: {API_URL}")

try:
    status, latency = api_get("/status")
    api_up = True
except Exception:
    status, latency, api_up = {}, None, False

if "latency_log" not in st.session_state:
    st.session_state.latency_log = []
if latency is not None:
    st.session_state.latency_log = (st.session_state.latency_log + [round(latency, 1)])[-50:]


def audio_from_widgets(key, label="Record the word"):
    """Returns (filename, bytes) from the microphone widget or an uploaded file, or None."""
    tab1, tab2 = st.tabs(["🎤 Record", "📁 Upload file"])
    with tab1:
        rec = st.audio_input(label, key=f"{key}_rec") if hasattr(st, "audio_input") else None
    with tab2:
        up = st.file_uploader("wav / flac / ogg", type=["wav", "flac", "ogg"], key=f"{key}_up")
    if rec is not None:
        return "recording.wav", rec.getvalue()
    if up is not None:
        return up.name, up.getvalue()
    return None


# ------------------------------------------------------------------ Dashboard
if page == "Dashboard & uptime":
    st.title("Model dashboard")
    if not api_up:
        st.error(f"API is DOWN or unreachable at {API_URL}")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API status", "UP ✅")
    c2.metric("Up-time", fmt_uptime(status["uptime_seconds"]))
    c3.metric("Model version", status.get("model_version") or "none")
    c4.metric("Status latency", f"{latency:.0f} ms")
    st.caption(f"Model trained at: {status.get('trained_at') or 'n/a'}  |  model loaded: {status['model_loaded']}  "
               f"|  words: {', '.join(status['classes'])}")
    m = status.get("metrics") or {}
    if m.get("accuracy") is not None:
        st.subheader("Test-set performance (speakers the model never heard in training)")
        cols = st.columns(5)
        for col, (k, label) in zip(cols, [("accuracy", "Accuracy"), ("loss", "Loss"), ("precision", "Precision"),
                                          ("recall", "Recall"), ("f1", "F1 score")]):
            col.metric(label, f"{m[k]:.3f}")
        cm, classes = status.get("confusion_matrix"), status["classes"]
        if cm:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.imshow(np.array(cm), cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=40)
            ax.set_yticks(range(len(classes)), classes)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, cm[i][j], ha="center", va="center")
            st.pyplot(fig)
        hist = status.get("history")
        if hist and "val_accuracy" in hist:
            st.subheader("Training curves")
            st.line_chart(pd.DataFrame({"train acc": hist["accuracy"], "val acc": hist["val_accuracy"]}))
    else:
        st.info("No model yet. Go to *Train & retrain* and press **Train first model**.")
    st.subheader("API response time (this session)")
    st.line_chart(pd.DataFrame({"ms": st.session_state.latency_log}))
    runs, _ = api_get("/runs")
    if runs:
        st.subheader("Training history (from the database)")
        st.dataframe(pd.DataFrame(runs)[["id", "kind", "status", "version", "accuracy", "f1",
                                         "n_train_clips", "finished_at", "note"]])

# ------------------------------------------------------------------ Predict
elif page == "Predict a word":
    st.title("Say a word, the model tells you which one it heard")
    st.caption("Words it knows: " + (", ".join(status.get("classes", [])) if api_up else "API down"))
    clip = audio_from_widgets("pred")
    if clip:
        st.audio(clip[1])
        if st.button("Predict", type="primary"):
            with st.spinner("Listening..."):
                t0 = time.time()
                r = requests.post(f"{API_URL}/predict", files={"file": (clip[0], clip[1], "audio/wav")}, timeout=60)
            if r.ok:
                out = r.json()
                st.success(f"I heard: **{out['label']}**  (confidence {out['confidence']:.1%})")
                st.caption(f"Response time: {(time.time() - t0) * 1000:.0f} ms")
                st.bar_chart(pd.Series(out["probabilities"], name="probability"))
            else:
                st.error(r.json().get("detail", r.text))

# ------------------------------------------------------------------ Visualisations
elif page == "Data visualisations":
    st.title("What does the voice data tell us?")
    try:
        ins, _ = api_get("/insights", timeout=180)
    except Exception as e:
        st.warning(f"No dataset available yet ({e}).")
        st.stop()
    st.subheader("1. Class balance and number of speakers")
    st.bar_chart(pd.DataFrame({"clips": ins["counts"], "speakers": ins["speakers"]}))
    titles = {"voiced_duration": "2. How long each word takes to say (seconds)",
              "spectral_centroid": "3. Spectral centroid: how 'bright' the sound is (Hz)",
              "loudness_db": "4. Loudness per class (dB)"}
    for key, title in titles.items():
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.boxplot([ins["features"][key][c] for c in ins["classes"]], tick_labels=ins["classes"])
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
    st.subheader("5. Average spectrogram of each word (what the model actually 'sees')")
    cols = st.columns(min(4, max(1, len(ins["classes"]))))
    for i, c in enumerate(ins["classes"]):
        if c in ins["mean_spectrograms"]:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(np.array(ins["mean_spectrograms"][c]), origin="lower", aspect="auto", cmap="magma")
            ax.set_title(c)
            ax.set_xlabel("time")
            ax.set_ylabel("frequency")
            cols[i % len(cols)].pyplot(fig)
    st.subheader("Interpretation: what story does it tell?")
    for note in ins["interpretations"]:
        st.markdown(f"- {note}")

# ------------------------------------------------------------------ Record / upload data
elif page == "Record / upload data":
    st.title("Add voice data (bulk upload or record)")
    if not api_up:
        st.error("API unreachable")
        st.stop()
    st.write(f"Pending (not yet used) clips: **{status['pending_uploads']}**. Automatic retraining triggers at "
             f"**{status['retrain_threshold']}** new clips ({'on' if status['auto_retrain'] else 'off'}).")
    speaker = st.text_input("Speaker name or nickname (used to keep voices separate)", value="anon")
    label = st.selectbox("Which word are you saying / uploading?", status["classes"] or ["(no words yet)"])
    st.subheader("A. Bulk upload many files of this word")
    files = st.file_uploader("Select many .wav files", type=["wav", "flac", "ogg"], accept_multiple_files=True,
                             key="bulk")
    if st.button("Upload files", disabled=not files):
        r = requests.post(f"{API_URL}/upload", data={"label": label, "speaker": speaker},
                          files=[("files", (f.name, f.getvalue(), "audio/wav")) for f in files], timeout=300)
        if r.ok:
            o = r.json()
            st.success(f"Saved {o['saved']} clips, rejected {len(o['rejected'])}. Pending: {o['pending_uploads']}. "
                       f"{'Automatic retraining triggered!' if o['auto_retrain_triggered'] else ''}")
        else:
            st.error(r.json().get("detail", r.text))
    st.subheader("B. Record clips right here (works on a phone too)")
    rec = st.audio_input(f"Say: {label}", key=f"rec_{label}") if hasattr(st, "audio_input") else None
    if rec is not None and st.button("Save this recording"):
        r = requests.post(f"{API_URL}/upload", data={"label": label, "speaker": speaker},
                          files=[("files", ("rec.wav", rec.getvalue(), "audio/wav"))], timeout=60)
        st.success("Saved ✅" if r.ok and r.json()["saved"] else r.text)

# ------------------------------------------------------------------ Train & retrain
else:
    st.title("Train & retrain")
    if not api_up:
        st.error("API unreachable")
        st.stop()
    training = status["training"]
    st.write(f"Pending uploads: **{status['pending_uploads']}** (auto-retrain at **{status['retrain_threshold']}**).")
    if training.get("status") == "training":
        st.warning(f"⏳ {training.get('kind')} in progress (started {training.get('started_at')}). "
                   "This page refreshes automatically.")
    elif training.get("error"):
        st.error(f"Last job failed: {training['error']}")
    c1, c2 = st.columns(2)
    busy = training.get("status") == "training"
    if c1.button("Train first model (from scratch)", disabled=busy):
        r = requests.post(f"{API_URL}/train", timeout=30)
        st.info(r.json().get("message") or r.json().get("detail"))
    if c2.button("🔁 Retrain with the uploaded data", type="primary", disabled=busy):
        r = requests.post(f"{API_URL}/retrain", timeout=30)
        st.info(r.json().get("message") or r.json().get("detail"))
    if busy:
        time.sleep(5)
        st.rerun()
