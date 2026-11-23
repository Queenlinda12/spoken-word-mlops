"""Model creation, training, testing and RETRAINING for spoken-word classification.

Model: MobileNetV2 (ImageNet pre-trained) on 128x128 log-mel spectrograms + regularised head.
Optimisation: transfer learning, 2-stage fine-tuning, dropout, L2, Adam, early stopping,
ReduceLROnPlateau, class weights, audio augmentation (shift, gain, speed, pitch, noise) + SpecAugment.
"""
import json
import os
import shutil
import threading
import time

import numpy as np

from src import preprocessing as pp

MODEL_PATH = pp.MODELS_DIR / "spoken_words_mobilenetv2.h5"
PREV_MODEL_PATH = pp.MODELS_DIR / "spoken_words_mobilenetv2_prev.h5"
CLASSES_PATH = pp.MODELS_DIR / "class_names.json"
METRICS_PATH = pp.MODELS_DIR / "metrics.json"
STATE_PATH = pp.MODELS_DIR / "training_state.json"

WEIGHTS = os.getenv("PRETRAINED_WEIGHTS", "imagenet")      # "none" only for quick tests
WEIGHTS = None if WEIGHTS.lower() == "none" else WEIGHTS
ARCH = os.getenv("MODEL_ARCH", "mobilenet")             # "mobilenet" (transfer learning) or "cnn" (from scratch)
AUG_FACTOR = int(os.getenv("AUG_FACTOR", 3))               # augmented copies per training clip
BATCH = 32

_job_lock = threading.Lock()


# ---- shared training state (a file, so several API containers see the same status) -----------
def get_state():
    try:
        state = json.loads(STATE_PATH.read_text())
    except Exception:
        return {"status": "idle"}
    if state.get("status") == "training" and time.time() - state.get("updated", 0) > 6 * 3600:
        return {"status": "idle"}
    return state


def set_state(**kw):
    pp.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    state = {**get_state(), **kw, "updated": time.time()}
    STATE_PATH.write_text(json.dumps(state))
    return state


def is_training():
    return get_state().get("status") == "training"


# ---- model ------------------------------------------------------------------------------------
def build_cnn(num_classes):
    """Small CNN trained from scratch (fallback / baseline): 4 conv blocks + BatchNorm + dropout + L2."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    inputs = tf.keras.Input(shape=(*pp.SPEC_SHAPE, 1))
    x = inputs
    for f in (16, 32, 64, 96):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="spoken_words_cnn")


def build_model(num_classes, weights=WEIGHTS):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    if ARCH == "cnn":
        return build_cnn(num_classes)

    base = tf.keras.applications.MobileNetV2(input_shape=(*pp.SPEC_SHAPE, 3), include_top=False, weights=weights)
    base.trainable = False                                   # stage 1: only the new head learns
    inputs = tf.keras.Input(shape=(*pp.SPEC_SHAPE, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])      # 1-channel spectrogram -> 3 channels
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="spoken_words_mobilenetv2")


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def unfreeze_top(model, n_layers=30):
    """Fine-tune the last n layers of MobileNetV2 (BatchNorm stays frozen)."""
    import tensorflow as tf
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base is None:                                           # plain CNN: everything already trains
        return
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False


def _callbacks(patience=3):
    import tensorflow as tf
    return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience - 1),
                                                 min_lr=1e-7)]


def _class_weights(y, n_classes):
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return {i: float(x) for i, x in enumerate(w)}


def _to_ds(X, y, shuffle):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=1)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def prepare_data(class_names):
    """Split the training folder into train/validation (by SPEAKER when >= 5 speakers exist),
    augment ONLY the training clips, return tf.data datasets + labels."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    files, labels = pp.list_labelled(pp.TRAIN_DIR, class_names)
    if len(files) < 10:
        raise ValueError("Not enough training audio (need at least 10 clips)")
    groups = np.array([pp.speaker_of(f) for f in files])
    labels = np.array(labels)
    idx = np.arange(len(files))
    if len(set(groups) - {"unknown"}) >= 5:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7).split(idx, labels, groups))
    else:
        tr, va = train_test_split(idx, test_size=0.2, stratify=labels, random_state=7)
    Xtr, ytr = pp.arrays_from_files([files[i] for i in tr], labels[tr], augment_factor=AUG_FACTOR)
    Xva, yva = pp.arrays_from_files([files[i] for i in va], labels[va], augment_factor=0)
    return _to_ds(Xtr, ytr, True), _to_ds(Xva, yva, False), ytr


# ---- evaluation (>= 4 metrics) ----------------------------------------------------------------
def evaluate_model(model, class_names):
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    files, labels = pp.list_labelled(pp.TEST_DIR, class_names)
    Xte, y_true = pp.arrays_from_files(files, labels)
    probs = model.predict(Xte, verbose=0)
    y_pred = probs.argmax(axis=1)
    loss, acc = model.evaluate(Xte, y_true, verbose=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "loss": float(loss), "precision": float(p), "recall": float(r), "f1": float(f1),
            "macro_precision": float(pm), "macro_recall": float(rm), "macro_f1": float(fm),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist(),
            "classification_report": classification_report(y_true, y_pred, target_names=class_names,
                                                            output_dict=True, zero_division=0),
            "n_test_clips": int(len(y_true))}


def _save(model, class_names, metrics, kind):
    pp.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, PREV_MODEL_PATH)             # keep the previous version for rollback
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(class_names))
    old = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics = {**metrics, "version": old.get("version", 0) + 1, "trained_at": pp.now(), "kind": kind}
    METRICS_PATH.write_text(json.dumps(metrics))
    return metrics


# ---- training and RETRAINING ------------------------------------------------------------------
def train(epochs=15, finetune_epochs=10):
    """First model: frozen MobileNetV2 -> fine-tune top layers -> evaluate on data/test -> save."""
    class_names = pp.get_class_names()
    if len(class_names) < 2:
        raise ValueError("Need at least 2 word folders in data/train/")
    train_ds, val_ds, ytr = prepare_data(class_names)
    cw = _class_weights(ytr, len(class_names))
    model = build_model(len(class_names))
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=_callbacks(4),
                   verbose=2)
    hist = {k: list(v) for k, v in h1.history.items()}
    if ARCH != "cnn":                                          # stage 2: fine-tune the top of MobileNetV2
        unfreeze_top(model, 30)
        compile_model(model, 1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, class_weight=cw,
                       callbacks=_callbacks(3), verbose=2)
        hist = {k: hist[k] + list(h2.history[k]) for k in hist}
    metrics = _save(model, class_names, evaluate_model(model, class_names), "train")
    metrics["history"] = {k: [float(x) for x in v] for k, v in hist.items()}
    METRICS_PATH.write_text(json.dumps(metrics))
    return metrics


def retrain(epochs=8):
    """RETRAINING: load the SAVED model as the pre-trained model, then fine-tune on data/train
    (which now includes newly uploaded recordings) with a low learning rate."""
    import tensorflow as tf
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Train a first model before retraining.")
    class_names = json.loads(CLASSES_PATH.read_text())
    if class_names != pp.get_class_names():
        raise ValueError("Word folders changed since the model was trained; train from scratch instead.")
    train_ds, val_ds, ytr = prepare_data(class_names)
    cw = _class_weights(ytr, len(class_names))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)      # <- custom model as pre-trained model
    unfreeze_top(model, 30)
    compile_model(model, 1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=_callbacks(3), verbose=2)
    metrics = _save(model, class_names, evaluate_model(model, class_names), "retrain")
    pp.mark_uploads_used()
    return metrics


def start_job(kind, epochs=None):
    """Run train / retrain in a background thread. False if a job is already running."""
    if not _job_lock.acquire(blocking=False):
        return False
    if is_training():
        _job_lock.release()
        return False
    set_state(status="training", kind=kind, started_at=pp.now(), error=None)

    def run():
        started = pp.now()
        n = sum(len(pp.list_audio(pp.TRAIN_DIR / c)) for c in pp.get_class_names())
        try:
            m = train(epochs or 15) if kind == "train" else retrain(epochs or 8)
            pp.log_run(kind, "success", m["version"], m["accuracy"], m["f1"], n, started=started)
            set_state(status="idle", last_finished=pp.now(), error=None)
            try:
                from src import prediction
                prediction.reload_model()
            except Exception:
                pass
        except Exception as e:  # noqa
            pp.log_run(kind, "failed", note=str(e)[:300], started=started)
            set_state(status="idle", error=str(e)[:300])
        finally:
            _job_lock.release()

    threading.Thread(target=run, daemon=True).start()
    return True


def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None
