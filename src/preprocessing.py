"""Audio data acquisition + preprocessing + upload bookkeeping.

Dataset layout (one folder per spoken word / class):

    data/train/<word>/<speaker>__<id>.wav
    data/test/<word>/<speaker>__<id>.wav

Every clip becomes a 128x128 log-mel spectrogram "image" scaled to [-1, 1].
"""
import io
import os
import random
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DB_PATH = DATA_DIR / "app.db"

SR = 16000                 # sample rate
DURATION = 2.0             # seconds kept per clip
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP, N_MELS = 1024, 250, 128   # 2 s * 16000 / 250 = 128 frames -> 128 x 128 spectrogram
SPEC_SHAPE = (N_MELS, 128)
ALLOWED_EXT = {".wav", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MIN_VOICED_SEC = 0.15


# --------------------------------------------------------------------------
# 1. DATA ACQUISITION
# --------------------------------------------------------------------------
def list_audio(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in ALLOWED_EXT)


def speaker_of(path):
    """Files are named <speaker>__<id>.wav; anything else counts as 'unknown'."""
    name = Path(path).stem
    return name.split("__")[0] if "__" in name else "unknown"


def split_raw_dataset(raw_dir, test_ratio=0.2, seed=42, clear=True):
    """raw_dir/<word>/*.wav  ->  data/train/<word>, data/test/<word>.

    If the clips carry >= 4 distinct speaker names (<speaker>__<id>.wav), whole SPEAKERS are held out
    for the test set (speaker-independent evaluation = honest estimate for new voices).
    Otherwise it falls back to a random split per word.
    """
    raw_dir = Path(raw_dir)
    classes = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in {raw_dir}")
    if clear:
        for d in (TRAIN_DIR, TEST_DIR):
            if d.exists():
                shutil.rmtree(d)
    rng = random.Random(seed)
    files_by_class = {c: list_audio(raw_dir / c) for c in classes}
    speakers = sorted({speaker_of(f) for fs in files_by_class.values() for f in fs} - {"unknown"})
    by_speaker = len(speakers) >= 4
    test_speakers = set(rng.sample(speakers, max(1, round(len(speakers) * test_ratio)))) if by_speaker else set()

    summary = {"mode": "speaker-independent" if by_speaker else "random",
               "test_speakers": sorted(test_speakers), "classes": {}}
    for c, files in files_by_class.items():
        if by_speaker:
            test = [f for f in files if speaker_of(f) in test_speakers]
            train = [f for f in files if speaker_of(f) not in test_speakers]
        else:
            files = files[:]
            rng.shuffle(files)
            n_test = max(1, int(len(files) * test_ratio))
            test, train = files[:n_test], files[n_test:]
        for split, items in (("train", train), ("test", test)):
            out = (TRAIN_DIR if split == "train" else TEST_DIR) / c
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f"{speaker_of(f)}__{uuid.uuid4().hex[:8]}{f.suffix.lower()}")
        summary["classes"][c] = {"train": len(train), "test": len(test)}
    return summary


def get_class_names():
    if not TRAIN_DIR.exists():
        return []
    return sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())


# --------------------------------------------------------------------------
# 2. PREPROCESSING  (waveform -> log-mel spectrogram in [-1, 1])
# --------------------------------------------------------------------------
def load_waveform(source):
    """path | bytes | file-like -> mono float32 waveform at 16 kHz."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, sr = sf.read(source, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)                       # stereo -> mono
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def trim_and_fit(y):
    """Remove leading/trailing silence, centre the word in a fixed 2 s window."""
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) == 0:
        yt = y
    if len(yt) >= N_SAMPLES:
        start = (len(yt) - N_SAMPLES) // 2
        return yt[start:start + N_SAMPLES]
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    start = (N_SAMPLES - len(yt)) // 2
    out[start:start + len(yt)] = yt
    return out


def waveform_to_spec(y):
    """fixed-length waveform -> (128, 128, 1) float32 log-mel spectrogram scaled to [-1, 1]."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=SR // 2)
    db = librosa.power_to_db(S, ref=np.max, top_db=80.0)       # [-80, 0]; loudness-normalised per clip
    spec = db / 40.0 + 1.0                                      # [-1, 1]
    if spec.shape[1] < SPEC_SHAPE[1]:
        spec = np.pad(spec, ((0, 0), (0, SPEC_SHAPE[1] - spec.shape[1])), constant_values=-1.0)
    return spec[:, :SPEC_SHAPE[1], None].astype(np.float32)


def audio_to_spec(source):
    return waveform_to_spec(trim_and_fit(load_waveform(source)))


def validate_audio(content):
    """Raises ValueError if the clip is unusable (not audio, too short, silent)."""
    try:
        y = load_waveform(content)
    except Exception:
        raise ValueError("not a readable audio file (use wav / flac / ogg)")
    if len(y) < SR * MIN_VOICED_SEC or float(np.abs(y).max()) < 1e-3:
        raise ValueError("clip is too short or silent")
    return y


# ---- augmentation (training clips only) -------------------------------------
def augment_waveform(y, rng):
    y = y.copy()
    y = np.roll(y, rng.randint(-int(0.15 * SR), int(0.15 * SR)))                    # time shift
    y = y * rng.uniform(0.6, 1.4)                                                    # gain
    if rng.random() < 0.5:
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))              # speed
    if rng.random() < 0.5:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.uniform(-1.5, 1.5))    # voice pitch
    y = y + np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(0.0, 0.01), len(y))
    return y.astype(np.float32)


def spec_augment(spec, rng):
    """SpecAugment: mask random frequency bands and time steps."""
    s = spec.copy()
    f = rng.randint(0, 12); f0 = rng.randint(0, N_MELS - f); s[f0:f0 + f] = -1.0
    t = rng.randint(0, 12); t0 = rng.randint(0, SPEC_SHAPE[1] - t); s[:, t0:t0 + t] = -1.0
    return s


def arrays_from_files(files, labels, augment_factor=0, seed=0):
    """files + integer labels -> X (N,128,128,1), y (N,).
    augment_factor=k adds k augmented copies of each clip (use for TRAIN clips only)."""
    rng = random.Random(seed)
    X, y = [], []
    for f, lab in zip(files, labels):
        try:
            wav = load_waveform(f)
        except Exception:
            continue
        X.append(waveform_to_spec(trim_and_fit(wav))); y.append(lab)
        for _ in range(augment_factor):
            aug = waveform_to_spec(trim_and_fit(augment_waveform(wav, rng)))
            X.append(spec_augment(aug, rng)); y.append(lab)
    if not X:
        raise ValueError("No usable audio found")
    return np.stack(X), np.array(y)


def list_labelled(directory, class_names):
    files, labels = [], []
    for i, c in enumerate(class_names):
        for f in list_audio(Path(directory) / c):
            files.append(f); labels.append(i)
    return files, labels


# --------------------------------------------------------------------------
# 3. UPLOADS + DATABASE (SQLite)
# --------------------------------------------------------------------------
def _conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS uploads(
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, speaker TEXT, path TEXT,
            uploaded_at TEXT, used_in_training INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS training_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, started_at TEXT, finished_at TEXT,
            status TEXT, version INTEGER, accuracy REAL, f1 REAL, n_train_clips INTEGER, note TEXT)""")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(s):
    return "".join(ch for ch in (s or "anon").lower() if ch.isalnum())[:20] or "anon"


def save_uploaded_files(files, label, speaker="anon"):
    """files: list of (filename, bytes). Validates, saves to data/train/<label>/, logs in the DB."""
    init_db()
    if label not in get_class_names():
        raise ValueError(f"Unknown label '{label}'. Valid labels: {get_class_names()}")
    speaker = safe_name(speaker)
    out_dir = TRAIN_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, []
    with _conn() as c:
        for name, content in files:
            try:
                if Path(name).suffix.lower() not in ALLOWED_EXT or len(content) > MAX_UPLOAD_BYTES:
                    raise ValueError("bad type or too large")
                validate_audio(content)
            except Exception:
                rejected.append(name)
                continue
            dest = out_dir / f"{speaker}__up{uuid.uuid4().hex[:8]}{Path(name).suffix.lower()}"
            dest.write_bytes(content)
            c.execute("INSERT INTO uploads(filename,label,speaker,path,uploaded_at) VALUES (?,?,?,?,?)",
                      (name, label, speaker, str(dest), now()))
            saved += 1
    return {"saved": saved, "rejected": rejected}


def pending_upload_count():
    init_db()
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM uploads WHERE used_in_training=0").fetchone()[0]


def mark_uploads_used():
    init_db()
    with _conn() as c:
        c.execute("UPDATE uploads SET used_in_training=1 WHERE used_in_training=0")


def log_run(kind, status, version=None, accuracy=None, f1=None, n_train=None, note="", started=None):
    init_db()
    with _conn() as c:
        c.execute("""INSERT INTO training_runs(kind,started_at,finished_at,status,version,accuracy,f1,n_train_clips,note)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (kind, started or now(), now(), status, version, accuracy, f1, n_train, note))


def list_runs(limit=20):
    init_db()
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))]


# --------------------------------------------------------------------------
# 4. DATA INSIGHTS (dashboard)
# --------------------------------------------------------------------------
_insight_cache = {"key": None, "value": None}


def dataset_insights(max_per_class=60):
    """Class balance, speakers, 3 acoustic features per class, mean spectrograms + interpretations."""
    classes = get_class_names()
    files = {c: list_audio(TRAIN_DIR / c) for c in classes}
    counts = {c: len(v) for c, v in files.items()}
    key = tuple(counts.items())
    if _insight_cache["key"] == key:
        return _insight_cache["value"]

    feats = {"voiced_duration": {}, "loudness_db": {}, "spectral_centroid": {}}
    mean_specs, speakers = {}, {}
    for c in classes:
        sample = random.Random(0).sample(files[c], min(len(files[c]), max_per_class))
        dur, loud, cen, specs = [], [], [], []
        speakers[c] = len({speaker_of(f) for f in files[c]})
        for f in sample:
            try:
                y = load_waveform(f)
            except Exception:
                continue
            yt, _ = librosa.effects.trim(y, top_db=30)
            dur.append(len(yt) / SR)
            loud.append(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)))
            cen.append(float(librosa.feature.spectral_centroid(y=yt if len(yt) > 512 else y, sr=SR).mean()))
            specs.append(waveform_to_spec(trim_and_fit(y))[:, :, 0])
        feats["voiced_duration"][c], feats["loudness_db"][c], feats["spectral_centroid"][c] = dur, loud, cen
        if specs:
            m = np.mean(specs, axis=0)
            mean_specs[c] = m[::4, ::4].round(3).tolist()   # 32 x 32 for plotting

    mean = lambda v: float(np.mean(v)) if v else 0.0
    summary = {f: {c: mean(v) for c, v in d.items()} for f, d in feats.items()}
    notes = []
    if counts and min(counts.values()) > 0:
        hi, lo = max(counts, key=counts.get), min(counts, key=counts.get)
        ratio = counts[hi] / counts[lo]
        notes.append(f"**Class balance:** '{hi}' has {counts[hi]} clips, '{lo}' has {counts[lo]} ({ratio:.1f}x). "
                     + ("Imbalanced: we use class weights and report per-class precision/recall/F1."
                        if ratio > 1.5 else "Roughly balanced, so accuracy is a fair headline metric.")
                     + f" Speakers per word: {min(speakers.values())} to {max(speakers.values())}; "
                       "more distinct voices means a model that generalises to new people.")
    desc = {"voiced_duration": ("Spoken duration (seconds, silence trimmed)",
                                "longer words such as multi-syllable numbers take more time to say"),
            "spectral_centroid": ("Spectral centroid (Hz, the 'brightness' of the sound)",
                                  "fricatives like 's' and 'ch' push energy to higher frequencies than vowels do"),
            "loudness_db": ("Loudness (dB)", "microphone distance and recording device rather than the word itself")}
    for f, (title, meaning) in desc.items():
        s = summary[f]
        if len(s) >= 2:
            hi, lo = max(s, key=s.get), min(s, key=s.get)
            spread = abs(s[hi] - s[lo]) / (abs(s[lo]) + 1e-9) * 100
            notes.append(f"**{title}:** highest for '{hi}' ({s[hi]:.2f}), lowest for '{lo}' ({s[lo]:.2f}). "
                         + (f"A real difference ({spread:.0f}%), consistent with {meaning}."
                            if spread > 15 else "Classes look alike here, so this feature alone cannot identify the "
                                                "word; the model must learn the time-frequency pattern."))
    result = {"classes": classes, "counts": counts, "speakers": speakers, "features": feats,
              "summary": summary, "mean_spectrograms": mean_specs, "interpretations": notes}
    _insight_cache.update(key=key, value=result)
    return result
