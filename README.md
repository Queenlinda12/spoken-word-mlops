# Spoken-Word Recognition: an end-to-end MLOps pipeline

Voice-driven tools only work if they understand what people say. This project recognises short spoken words
(*yes, no, up, down, left, right, stop, go*) so that a voice-driven app (hands-free control, accessibility tools,
learning apps) can respond. It covers the full ML cycle: data acquisition → preprocessing → model creation →
testing → retraining (with trigger) → API → UI → cloud deployment → load testing. It is a proof of concept: the same
pipeline can be retrained on another language (for example Kinyarwanda) by replacing the recordings.

| | |
|---|---|
| 🎥 **Video demo (YouTube)** | `PASTE YOUR YOUTUBE LINK HERE` (camera on: prediction, upload, retrain, Locust) |
| 🌐 **Live UI** | `http://YOUR_SERVER_IP:8501` |
| 🔌 **Live API + Swagger docs** | `http://YOUR_SERVER_IP:8000/docs` |
| 💻 **GitHub repo** | `https://github.com/Queenlinda12/spoken-word-mlops` |

## Project description

* **Data type:** sound (.wav). **Dataset:** [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209) (public, CC BY 4.0):
  8 words, 40 speakers for the main dataset plus 6 different speakers kept for the retraining demo.
* **Words (classes):** `yes, no, up, down, left, right, stop, go`. The classes are just the folder names in `raw_recordings/`.
* **Preprocessing:** mono 16 kHz → trim silence → centre in 2 s → 128×128 log-mel spectrogram → loudness-normalised to [-1, 1].
* **Models compared:** vanilla CNN (baseline) vs MobileNetV2 transfer learning vs a regularised CNN. The best model by
  **validation** F1 is saved; the test set is only used to report.
* **Optimisation:** transfer learning, two-stage fine-tuning, dropout, L2, Adam, early stopping, ReduceLROnPlateau, class
  weights, audio augmentation (time shift, gain, speed, pitch, noise) and SpecAugment.
* **Honest evaluation:** the test set contains **speakers never seen in training** (whole speakers are held out), and the
  validation split is also by speaker.
* **Metrics:** accuracy, loss, precision, recall, F1 (weighted + macro), ROC-AUC, confusion matrix.
* **Retraining:** uploaded or recorded clips are validated (real audio, not silent), saved to `data/train/<word>/` and logged in
  a SQLite database (`data/app.db`). Retraining **loads the saved model as the pre-trained model**, fine-tunes it on the larger
  dataset and saves a new version (the previous one is kept as `*_prev.h5`). Triggers: the **Retrain button** or
  **automatically** when `RETRAIN_THRESHOLD` (default 30) new clips are pending.

## Repository layout

```
spoken-word-mlops/
├── README.md
├── notebook/kinyarwanda_voice_mlops.ipynb   # preprocessing, training, evaluation, prediction, retrain demo
├── src/
│   ├── preprocessing.py   # acquisition, audio preprocessing + augmentation, uploads DB, data insights
│   ├── model.py           # build, train, evaluate, RETRAIN, background jobs
│   └── prediction.py      # single-clip prediction (hot-reloads new model versions)
├── api/main.py            # FastAPI: /predict /upload /train /retrain /status /insights /runs
├── ui/app.py              # Streamlit UI (uptime, predict, visualisations, record/upload, train/retrain)
├── locustfile.py          # flood test
├── scripts/               # prepare_speech_commands.py, check_recordings.py, prepare_dataset.py, convert_to_wav.sh, run_flood_tests.sh, summarize_locust.py
├── Dockerfile, ui/Dockerfile, docker-compose.yml, nginx.conf
├── data/train/ data/test/ # <word>/<speaker>__<id>.wav
└── models/                # spoken_words_mobilenetv2.h5, class_names.json, metrics.json
```

## Step 0: get the data (no recording needed)

```bash
python scripts/prepare_speech_commands.py      # downloads Speech Commands (~2.3 GB), builds raw_recordings/ and new_recordings/
python scripts/check_recordings.py raw_recordings
```

* `raw_recordings/<word>/<speaker>__<n>.wav`: main dataset (40 speakers, ≤3 clips per speaker per word).
* `new_recordings/<word>/<speaker>__<n>.wav`: 6 different speakers, used for the retraining demo and the bulk-upload demo.
* Change the words with `WORDS="yes,no,stop" python scripts/prepare_speech_commands.py`.
* To use your own recordings instead, put them in the same folder layout (speaker name, double underscore, number).

## Setup

### 1. Build the dataset and train the first model

```bash
git clone https://github.com/Queenlinda12/spoken-word-mlops.git && cd spoken-word-mlops
pip install -r requirements.txt
jupyter notebook notebook/kinyarwanda_voice_mlops.ipynb      # run all cells (Google Colab works too)
```

The notebook creates `data/train` + `data/test`, trains and compares the models, and saves the best one to `models/`.
No notebook? `python scripts/prepare_dataset.py raw_recordings && python -c "from src import model; model.train()"`
(set `MODEL_ARCH=cnn` to train the small CNN instead of MobileNetV2).

### 2. Run locally with Docker (API + UI + load balancer)

```bash
docker compose up --build -d
# UI: http://localhost:8501      API docs: http://localhost:8000/docs
docker compose up -d --scale api=3 && docker compose restart lb      # run 3 API containers
```

### 3. Run locally without Docker

```bash
uvicorn api.main:app --port 8000                                    # terminal 1
pip install -r ui/requirements.txt
API_URL=http://localhost:8000 streamlit run ui/app.py               # terminal 2
```

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | up-time, model version, test metrics, training state, pending uploads |
| POST | `/predict` | one audio clip → predicted word, confidence, probabilities |
| POST | `/upload` | bulk clips + `label` + `speaker` → saved + logged (auto-triggers retrain at threshold) |
| POST | `/train` / `/retrain` | start first training / retraining (background) |
| GET | `/insights` | class balance, 3 acoustic features, spectrograms, interpretations |
| GET | `/runs` | training history from the database |

## Cloud deployment

Deployed with Docker Compose on a cloud VM (AWS EC2 / GCP Compute Engine / Azure / DigitalOcean; 2+ vCPU, 4+ GB RAM, Ubuntu).

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
git clone https://github.com/Queenlinda12/spoken-word-mlops.git && cd spoken-word-mlops
# copy your trained models/ and data/ folders to the VM (scp, git-lfs or a bucket), then:
sudo docker compose up --build -d
```

Open ports **8501** (UI) and **8000** (API) in the cloud firewall. Put the public URLs at the top of this README.
Model evaluation in production: the **Dashboard** page shows live test metrics, the confusion matrix, up-time and training
history from the production server. Note: browser microphone recording needs **HTTPS** (or localhost); if your server is plain
HTTP, use the file-upload tabs in the demo, or put the UI behind a free HTTPS proxy (for example Caddy or a Cloudflare tunnel).

## Flood request simulation (Locust)

```bash
bash scripts/run_flood_tests.sh 100 10 60s    # 100 users, 10 new users/s, 60 s, on 1, 2 and 3 API containers
python scripts/summarize_locust.py            # prints the markdown table below
```

Or interactively: `locust -f locustfile.py --host http://YOUR_SERVER_IP:8000` and open http://localhost:8089.

**Results (paste the table printed by `summarize_locust.py` from YOUR run here):**

| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |

Add a screenshot of the Locust charts and 2 to 3 sentences on how latency and throughput changed with more containers.

## Demo checklist (video, camera on)

1. Dashboard: up-time, metrics, confusion matrix. 2. **Predict**: say a word into the mic (or upload one the model gets right).
3. Visualisations page and your interpretation. 4. Upload many clips (or record a few) → pending count goes up.
5. Press **Retrain** → model version increases. 6. Locust flood and the results table.

## Limitations

Small vocabulary, limited number of voices and microphones, quiet-room recordings. Next steps: more speakers (children and
adults), background-noise augmentation, more words, and a "please repeat" prompt when confidence is low.
