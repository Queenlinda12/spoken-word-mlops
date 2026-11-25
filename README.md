# Spoken-Word Recognition: an end-to-end MLOps pipeline

Voice-driven tools only work if they understand what people say. This project recognises short spoken words
(*yes, no, up, down, left, right, stop, go*) so that a voice-driven app (hands-free control, accessibility tools,
learning apps) can respond. It covers the full ML cycle: data acquisition → preprocessing → model creation →
testing → retraining (with trigger) → API → UI → Docker deployment → load testing. It is a proof of concept: the same
pipeline can be retrained on another language (for example Kinyarwanda) by replacing the recordings.

| | |
|---|---|
| 🎥 **Video demo (YouTube)** | `PASTE YOUR YOUTUBE LINK HERE` (camera on: prediction, upload, retrain, Locust) |
| 🌐 **Run the UI (Docker)** | `docker compose up --build -d`, then open http://localhost:8501 |
| 🔌 **API + Swagger docs** | http://localhost:8000/docs (after starting with Docker) |
| 💻 **GitHub repo** | https://github.com/Queenlinda12/spoken-word-mlops |

> The project is fully **Dockerized** (API, nginx load balancer and UI), so it runs the same way on any computer with Docker.
> See "Run with Docker" below. The first prediction after start-up is slow (about 40 s) because the model is loading; later ones are fast.

## Project description

* **Data type:** sound (.wav). **Dataset:** [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209) (public, CC BY 4.0):
  8 words, 40 speakers for the main dataset plus 6 different speakers kept for the retraining demo.
* **Words (classes):** `yes, no, up, down, left, right, stop, go`. The classes are the folder names in `raw_recordings/`.
* **Preprocessing:** mono 16 kHz → trim silence → centre in 2 s → 128×128 log-mel spectrogram → scaled to [-1, 1].
* **Models compared:** vanilla CNN (baseline), MobileNetV2 transfer learning, and a regularised CNN. The best model by
  **validation** F1 is saved; the test set is only used to report. In my run the regularised CNN won, so the saved file
  `models/spoken_words_mobilenetv2.h5` (name kept for compatibility) holds the regularised CNN.
* **Optimisation:** dropout, L2, batch norm, Adam, early stopping, ReduceLROnPlateau, class weights, audio augmentation
  (time shift, gain, speed, pitch, noise) and SpecAugment; transfer learning with two-stage fine-tuning for MobileNetV2.
* **Honest evaluation:** whole speakers are held out, so the test set (and the validation set) contain **speakers never seen
  in training**.
* **Metrics:** accuracy, loss, precision, recall, F1 (weighted + macro), ROC-AUC, confusion matrix.
* **Retraining:** uploaded or recorded clips are validated (real audio, not silent), saved to `data/train/<word>/` and logged
  in a SQLite database (`data/app.db`). Retraining **loads the saved model as the pre-trained model**, fine-tunes it
  (learning rate 1e-5) on the larger dataset and saves a new version (the previous one is kept as `*_prev.h5`).
  Triggers: the **Retrain button**, or **automatically** when `RETRAIN_THRESHOLD` (default 30) new clips are pending.

## Results (model version 2, held-out test speakers)

| Accuracy | Loss | Precision | Recall | F1 (weighted) |
|---|---|---|---|---|
| 0.780 | 0.737 | 0.808 | 0.780 | 0.783 |

Full tables (validation vs test, per-class report, confusion matrix, retraining before/after) are in the notebook.

## Repository layout

```
spoken-word-mlops/
├── README.md
├── notebook/kinyarwanda_voice_mlops.ipynb   # preprocessing, training, evaluation, prediction, retrain demo
├── src/
│   ├── preprocessing.py   # acquisition, audio preprocessing + augmentation, uploads DB, data insights
│   ├── model.py           # build, train, evaluate, RETRAIN, background jobs
│   └── prediction.py      # single-clip prediction (hot-reloads new model versions)
├── api/main.py            # FastAPI: /health /status /predict /upload /train /retrain /insights /runs
├── ui/app.py              # Streamlit UI (uptime, predict, visualisations, record/upload, train/retrain)
├── locustfile.py          # flood test
├── scripts/               # prepare_speech_commands.py, check_recordings.py, prepare_dataset.py, convert_to_wav.sh,
│                          # run_flood_tests.sh, summarize_locust.py
├── Dockerfile, ui/Dockerfile, docker-compose.yml, nginx.conf
├── data/train/ data/test/ # <word>/<speaker>__<id>.wav
└── models/                # spoken_words_mobilenetv2.h5, class_names.json, metrics.json
```

## Step 0: get the data (no recording needed)

```bash
python scripts/prepare_speech_commands.py      # downloads Speech Commands (~2.3 GB), builds raw_recordings/ and new_recordings/
python scripts/check_recordings.py raw_recordings
```

* `raw_recordings/<word>/<speaker>__<n>.wav`: main dataset (40 speakers, up to 3 clips per speaker per word).
* `new_recordings/<word>/<speaker>__<n>.wav`: 6 different speakers, used for the retraining demo and the bulk-upload demo.
* Change the words with `WORDS="yes,no,stop" python scripts/prepare_speech_commands.py`.
* To use your own recordings, put them in the same layout (speaker name, double underscore, number).

## Setup

### 1. Build the dataset and train the first model

```bash
git clone https://github.com/Queenlinda12/spoken-word-mlops.git && cd spoken-word-mlops
pip install -r requirements.txt
jupyter notebook notebook/kinyarwanda_voice_mlops.ipynb      # run all cells (Google Colab works too, use a GPU)
```

The notebook creates `data/train` + `data/test`, trains and compares the models, and saves the best one to `models/`.
No notebook? `python scripts/prepare_dataset.py raw_recordings && python -c "from src import model; model.train()"`
(set `MODEL_ARCH=cnn` to train the small CNN instead of MobileNetV2).

### 2. Run with Docker (API + UI + load balancer)

The `models/` and `data/train/` folders must be present (they are in this repo). Then:

```bash
docker compose up --build -d
# UI: http://localhost:8501      API docs: http://localhost:8000/docs
docker compose up -d --scale api=3 && docker compose restart lb      # run 3 API containers
docker compose down                                                    # stop everything
```

If the UI says "API is DOWN", wait about a minute for the model to load, then run `docker compose restart lb`.

### 3. Run without Docker

```bash
python -m uvicorn api.main:app --port 8000                          # terminal 1
pip install -r ui/requirements.txt
API_URL=http://localhost:8000 streamlit run ui/app.py               # terminal 2
```

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/status` | up-time, model version, test metrics, training state, pending uploads |
| POST | `/predict` | one audio clip → predicted word, confidence, probabilities |
| POST | `/upload` | bulk clips + `label` + `speaker` → saved + logged (auto-triggers retrain at threshold) |
| POST | `/train` / `/retrain` | start first training / retraining (background) |
| GET | `/insights` | class balance, 3 acoustic features, spectrograms, interpretations |
| GET | `/runs` | training history from the database |

## Flood request simulation (Locust)

```bash
bash scripts/run_flood_tests.sh 100 10 60s    # 100 users, 10 new users/s, 60 s, on 1, 2 and 3 API containers
python scripts/summarize_locust.py            # prints the markdown table below
```

Or interactively: `locust -f locustfile.py --host http://localhost:8000` and open http://localhost:8089.

**Results (100 users, 60 s, through the nginx load balancer):**

| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Req/s |
|---|---|---|---|---|---|---|---|
| 1 | 284 | 0 | 14282 | 13000 | 24000 | 25999 | 4.90 |
| 2 | 99 | 0 | 4063 | 4800 | 6700 | 7544 | 4.17 |
| 3 | 200 | 0 | 4435 | 4400 | 8200 | 8930 | 5.79 |

(All rows are the aggregated totals of `POST /predict`, `GET /status` and `GET /health`. Raw CSVs are in `results/`.)

![Locust charts](results/locust_screenshot.png.png)

*Locust web UI charts from a separate interactive run (3 API containers, 100 users, 60 s). The small first bump is a 10-user trial
run; the second is the 100-user run. Its absolute numbers differ a little from the table above, which comes from the scripted run.*

**Observation:** There were no failed requests in any run. Going from 1 to 2 or 3 API containers cut the average latency from about
14.3 s to about 4.1 to 4.4 s and the p95 from 24 s to 6.7 to 8.2 s, because the load balancer spreads the predictions across
containers instead of queueing them in one. Throughput (requests/s) rose only slightly, because the load generator, nginx and all
containers share one laptop's CPU, so the gain is limited by that machine; on separate cloud servers the extra containers would
add more throughput.

## Demo checklist (video, camera on)

1. Dashboard: up-time, metrics, confusion matrix. 2. **Predict**: upload or record a word.
3. Visualisations page and interpretation. 4. Upload clips from `new_recordings/` → pending count goes up.
5. Press **Retrain** → model version increases (2 → 3). 6. Locust flood and the results table.

## Limitations

Small vocabulary, limited number of voices and microphones, mostly quiet-room recordings. Next steps: more speakers
(children and adults), background-noise augmentation, more words, cloud hosting behind HTTPS (browser microphone
recording needs HTTPS or localhost), and a "please repeat" prompt when confidence is low.
