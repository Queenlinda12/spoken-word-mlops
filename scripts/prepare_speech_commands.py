"""Build raw_recordings/ and new_recordings/ from the public Google Speech Commands dataset (no recording needed).

    python scripts/prepare_speech_commands.py            # downloads ~2.3 GB, keeps only the chosen words (run on Colab)
    python scripts/prepare_speech_commands.py path/to/speech_commands_v0.02.tar.gz   # use a tar.gz you already have

Output (speaker = the dataset's anonymised speaker id, so the pipeline can hold out whole speakers):
    raw_recordings/<word>/<speaker>__<n>.wav   main dataset (train / validation / test)
    new_recordings/<word>/<speaker>__<n>.wav   DIFFERENT speakers, used for the retraining demo
Dataset: Warden (2018), "Speech Commands", CC BY 4.0, https://arxiv.org/abs/1804.03209
"""
import os
import random
import re
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
WORDS = os.getenv("WORDS", "yes,no,up,down,left,right,stop,go").split(",")
N_MAIN = int(os.getenv("N_MAIN", 40))        # speakers in the main dataset
N_NEW = int(os.getenv("N_NEW", 6))           # extra speakers for the retraining demo
MAX_PER = int(os.getenv("MAX_PER", 3))       # clips kept per speaker per word
SEED = 42
PAT = re.compile(r"(?:^|/)(%s)/([0-9a-f]{8})_nohash_(\d+)\.wav$" % "|".join(map(re.escape, WORDS)))

src = sys.argv[1] if len(sys.argv) > 1 else "speech_commands_v0.02.tar.gz"
tmp = Path("speech_commands_tmp")
if not Path(src).exists():
    print("downloading", URL, "(about 2.3 GB, a few minutes)...")
    urllib.request.urlretrieve(URL, src)

print("extracting only:", WORDS)
shutil.rmtree(tmp, ignore_errors=True)
by_speaker = defaultdict(lambda: defaultdict(list))        # speaker -> word -> [(n, path)]
with tarfile.open(src, "r|gz") as tar:
    for m in tar:
        hit = PAT.search(m.name) if m.isfile() else None
        if not hit:
            continue
        word, spk, n = hit.group(1), hit.group(2), int(hit.group(3))
        out = tmp / word / f"{spk}_{n}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(tar.extractfile(m).read())
        by_speaker[spk][word].append((n, out))

rng = random.Random(SEED)
speakers = sorted(by_speaker)
rng.shuffle(speakers)
speakers.sort(key=lambda s: -len(by_speaker[s]))           # speakers who said the most of our words first
need = N_MAIN + N_NEW
if len(speakers) < need:
    sys.exit(f"only {len(speakers)} speakers found, need {need}")
chosen = {"raw_recordings": speakers[:N_MAIN], "new_recordings": speakers[N_MAIN:need]}

for folder, spks in chosen.items():
    shutil.rmtree(folder, ignore_errors=True)
    total = 0
    for spk in spks:
        for word, items in by_speaker[spk].items():
            for i, (_, p) in enumerate(sorted(items)[:MAX_PER], 1):
                dest = Path(folder) / word / f"{spk}__{i}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                total += 1
    print(f"{folder}: {len(spks)} speakers, {total} clips")
shutil.rmtree(tmp, ignore_errors=True)
print("next: python scripts/check_recordings.py raw_recordings")
