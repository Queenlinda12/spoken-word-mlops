"""Check your recordings BEFORE training. Prints a words x speakers table and flags problems.
Usage: python scripts/check_recordings.py raw_recordings          (main dataset)
       python scripts/check_recordings.py new_recordings          (extra voices for the retraining demo)"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(sys.argv[1] if len(sys.argv) > 1 else "raw_recordings")
if not root.is_dir():
    sys.exit(f"Folder '{root}' not found. Expected {root}/<word>/<speaker>__<number>.wav")

words = sorted(d.name for d in root.iterdir() if d.is_dir())
table = defaultdict(lambda: defaultdict(int))
problems = []
for w in words:
    for f in sorted(p for p in (root / w).iterdir() if p.is_file()):
        if f.suffix.lower() != ".wav":
            problems.append(f"{f}: not a .wav file (convert with scripts/convert_to_wav.sh)"); continue
        if "__" not in f.stem:
            problems.append(f"{f.name}: name must be <speaker>__<number>.wav (double underscore)"); continue
        spk = f.stem.split("__")[0]
        table[w][spk] += 1
        try:
            y, sr = sf.read(f, dtype="float32")
        except Exception as e:
            problems.append(f"{f}: unreadable ({e})"); continue
        if y.ndim > 1: y = y.mean(axis=1)
        dur, peak = len(y) / sr, float(np.abs(y).max()) if len(y) else 0.0
        if dur < 0.3: problems.append(f"{f}: too short ({dur:.2f}s)")
        if dur > 3.5: problems.append(f"{f}: long ({dur:.1f}s) - one word per clip, trim it")
        if peak < 0.02: problems.append(f"{f}: almost silent (peak {peak:.3f})")
        if peak >= 0.999: problems.append(f"{f}: clipped / too loud")

speakers = sorted({s for w in table for s in table[w]})
print(f"\nWords ({len(words)}): {', '.join(words)}")
print(f"Speakers ({len(speakers)}): {', '.join(speakers)}\n")
print("clips per word x speaker")
print("word".ljust(12) + "".join(s[:7].ljust(8) for s in speakers) + "total")
for w in words:
    print(w.ljust(12) + "".join(str(table[w].get(s, 0)).ljust(8) for s in speakers) + str(sum(table[w].values())))
total = sum(sum(v.values()) for v in table.values())
print(f"\nTotal clips: {total}")
if root.name != "new_recordings":
    if len(speakers) < 8: problems.append(f"only {len(speakers)} speakers: aim for 10-12 (whole speakers are held out for testing)")
    if any(sum(table[w].values()) < 40 for w in words): problems.append("some word has < 40 clips")
miss = [(w, s) for w in words for s in speakers if table[w].get(s, 0) == 0]
if miss: print(f"\nnote: {len(miss)} speaker/word pairs have no clips (fine for a public dataset)")
print("\nPROBLEMS:" if problems else "\nNo problems found.")
for p in problems[:40]: print(" -", p)
if len(problems) > 40: print(f" ... and {len(problems) - 40} more")
