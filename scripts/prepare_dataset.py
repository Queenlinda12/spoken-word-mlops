"""Build data/train + data/test from raw_recordings/<word>/<speaker>__<id>.wav and print a summary.
Usage: python scripts/prepare_dataset.py raw_recordings"""
import json
import sys

sys.path.append(".")
from src import preprocessing as pp  # noqa: E402

raw = sys.argv[1] if len(sys.argv) > 1 else "raw_recordings"
print(json.dumps(pp.split_raw_dataset(raw), indent=2))
