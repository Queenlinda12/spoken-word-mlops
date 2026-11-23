#!/usr/bin/env bash
# Convert phone recordings (m4a, mp3, aac, 3gp, opus...) to 16 kHz mono wav. Needs ffmpeg.
# Usage: bash scripts/convert_to_wav.sh input_folder output_folder
set -e
IN=${1:?input folder}; OUT=${2:?output folder}
mkdir -p "$OUT"
for f in "$IN"/*; do
  [ -f "$f" ] || continue
  ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 "$OUT/$(basename "${f%.*}").wav"
done
echo "done -> $OUT"
