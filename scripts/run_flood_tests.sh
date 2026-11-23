#!/usr/bin/env bash
# Run the same Locust flood against 1, 2 and 3 API containers and save CSV results.
# Usage: bash scripts/run_flood_tests.sh [users] [spawn_rate] [duration]
set -e
USERS=${1:-100}; RATE=${2:-10}; TIME=${3:-60s}
for N in 1 2 3; do
  echo "=== $N container(s) ==="
  docker compose up -d --scale api=$N
  docker compose restart lb          # nginx re-resolves all API containers
  sleep 15
  locust -f locustfile.py --headless -u "$USERS" -r "$RATE" -t "$TIME" \
         --host http://localhost:8000 --csv "results/containers_$N" --only-summary
done
python scripts/summarize_locust.py
