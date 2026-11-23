"""Turn results/containers_N_stats.csv into a markdown table for the README."""
import csv
import glob
import re

rows = []
for path in sorted(glob.glob("results/containers_*_stats.csv")):
    n = re.search(r"containers_(\d+)_stats", path).group(1)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["Name"] == "POST /predict":
                rows.append((n, r))
print("| API containers | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Req/s |")
print("|---|---|---|---|---|---|---|---|")
for n, r in rows:
    print(f"| {n} | {r['Request Count']} | {r['Failure Count']} | {float(r['Average Response Time']):.0f} | "
          f"{r['Median Response Time']} | {r['95%']} | {float(r['Max Response Time']):.0f} | "
          f"{float(r['Requests/s']):.1f} |")
