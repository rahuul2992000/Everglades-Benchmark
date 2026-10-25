"""Ask the running services to compute every model and horizon, and wait.

    python scripts/warm_cache.py            # all 12 models x 4 horizons
    python scripts/warm_cache.py --family chronos
Finished configurations return immediately, so this is safe to rerun.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from everglades_core import config  # noqa: E402

URLS = {"nf": os.environ.get("NF_SERVICE_URL", "http://localhost:8001"),
        "chronos": os.environ.get("CHRONOS_SERVICE_URL", "http://localhost:8002")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["nf", "chronos"])
    args = ap.parse_args()
    jobs = []
    for m, spec in config.MODELS.items():
        if args.family and spec.family != args.family:
            continue
        for h in config.HORIZONS:
            r = requests.post(URLS[spec.family] + "/runs", json={"model": m, "horizon": h}, timeout=15)
            r.raise_for_status()
            jobs.append((spec.family, m, h, r.json()["run_id"]))
    print(f"Submitted {len(jobs)} configurations")
    left = set(jobs)
    while left:
        for j in sorted(left):
            s = requests.get(f"{URLS[j[0]]}/runs/{j[3]}", timeout=15).json()
            if s["state"] in ("succeeded", "failed"):
                left.discard(j)
                msg = f"{s.get('runtime_seconds', '')} s" if s["state"] == "succeeded" else s.get("error")
                print(f"  {j[1]:<14} h{j[2]:<3} {s['state']}  {msg}")
        if left:
            time.sleep(10)
    print("Done")


if __name__ == "__main__":
    main()
