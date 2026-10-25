"""Collect every finished website run into one CSV, in the same layout as the
paper's results_all_horizons.csv (per-station MAE/RMSE, then overall metrics),
plus ratings and timings.

    python scripts/export_results.py                 # writes site_results_all_horizons.csv
    python scripts/export_results.py --out my.csv

Only runs of the current configuration version are included.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from everglades_core import config  # noqa: E402
from everglades_core.store import RunStore, run_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", help="default: EVERGLADES_STORE or ./store")
    ap.add_argument("--out", default="site_results_all_horizons.csv")
    args = ap.parse_args()

    store, rows, missing = RunStore(args.store), [], []
    for model in config.MODELS:
        for h in config.HORIZONS:
            rid = run_id(model, h)
            s = store.status(rid)
            if not s or s.get("state") != "succeeded":
                missing.append(f"{model} {h}d")
                continue
            m, meta = store.metrics(rid), store.meta(rid)
            row = {"Model": model, "Lead Time": h}
            for st in m["per_station"]:
                row[f"{st['station']}_MAE"] = st["MAE"]
                row[f"{st['station']}_RMSE"] = st["RMSE"]
            o = m["overall"]
            for k in ("MAE", "RMSE", "NSE", "PBIAS", "R2"):
                row[f"Overall_{k}"] = o[k]
            for k in ("NSE", "PBIAS", "R2"):
                row[f"{k}_rating"] = o.get(f"{k}_rating")
            t = meta.get("timings", {})
            row["training_seconds"] = t.get("training_seconds")
            row["forecast_seconds"] = t.get("forecast_seconds")
            row["gpu"] = (meta.get("environment") or {}).get("gpu")
            row["config_version"] = meta.get("config_version")
            row["run_id"] = rid
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} runs to {args.out} (configuration {config.CONFIG_VERSION})")
    if missing:
        print(f"Not finished ({len(missing)}): " + ", ".join(missing))
    if len(df):
        show = df[["Model", "Lead Time", "Overall_MAE", "Overall_NSE", "forecast_seconds"]]
        print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
