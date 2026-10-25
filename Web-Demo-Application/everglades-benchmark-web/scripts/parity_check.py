"""Compare runs in the store with the published prediction files.

Reference files are found with the same rules as NEWgof_all_models:
  <paper1-root>/Everglades-Models-Results-{h}days/<Model>/...results*.csv
      (the -Multivariate folder is preferred where it exists)
  <informer-dir>/Informer-Results-{h}days.csv
  <chronos-dir>/Chronos_predictions_{h}days.csv   (defaults to
      <paper1-root>/Everglades-Models-Results-{h}days/Chronos/)

A run passes when every metric rounds to the same 3 decimals as the
reference (2 for PBIAS). Row-level differences are reported too.

    python scripts/parity_check.py --paper1-root /aul/homes/rrang016/ALL-MODEL-RESULTS-PAPER1/Paper1-ALL-MODELS-RESULTS \\
        --informer-dir /aul/homes/rrang016/Documents
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from everglades_core import config  # noqa: E402
from everglades_core.jobs import validate  # noqa: E402
from everglades_core.scoring import METRICS, score_predictions  # noqa: E402
from everglades_core.store import RunStore, run_id  # noqa: E402

NF_FILES = {
    "NBEATS": ["NBEATS/NBEATS_results.csv"],
    "NLinear": ["NLinear/Nlinear_results.csv", "NLinear/NLinear_results.csv"],
    "DLinear": ["DLinear/Dlinear_results.csv", "DLinear/DLinear_results.csv"],
    "KAN": ["KAN/KAN_results.csv"],
    "PatchTST": ["Informer_Autoformer_PatchTST/allModels_results.csv"],
    "TimeMixer": ["TimeMixer-Multivariate/TimeMixer_results-Multivariate.csv",
                  "TimeMixer/TimeMixer_results.csv"],
    "TSMixer": ["TSMixer-Multivariate/TSMixer-results-Multivariate.csv",
                "TSMixer-Multivariate/TSMixer_results-Multivariate.csv",
                "TSMixer/TSMixer-results.csv", "TSMixer/TSMixer_results.csv"],
    "TSMixerx": ["TSMixerx-Multivariate/TSMixerx_results-Multivariate.csv",
                 "TSMixerx/TSMixerx_results.csv"],
    "iTransformer": ["iTransformer-Multivariate/iTransformer_results-Multivariate.csv",
                     "iTransformer/iTransformer_results.csv"],
    "RMoK": ["RMoK-Multivariate/RMoK_results-Multivariate.csv", "RMoK/RMoK_results.csv"],
}


def first(paths):
    return next((p for p in paths if p.exists()), None)


def reference(model, h, args):
    """-> (DataFrame with unique_id, ds, y, yhat) or (None, reason)."""
    hdir = Path(args.paper1_root) / f"Everglades-Models-Results-{h}days" if args.paper1_root else None
    if model == "Informer":
        d = Path(args.informer_dir) if args.informer_dir else None
        p = first([d / f"Informer-Results-{h}days.csv", d / f"NEWInformer-Results-{h}days.csv"]) if d else None
        cols = ("Informer", "y")
    elif model == "Chronos-Bolt":
        d = Path(args.chronos_dir) if args.chronos_dir else (hdir / "Chronos" if hdir else None)
        p = first([d / f"Chronos_predictions_{h}days.csv"]) if d else None
        cols = (f"Chronos-{h}", "target")
    else:
        p = first([hdir / r for r in NF_FILES[model]]) if hdir else None
        cols = (model, "y")
    if p is None:
        return None, "reference file not found"
    df = pd.read_csv(p)
    df = df.rename(columns={cols[0]: "yhat", cols[1]: "y"})
    missing = [c for c in ("unique_id", "y", "yhat") if c not in df.columns]
    if missing:
        return None, f"reference file lacks {missing}: {p} (columns: {list(df.columns)})"
    return df[[c for c in ("unique_id", "ds", "y", "yhat") if c in df.columns]], str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper1-root")
    ap.add_argument("--informer-dir")
    ap.add_argument("--chronos-dir")
    ap.add_argument("--store", help="default: EVERGLADES_STORE or ./store")
    ap.add_argument("--out", default="parity_report.csv")
    args = ap.parse_args()

    store = RunStore(args.store)
    rows = []
    for model in config.MODELS:
        for h in config.HORIZONS:
            rid = run_id(model, h)
            s = store.status(rid)
            rec = {"model": model, "horizon": h}
            if not s or s.get("state") != "succeeded":
                rows.append({**rec, "result": "no finished run"})
                continue
            ref, src = reference(model, h, args)
            if ref is None:
                rows.append({**rec, "result": src})
                continue
            try:
                ours = store.predictions(rid)
                has_ds = "ds" in ref.columns
                # Without dates, score the file as-is, like NEWgof_all_models does.
                ref = validate(ref) if has_ds else ref.dropna(subset=["y", "yhat"])
                ref_m, our_m = score_predictions(ref)["overall"], store.metrics(rid)["overall"]
            except Exception as exc:  # noqa: BLE001 - report and continue
                rows.append({**rec, "result": f"error: {type(exc).__name__}: {exc}",
                             "reference": src})
                continue
            if has_ds:
                j = ours.merge(ref, on=["unique_id", "ds"], suffixes=("", "_ref"))
                j = j[j["unique_id"].isin(config.STATIONS)]
                n, dmax = len(j), (j["yhat"] - j["yhat_ref"]).abs().max()
                y_same = bool((j["y"] - j["y_ref"]).abs().max() < 1e-9)
            else:
                n, dmax, y_same = None, None, None
            ok = all(round(our_m[k], 2 if k == "PBIAS" else 3) ==
                     round(ref_m[k], 2 if k == "PBIAS" else 3) for k in METRICS)
            rows.append({**rec, "result": ("match" if ok else "DIFFERENT")
                         + ("" if has_ds else " (no dates in reference)"),
                         "rows_compared": n,
                         "max_abs_diff_yhat": dmax,
                         "y_identical": y_same,
                         **{f"{k}": round(our_m[k], 4) for k in METRICS},
                         **{f"{k}_ref": round(ref_m[k], 4) for k in METRICS},
                         "reference": src})
    rep = pd.DataFrame(rows)
    rep.to_csv(args.out, index=False)
    show = [c for c in ["model", "horizon", "result", "max_abs_diff_yhat", "MAE", "MAE_ref",
                        "NSE", "NSE_ref"] if c in rep]
    print(rep[show].to_string(index=False))
    print(f"\nFull report: {args.out}")
    counts = rep["result"].value_counts().to_dict()
    print("Summary:", ", ".join(f"{v} {k}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
