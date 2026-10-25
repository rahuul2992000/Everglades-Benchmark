"""Run models directly in this environment, without the web service.

Results go to the same store the services read (EVERGLADES_STORE, default
./store), so this is also how to fill the cache before a demo.

    # in the NeuralForecast environment
    python scripts/run_local.py --family nf --model NBEATS --horizon 7
    python scripts/run_local.py --family nf --all
    # in the Chronos environment
    python scripts/run_local.py --family chronos --all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from everglades_core import config  # noqa: E402
from everglades_core.jobs import execute  # noqa: E402
from everglades_core.store import RunStore, run_id  # noqa: E402


def runner_for(family):
    if family == "nf":
        sys.path.insert(0, str(ROOT / "workers" / "nf_worker"))
        from nf_runner import NFRunner
        return NFRunner()
    sys.path.insert(0, str(ROOT / "workers" / "chronos_worker"))
    from chronos_runner import ChronosRunner
    return ChronosRunner()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["nf", "chronos"], required=True)
    ap.add_argument("--model", action="append", help="repeatable; default all in family with --all")
    ap.add_argument("--horizon", type=int, action="append", choices=config.HORIZONS)
    ap.add_argument("--all", action="store_true", help="every model and horizon in the family")
    ap.add_argument("--force", action="store_true", help="recompute finished runs")
    args = ap.parse_args()

    models = config.models_for(args.family) if args.all or not args.model else args.model
    horizons = list(config.HORIZONS) if args.all or not args.horizon else args.horizon
    bad = [m for m in models if config.MODELS.get(m, None) is None
           or config.MODELS[m].family != args.family]
    if bad:
        ap.error(f"Not in the {args.family} family: {bad}")

    store, runner = RunStore(), runner_for(args.family)
    print(f"Store: {store.root}")
    for m in models:
        for h in horizons:
            s = store.status(run_id(m, h))
            if s and s.get("state") == "succeeded" and not args.force:
                print(f"  {m:<14} h{h:<3} already done")
                continue
            print(f"  {m:<14} h{h:<3} running...", flush=True)
            s = execute(store, runner, m, h)
            if s["state"] == "succeeded":
                o = store.metrics(s["run_id"])["overall"]
                print(f"  {m:<14} h{h:<3} MAE {o['MAE']:.3f}  RMSE {o['RMSE']:.3f}  "
                      f"NSE {o['NSE']:.3f}  PBIAS {o['PBIAS']:.2f}  R2 {o['R2']:.3f}  ({s['runtime_seconds']} s)")
            else:
                print(f"  {m:<14} h{h:<3} FAILED: {s.get('error')}  "
                      f"(see {store.dir(s['run_id']) / 'error.log'})")


if __name__ == "__main__":
    main()
