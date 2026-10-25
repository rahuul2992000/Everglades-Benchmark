"""Run one (model, horizon) job end to end: predict, validate, score, store.

A runner is any object with:
    family: str
    run(model, horizon, progress, ctx) -> DataFrame with unique_id, ds, y, yhat
        progress(stage: str, fraction: float | None)
        ctx.artifacts_dir: where to save a trained model (for forecasting)
        ctx.timings: dict the runner fills (training_seconds, forecast_seconds, ...)
    forecast(model, horizon, artifacts_dir) -> (DataFrame unique_id, ds, yhat, timings)
    environment() -> dict   (library versions, device)
Both model services and scripts/run_local.py call execute(), so a run is
computed the same way no matter how it was started.
"""
from __future__ import annotations

import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .data import fingerprint
from .scoring import score_predictions
from .store import RunStore, run_id

REQUIRED = ["unique_id", "ds", "y", "yhat"]


@dataclass
class RunContext:
    artifacts_dir: Path
    timings: dict = field(default_factory=dict)


def validate(pred: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in pred.columns]
    if missing:
        raise ValueError(f"Runner output is missing columns {missing}")
    pred = pred[REQUIRED].copy()
    pred["ds"] = pd.to_datetime(pred["ds"])
    n_dates = pred.groupby("unique_id")["ds"].nunique()
    bad = n_dates[n_dates != config.TEST_SIZE]
    if len(bad):
        raise ValueError(f"Expected {config.TEST_SIZE} dates per series, got "
                         f"{bad.to_dict()}")
    start, end = pred["ds"].min().date(), pred["ds"].max().date()
    if (str(start), str(end)) != (config.TEST_START, config.TEST_END):
        raise ValueError(f"Predictions cover {start} to {end}, expected "
                         f"{config.TEST_START} to {config.TEST_END}")
    return pred.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def execute(store: RunStore, runner, model: str, horizon: int) -> dict:
    rid = run_id(model, horizon)
    started = time.time()
    store.set_status(rid, model=model, horizon=horizon, state="running",
                     stage="Starting", progress=None, started_at=started)

    def progress(stage: str, frac: float | None = None):
        store.set_status(rid, stage=stage, progress=frac)

    ctx = RunContext(artifacts_dir=store.dir(rid) / "model")
    try:
        pred = validate(runner.run(model, horizon, progress, ctx))
        progress("Scoring", None)
        metrics = score_predictions(pred)
        finished = time.time()
        meta = {
            "run_id": rid, "model": model, "horizon": horizon,
            "config_version": config.CONFIG_VERSION,
            "model_config": config.MODELS[model].describe(),
            "data_file": config.DATA_PATH.name, "data_sha256": fingerprint(),
            "test_period": [config.TEST_START, config.TEST_END],
            "started_at": started, "finished_at": finished,
            "runtime_seconds": round(finished - started, 1),
            "timings": ctx.timings,
            "python": platform.python_version(),
            "environment": runner.environment(),
        }
        store.save_result(rid, pred, metrics, meta)
        return store.set_status(rid, state="succeeded", stage="Done", progress=1.0,
                                finished_at=finished,
                                runtime_seconds=meta["runtime_seconds"])
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        store.write_error(rid, traceback.format_exc())
        return store.set_status(rid, state="failed", stage="Failed",
                                error=f"{type(exc).__name__}: {exc}",
                                finished_at=time.time())
