"""FastAPI app shared by the NeuralForecast and Chronos services.

One background thread runs jobs one at a time, so a service never puts two
trainings on the GPU at once. Status lives on disk, so a restart loses
nothing that finished; runs interrupted by a restart are marked failed.
"""
from __future__ import annotations

import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd

from . import config
from .jobs import execute
from .store import RunStore, run_id


class RunRequest(BaseModel):
    model: str
    horizon: int
    force: bool = False  # recompute even if a finished run exists (admin use)


class ForecastRequest(BaseModel):
    model: str
    horizon: int


def check_forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast must cover exactly the h days after the last observed day."""
    need = ["unique_id", "ds", "yhat"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast is missing columns {missing}")
    df = df[need].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    last = pd.Timestamp(config.TEST_END)
    expected = list(pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D"))
    for uid, g in df.groupby("unique_id"):
        if sorted(g["ds"]) != expected:
            raise ValueError(f"Forecast for {uid} covers {g['ds'].min().date()} to "
                             f"{g['ds'].max().date()}, expected the {horizon} days after {last.date()}")
    if df["unique_id"].nunique() != config.N_SERIES:
        raise ValueError(f"Forecast has {df['unique_id'].nunique()} series, "
                         f"expected {config.N_SERIES}")
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def make_app(runner, store: RunStore | None = None) -> FastAPI:
    store = store or RunStore()
    models = config.models_for(runner.family)
    jobs: queue.Queue = queue.Queue()
    pending: list[str] = []
    lock = threading.Lock()
    forecast_lock = threading.Lock()
    needs_training = getattr(runner, "needs_training", True)

    for s in store.all_statuses():
        if s.get("state") in ("queued", "running") and s.get("model") in models:
            store.set_status(s["run_id"], state="failed", stage="Failed",
                             error="The service restarted before this run finished.")

    def worker():
        while True:
            rid, model, horizon = jobs.get()
            try:
                execute(store, runner, model, horizon)
            finally:
                with lock:
                    if rid in pending:
                        pending.remove(rid)
                jobs.task_done()

    threading.Thread(target=worker, daemon=True, name="run-worker").start()
    app = FastAPI(title=f"Everglades benchmark - {runner.family} service")

    def _with_position(s: dict) -> dict:
        with lock:
            if s.get("state") == "queued" and s["run_id"] in pending:
                s = {**s, "queue_position": pending.index(s["run_id"])}
        return s

    def _get(rid: str) -> dict:
        s = store.status(rid)
        if s is None:
            raise HTTPException(404, f"No run {rid}")
        return s

    @app.get("/health")
    def health():
        with lock:
            depth = len(pending)
        return {"family": runner.family, "models": models,
                "config_version": config.CONFIG_VERSION,
                "queue_depth": depth, "environment": runner.environment()}

    @app.post("/runs")
    def submit(req: RunRequest):
        if req.model not in models:
            raise HTTPException(400, f"{req.model} is not served here. "
                                     f"Available: {', '.join(models)}")
        if req.horizon not in config.HORIZONS:
            raise HTTPException(400, f"Horizon must be one of {config.HORIZONS}")
        rid = run_id(req.model, req.horizon)
        cur = store.status(rid)
        if cur and not req.force:
            if cur["state"] in ("queued", "running", "succeeded"):
                return _with_position(cur)
        s = store.set_status(rid, model=req.model, horizon=req.horizon,
                             state="queued", stage="Waiting to start",
                             progress=None, error=None)
        with lock:
            pending.append(rid)
        jobs.put((rid, req.model, req.horizon))
        return _with_position(s)

    @app.get("/runs/{rid}")
    def status(rid: str):
        return _with_position(_get(rid))

    def _done(rid: str):
        s = _get(rid)
        if s["state"] != "succeeded":
            raise HTTPException(409, f"Run {rid} is {s['state']}")

    @app.get("/runs/{rid}/metrics")
    def metrics(rid: str):
        _done(rid)
        return {"metrics": store.metrics(rid), "meta": store.meta(rid)}

    @app.get("/runs/{rid}/predictions")
    def predictions(rid: str, all_series: bool = False):
        _done(rid)
        df = store.predictions(rid)
        if not all_series:
            df = df[df["unique_id"].isin(config.STATIONS)]
        df = df.assign(ds=df["ds"].dt.strftime("%Y-%m-%d"))
        return df.to_dict("records")

    @app.post("/forecast")
    def forecast(req: ForecastRequest):
        if req.model not in models:
            raise HTTPException(400, f"{req.model} is not served here.")
        if req.horizon not in config.HORIZONS:
            raise HTTPException(400, f"Horizon must be one of {config.HORIZONS}")
        rid = run_id(req.model, req.horizon)
        s = store.status(rid)
        training_seconds = None
        artifacts = store.artifacts_dir(rid)
        if needs_training:
            if not s or s.get("state") != "succeeded" or not artifacts.exists():
                raise HTTPException(409, {
                    "code": "not_trained", "run_id": rid,
                    "message": f"{req.model} hasn't been trained for a "
                               f"{req.horizon}-day horizon yet."})
            training_seconds = store.meta(rid).get("timings", {}).get("training_seconds")
        with forecast_lock:
            fc, timings = runner.forecast(req.model, req.horizon, artifacts)
        fc = check_forecast(fc, req.horizon)
        fc = fc.assign(ds=fc["ds"].dt.strftime("%Y-%m-%d"))
        return {"model": req.model, "horizon": req.horizon,
                "origin": config.TEST_END, "run_id": rid,
                "rows": fc.to_dict("records"),
                "timings": {**timings, "training_seconds": training_seconds,
                            "needs_training": needs_training},
                "environment": runner.environment()}

    @app.get("/runs/{rid}/predictions.csv")
    def predictions_csv(rid: str):
        _done(rid)
        s = store.status(rid)
        return FileResponse(store.dir(rid) / "predictions.csv", media_type="text/csv",
                            filename=f"{s['model']}_h{s['horizon']}_predictions.csv")

    return app
