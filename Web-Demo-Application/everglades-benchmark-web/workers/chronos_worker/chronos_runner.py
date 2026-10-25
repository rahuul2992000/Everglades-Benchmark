"""Chronos-Bolt runner, following
FOllow-for-chronos-UPDATED-OldDATAEverglades-Chronos-7-days copy 2.ipynb.

Evaluation:
    history_window_size = 100 + horizon (107, 114, 121, 128), prediction_length = horizon
    first_date = test start (2024-01-29)
    predictor = TimeSeriesPredictor(prediction_length).fit(
        last H days per series up to first_date, presets="bolt_small")
    for current_date in test period:
        forecast = predictor.predict(last H days per series up to current_date)
        take forecast['mean'].groupby(level='item_id').last(), ds = current_date
    merge predictions with Y_df on (unique_id, ds), inner

Forecast (Forecast page): the same predictor setup, one predict call from the
last H days of data, returning every day of the forecast (not just the last).
"""
from __future__ import annotations

import platform
import tempfile
import time

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from everglades_core import config, data
from everglades_core.timing import timed_forecast


def _ts(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(df, id_column="unique_id",
                                               timestamp_column="ds")


def _full_path(forecast) -> pd.DataFrame:
    fc = forecast["mean"].reset_index()
    return fc.rename(columns={"item_id": "unique_id", "timestamp": "ds",
                              "mean": "yhat"})[["unique_id", "ds", "yhat"]]


class ChronosRunner:
    family = "chronos"
    needs_training = False

    def __init__(self):
        self._predictors: dict[int, TimeSeriesPredictor] = {}

    def run(self, model: str, horizon: int, progress, ctx):
        progress("Loading data", None)
        long_df = data.load_long()
        s = data.split(long_df)
        Y_df = long_df.rename(columns={"y": "target"})
        H = config.chronos_history(horizon)
        first_date = s.test_start

        progress("Loading Chronos-Bolt", None)
        initial = Y_df[Y_df["ds"] <= first_date].groupby("unique_id").tail(H)
        with tempfile.TemporaryDirectory(prefix="chronos-") as workdir:
            t0 = time.perf_counter()
            predictor = TimeSeriesPredictor(
                prediction_length=horizon, path=f"{workdir}/predictor", verbosity=0,
            ).fit(_ts(initial), presets=config.CHRONOS_PRESET)
            ctx.timings["setup_seconds"] = round(time.perf_counter() - t0, 2)
            ctx.timings["training_seconds"] = None  # zero-shot: nothing is trained

            dates = pd.date_range(first_date, s.test_end, freq="D")
            predictions_list = []
            t0 = time.perf_counter()
            for i, current_date in enumerate(dates):
                train_data = Y_df[Y_df["ds"] <= current_date]
                train_data = train_data.groupby("unique_id").tail(H)
                forecast = predictor.predict(_ts(train_data))
                last = forecast["mean"].groupby(level="item_id").last().reset_index()
                last["ds"] = current_date
                last = last.rename(columns={"item_id": "unique_id", 0: "forecast_last_day"})
                predictions_list.append(last)
                progress("Forecasting across the test period", (i + 1) / len(dates))
            ctx.timings["evaluation_loop_seconds"] = round(time.perf_counter() - t0, 2)

            progress("Timing a single forecast", None)
            latest = Y_df.groupby("unique_id").tail(H)
            _, t = timed_forecast(lambda: predictor.predict(_ts(latest)))
            ctx.timings.update(t)

        df_predictions = pd.concat(predictions_list, ignore_index=True)
        df_predictions = df_predictions.rename(columns={"mean": "Chronos"})
        predictions_df = df_predictions.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)
        results_df = pd.merge(predictions_df, Y_df, on=["unique_id", "ds"], how="inner")
        return results_df.rename(columns={"target": "y", "Chronos": "yhat"})[
            ["unique_id", "ds", "y", "yhat"]]

    def forecast(self, model: str, horizon: int, artifacts_dir=None):
        Y_df = data.load_long().rename(columns={"y": "target"})
        latest = Y_df.groupby("unique_id").tail(config.chronos_history(horizon))

        cold = horizon not in self._predictors
        load_seconds = None
        if cold:
            workdir = tempfile.mkdtemp(prefix=f"chronos-h{horizon}-")
            t0 = time.perf_counter()
            self._predictors[horizon] = TimeSeriesPredictor(
                prediction_length=horizon, path=f"{workdir}/predictor", verbosity=0,
            ).fit(_ts(latest), presets=config.CHRONOS_PRESET)
            load_seconds = round(time.perf_counter() - t0, 3)
        predictor = self._predictors[horizon]
        fc, t = timed_forecast(lambda: predictor.predict(_ts(latest)))
        return _full_path(fc), {**t, "load_seconds": load_seconds, "cold_start": cold}

    def environment(self) -> dict:
        import autogluon.timeseries as agts
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        try:
            import chronos
            chronos_version = getattr(chronos, "__version__", "installed")
        except ImportError:
            chronos_version = None
        return {"autogluon.timeseries": agts.__version__,
                "chronos-forecasting": chronos_version,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": gpu, "host": platform.node()}
