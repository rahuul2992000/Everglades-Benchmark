"""NeuralForecast runner: the 11 task-specific models.

Evaluation mirrors the SOTA notebooks cell for cell:
    model = <Model>(h=horizon, input_size=100, max_steps=1000,
                    val_check_steps=100, early_stop_patience_steps=..., ...)
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df=Y_train_df, val_size=val_size)          # Y_train_df = train + val
    Y_hat_df = nf.cross_validation(df=Y_df, val_size=val_size,
                                   test_size=test_size, n_windows=None)
    Y_hat_df = Y_hat_df.reset_index()
    Y_hat_df_agg = Y_hat_df.groupby(['unique_id', 'ds']).agg(
        {'<Model>': 'mean', 'y': 'mean'}).reset_index()

The fit-then-cross_validation sequence is kept exactly as in the notebooks,
because that is how the published numbers were produced. Afterwards, as the
notebooks do, the trained model is saved (nf.save) and used for a forecast
from the latest data (nf.predict(df=Y_df)). Those steps run after the scored
predictions exist, so they cannot change them.
"""
from __future__ import annotations

import contextlib
import platform
import time

import torch
from neuralforecast import NeuralForecast
from neuralforecast import models as nf_models

from everglades_core import config, data
from everglades_core.timing import timed_forecast


@contextlib.contextmanager
def matmul_precision(value: str | None):
    previous = torch.get_float32_matmul_precision()
    if value:
        torch.set_float32_matmul_precision(value)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _as_frame(fc, model: str):
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    return fc.rename(columns={model: "yhat"})[["unique_id", "ds", "yhat"]]


class NFRunner:
    family = "nf"
    needs_training = True

    def __init__(self):
        self._loaded: dict[str, NeuralForecast] = {}

    def build(self, model: str, horizon: int):
        spec = config.MODELS[model]
        cls = getattr(nf_models, spec.name)
        return cls(**spec.nf_kwargs(horizon))

    def run(self, model: str, horizon: int, progress, ctx):
        spec = config.MODELS[model]
        progress("Loading data", None)
        Y_df = data.load_long()
        s = data.split(Y_df)

        with matmul_precision(spec.matmul_precision):
            nf = NeuralForecast(models=[self.build(model, horizon)], freq="D")
            progress("Training", None)
            t0 = time.perf_counter()
            nf.fit(df=s.Y_fit_df, val_size=s.val_size)
            ctx.timings["training_seconds"] = round(time.perf_counter() - t0, 2)

            progress("Cross-validating on the test period", None)
            t0 = time.perf_counter()
            Y_hat_df = nf.cross_validation(df=Y_df, val_size=s.val_size,
                                           test_size=s.test_size, n_windows=None)
            ctx.timings["cross_validation_seconds"] = round(time.perf_counter() - t0, 2)

            Y_hat_df = Y_hat_df.reset_index()
            agg = (Y_hat_df.groupby(["unique_id", "ds"])
                   .agg({spec.name: "mean", "y": "mean"}).reset_index())

            progress("Saving the trained model", None)
            nf.save(path=str(ctx.artifacts_dir), overwrite=True, save_dataset=False)

            progress("Timing a single forecast", None)
            _, t = timed_forecast(lambda: nf.predict(df=Y_df))
            ctx.timings.update(t)

        return agg.rename(columns={spec.name: "yhat"})[["unique_id", "ds", "y", "yhat"]]

    def forecast(self, model: str, horizon: int, artifacts_dir):
        spec = config.MODELS[model]
        key = str(artifacts_dir)
        cold = key not in self._loaded
        load_seconds = None
        if cold:
            t0 = time.perf_counter()
            self._loaded[key] = NeuralForecast.load(path=key)
            load_seconds = round(time.perf_counter() - t0, 3)
        nf = self._loaded[key]
        Y_df = data.load_long()
        with matmul_precision(spec.matmul_precision):
            fc, t = timed_forecast(lambda: nf.predict(df=Y_df))
        return _as_frame(fc, spec.name), {**t, "load_seconds": load_seconds, "cold_start": cold}

    def environment(self) -> dict:
        import neuralforecast
        import pytorch_lightning
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {"neuralforecast": neuralforecast.__version__,
                "torch": torch.__version__,
                "pytorch_lightning": pytorch_lightning.__version__,
                "cuda": torch.version.cuda, "gpu": gpu,
                "host": platform.node()}
