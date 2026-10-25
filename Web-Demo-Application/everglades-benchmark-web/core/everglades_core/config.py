"""Fixed experiment configuration for the Everglades benchmark.

Every value here reproduces the published paper setup. Changing anything in
this file changes results, so bump CONFIG_VERSION whenever you do: the run
cache is keyed on it and old results will not be reused.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_VERSION = "paper-v3"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get(
    "EVERGLADES_DATA", REPO_ROOT / "data" / "final_concatenated_data.csv"))

N_SERIES = 37
VAL_SIZE = 211
TEST_SIZE = 211
# Guards: data.split() fails loudly if the data no longer produce these dates.
TEST_START = "2024-01-29"
TEST_END = "2024-08-26"

STATIONS = ["NP205_stage", "P33_stage", "G620_water_level", "NESRS1", "NESRS2"]
STATION_LABELS = {
    "NP205_stage": "NP205",
    "P33_stage": "P33",
    "G620_water_level": "G620",
    "NESRS1": "NESRS1",
    "NESRS2": "NESRS2",
}

HORIZONS = (7, 14, 21, 28)
METRIC_SCHEME = "moriasi2015"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
NF_COMMON = {"input_size": 100, "max_steps": 1000, "val_check_steps": 100}

CHRONOS_INPUT = 100  # history window = 100 + horizon (107, 114, 121, 128)


def chronos_history(horizon: int) -> int:
    return CHRONOS_INPUT + horizon
CHRONOS_PRESET = "bolt_small"


@dataclass(frozen=True)
class ModelSpec:
    name: str                      # column name NeuralForecast gives the model
    family: str                    # "nf" or "chronos"
    category: str                  # "Task-specific" or "Foundation model"
    multivariate: bool = False     # passes n_series
    early_stop: int | None = None  # early_stop_patience_steps
    extra: dict = field(default_factory=dict)
    matmul_precision: str | None = None  # torch.set_float32_matmul_precision

    def nf_kwargs(self, horizon: int) -> dict:
        kw = {"h": horizon, **NF_COMMON,
              "early_stop_patience_steps": self.early_stop, **self.extra}
        if self.multivariate:
            kw["n_series"] = N_SERIES
        return kw

    def describe(self) -> dict:
        if self.family == "chronos":
            return {"preset": CHRONOS_PRESET, "history_window": "100 + horizon",
                    "prediction_length": "horizon"}
        d = {k: v for k, v in self.nf_kwargs(horizon=0).items() if k != "h"}
        if self.matmul_precision:
            d["float32_matmul_precision"] = self.matmul_precision
        return d


def _nf(name, mv=False, es=50, **kw):
    return ModelSpec(name=name, family="nf", category="Task-specific",
                     multivariate=mv, early_stop=es, **kw)


MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    _nf("NBEATS"),
    _nf("NLinear"),
    _nf("DLinear"),
    _nf("KAN"),
    _nf("PatchTST"),
    _nf("Informer", extra={"hidden_size": 256, "conv_hidden_size": 64},
        matmul_precision="medium"),
    _nf("TimeMixer", mv=True, es=100),
    _nf("TSMixer", mv=True, es=100),
    _nf("TSMixerx", mv=True, es=100),
    _nf("iTransformer", mv=True, es=100),
    _nf("RMoK", mv=True, es=50),
    ModelSpec(name="Chronos-Bolt", family="chronos", category="Foundation model"),
]}


def models_for(family: str) -> list[str]:
    return [m for m, s in MODELS.items() if s.family == family]
