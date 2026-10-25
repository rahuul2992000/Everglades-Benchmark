"""Scoring, matching NEWgof_all_models: evaluate_by_station on the 5 stations,
overall row = mean across stations, Moriasi 2015 ratings."""
from __future__ import annotations

import math

import pandas as pd

from . import config
from .hydro_metrics import evaluate_by_station

METRICS = ["MAE", "RMSE", "NSE", "PBIAS", "R2"]


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def score_predictions(df: pd.DataFrame, obs_col: str = "y",
                      sim_col: str = "yhat") -> dict:
    """Return {"overall": {...}, "per_station": [{...}, ...]}."""
    df = df.dropna(subset=[obs_col, sim_col])
    t = evaluate_by_station(df, obs_col=obs_col, sim_col=sim_col,
                            stations=config.STATIONS, scheme=config.METRIC_SCHEME)
    rows = [{k: _clean(v) for k, v in r.items()} for r in t.to_dict("records")]
    overall = [r for r in rows if str(r["station"]).startswith("Overall")]
    per_station = [r for r in rows if not str(r["station"]).startswith("Overall")]
    order = {s: i for i, s in enumerate(config.STATIONS)}
    per_station.sort(key=lambda r: order.get(r["station"], 99))
    if len(per_station) != len(config.STATIONS):
        found = {r["station"] for r in per_station}
        missing = [s for s in config.STATIONS if s not in found]
        raise ValueError(f"Predictions are missing stations: {missing}")
    return {"overall": overall[0], "per_station": per_station}
