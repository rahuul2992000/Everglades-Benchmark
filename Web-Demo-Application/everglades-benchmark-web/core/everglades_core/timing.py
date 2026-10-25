"""Timing for single forecasts: one warm-up call, then the median of several.

GPUs on shared servers are noisy and the first call pays one-time warm-up
costs, so a single measurement would mislead. The warm-up call's result is
the one returned, so timing never changes what is forecast.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

FORECAST_REPEATS = 5


def timed_forecast(fn: Callable[[], object], repeats: int = FORECAST_REPEATS):
    """Return (result of the first call, timing dict)."""
    t0 = time.perf_counter()
    result = fn()
    warmup = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return result, {
        "forecast_seconds": round(statistics.median(samples), 4),
        "forecast_seconds_samples": [round(s, 4) for s in samples],
        "warmup_seconds": round(warmup, 4),
        "repeats": repeats,
    }
