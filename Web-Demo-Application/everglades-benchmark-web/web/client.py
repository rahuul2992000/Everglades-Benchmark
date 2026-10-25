"""Talks to the two model services. The website never runs models itself."""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

from everglades_core import config

URLS = {
    "nf": os.environ.get("NF_SERVICE_URL", "http://localhost:8001"),
    "chronos": os.environ.get("CHRONOS_SERVICE_URL", "http://localhost:8002"),
}
TERMINAL = ("succeeded", "failed")


class ServiceError(Exception):
    """Raised with a message that is safe to show visitors."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


class NotTrained(ServiceError):
    """The model has to be trained (evaluated once) before it can forecast."""


def family(model: str) -> str:
    return config.MODELS[model].family


def _call(fam: str, method: str, path: str, **kw):
    url = URLS[fam] + path
    try:
        r = requests.request(method, url, timeout=kw.pop("timeout", 15), **kw)
    except requests.RequestException as exc:
        name = "Chronos-Bolt" if fam == "chronos" else "task-specific model"
        raise ServiceError(f"The {name} service is offline. Try again in a few minutes.",
                           f"{method} {url}: {exc}") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, dict) and detail.get("code") == "not_trained":
            raise NotTrained(detail.get("message", "Model not trained yet."))
        if isinstance(detail, dict):
            detail = detail.get("message", str(detail))
        raise ServiceError(str(detail), f"{method} {url} -> {r.status_code}")
    return r


def submit(model: str, horizon: int) -> dict:
    return _call(family(model), "POST", "/runs",
                 json={"model": model, "horizon": horizon}).json()


def status(model: str, rid: str) -> dict:
    return _call(family(model), "GET", f"/runs/{rid}").json()


def metrics(model: str, rid: str) -> dict:
    return _call(family(model), "GET", f"/runs/{rid}/metrics").json()


def predictions(model: str, rid: str) -> pd.DataFrame:
    rows = _call(family(model), "GET", f"/runs/{rid}/predictions", timeout=30).json()
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def predictions_csv(model: str, rid: str) -> bytes:
    return _call(family(model), "GET", f"/runs/{rid}/predictions.csv", timeout=60).content


def forecast(model: str, horizon: int) -> dict:
    res = _call(family(model), "POST", "/forecast",
                json={"model": model, "horizon": horizon}, timeout=180).json()
    df = pd.DataFrame(res["rows"])
    df["ds"] = pd.to_datetime(df["ds"])
    res["frame"] = df
    return res


def health(fam: str) -> dict | None:
    try:
        return _call(fam, "GET", "/health", timeout=5).json()
    except ServiceError:
        return None


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))
