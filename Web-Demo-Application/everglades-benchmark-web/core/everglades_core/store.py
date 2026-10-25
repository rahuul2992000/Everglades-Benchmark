"""On-disk store for runs: status, predictions, metrics, and metadata.

Layout:  <root>/runs/<run_id>/{status.json, predictions.csv, metrics.json,
                                meta.json, error.log}
A run_id is a hash of (model, horizon, CONFIG_VERSION, data fingerprint), so
the same configuration always maps to the same run and is computed once.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import config
from .data import fingerprint

STATES = ("queued", "running", "succeeded", "failed")


def default_root() -> Path:
    return Path(os.environ.get("EVERGLADES_STORE", config.REPO_ROOT / "store"))


def run_id(model: str, horizon: int) -> str:
    payload = json.dumps({"model": model, "horizon": int(horizon),
                          "config": config.CONFIG_VERSION,
                          "data": fingerprint()}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class RunStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_root()
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def dir(self, rid: str) -> Path:
        return self.root / "runs" / rid

    # -- status --------------------------------------------------------------
    def status(self, rid: str) -> dict | None:
        p = self.dir(rid) / "status.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def set_status(self, rid: str, **fields) -> dict:
        cur = self.status(rid) or {"run_id": rid}
        cur.update(fields, updated_at=time.time())
        _atomic_write(self.dir(rid) / "status.json", json.dumps(cur, indent=2))
        return cur

    def all_statuses(self) -> list[dict]:
        out = []
        for d in (self.root / "runs").iterdir():
            s = self.status(d.name) if d.is_dir() else None
            if s:
                out.append(s)
        return out

    # -- results -------------------------------------------------------------
    def save_result(self, rid: str, predictions: pd.DataFrame,
                    metrics: dict, meta: dict) -> None:
        d = self.dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".csv")
        os.close(fd)
        predictions.to_csv(tmp, index=False)
        os.replace(tmp, d / "predictions.csv")
        _atomic_write(d / "metrics.json", json.dumps(metrics, indent=2))
        _atomic_write(d / "meta.json", json.dumps(meta, indent=2, default=str))

    def predictions(self, rid: str) -> pd.DataFrame:
        return pd.read_csv(self.dir(rid) / "predictions.csv", parse_dates=["ds"])

    def metrics(self, rid: str) -> dict:
        return json.loads((self.dir(rid) / "metrics.json").read_text())

    def meta(self, rid: str) -> dict:
        return json.loads((self.dir(rid) / "meta.json").read_text())

    def artifacts_dir(self, rid: str) -> Path:
        return self.dir(rid) / "model"

    def write_error(self, rid: str, text: str) -> None:
        _atomic_write(self.dir(rid) / "error.log", text)
