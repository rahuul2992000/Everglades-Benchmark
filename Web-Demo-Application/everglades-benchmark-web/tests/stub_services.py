"""Development only: serve both families with a stub runner that returns noisy
fake predictions, so the website can be exercised without GPU libraries.
Never deploy this.   python tests/stub_services.py"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from everglades_core import config, data  # noqa: E402
from everglades_core.service import make_app  # noqa: E402
from everglades_core.store import RunStore  # noqa: E402


class Stub:
    def __init__(self, family, delay):
        self.family, self.delay = family, delay

    @property
    def needs_training(self):
        return self.family == "nf"

    def run(self, model, horizon, progress, ctx):
        for i in range(5):
            progress("Stub run", (i + 1) / 5)
            time.sleep(self.delay / 5)
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        ctx.timings.update(training_seconds=(8.0 if self.needs_training else None),
                           forecast_seconds=0.05 if self.needs_training else 0.3)
        Y = data.load_long()
        t = Y[Y["ds"] >= config.TEST_START].copy()
        rng = np.random.default_rng(abs(hash((model, horizon))) % 2**32)
        t["yhat"] = t["y"] + rng.normal(0, 0.02 * horizon ** 0.5, len(t))
        return t

    def forecast(self, model, horizon, artifacts_dir):
        Y = data.load_long()
        rows = []
        rng = np.random.default_rng(abs(hash((model, horizon, "f"))) % 2**32)
        for uid, g in Y.groupby("unique_id"):
            last = g["y"].iloc[-1]
            drift = np.cumsum(rng.normal(0, 0.03, horizon))
            for k in range(horizon):
                rows.append({"unique_id": uid, "ds": g["ds"].iloc[-1] + pd.Timedelta(days=k + 1),
                             "yhat": last + drift[k]})
        return pd.DataFrame(rows), {"forecast_seconds": 0.05, "warmup_seconds": 0.4,
                                    "load_seconds": 1.2, "cold_start": True, "repeats": 5}

    def environment(self):
        return {"stub": "not a real model"}


def serve(fam, port, delay):
    app = make_app(Stub(fam, delay), RunStore(Path("/tmp/eb-stub-store")))
    uvicorn.run(app, port=port, log_level="warning")


if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    threading.Thread(target=serve, args=("nf", 8001, delay), daemon=True).start()
    serve("chronos", 8002, delay)
