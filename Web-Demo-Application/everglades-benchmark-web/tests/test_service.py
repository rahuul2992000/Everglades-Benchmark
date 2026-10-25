"""Service behaviour with a stub runner (no ML libraries needed)."""
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from everglades_core import config, data
from everglades_core.service import make_app
from everglades_core.store import RunStore


def stub_forecast(horizon, shift=0.0):
    Y = data.load_long()
    last = Y.groupby("unique_id").tail(1)
    rows = []
    for _, r in last.iterrows():
        for k in range(1, horizon + 1):
            rows.append({"unique_id": r["unique_id"], "ds": r["ds"] + pd.Timedelta(days=k + shift),
                         "yhat": r["y"]})
    return pd.DataFrame(rows)


class StubRunner:
    family = "nf"
    needs_training = True

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def run(self, model, horizon, progress, ctx):
        self.calls += 1
        progress("Training", None)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        ctx.artifacts_dir.mkdir(parents=True)
        (ctx.artifacts_dir / "model.bin").write_text("stub")
        ctx.timings.update(training_seconds=12.5, forecast_seconds=0.2)
        Y = data.load_long()
        test = Y[Y["ds"] >= config.TEST_START].copy()
        test["yhat"] = test["y"] + np.random.default_rng(0).normal(0, 0.05, len(test))
        return test

    def forecast(self, model, horizon, artifacts_dir):
        assert (artifacts_dir / "model.bin").exists()
        return stub_forecast(horizon), {"forecast_seconds": 0.2, "cold_start": True}

    def environment(self):
        return {"stub": True}


def wait(client, rid, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/runs/{rid}").json()
        if s["state"] in ("succeeded", "failed"):
            return s
        time.sleep(0.05)
    raise TimeoutError(s)


@pytest.fixture
def setup(tmp_path):
    runner = StubRunner()
    return TestClient(make_app(runner, RunStore(tmp_path))), runner


def test_run_lifecycle_and_cache(setup):
    client, runner = setup
    r = client.post("/runs", json={"model": "NBEATS", "horizon": 7}).json()
    s = wait(client, r["run_id"])
    assert s["state"] == "succeeded"
    m = client.get(f"/runs/{r['run_id']}/metrics").json()
    assert set(m["metrics"]["overall"]) >= {"MAE", "RMSE", "NSE", "PBIAS", "R2"}
    assert len(m["metrics"]["per_station"]) == 5
    assert m["meta"]["model_config"]["early_stop_patience_steps"] == 50
    rows = client.get(f"/runs/{r['run_id']}/predictions").json()
    assert len(rows) == 5 * 211
    csv = client.get(f"/runs/{r['run_id']}/predictions.csv")
    assert csv.status_code == 200 and csv.text.count("\n") == 37 * 211 + 1
    again = client.post("/runs", json={"model": "NBEATS", "horizon": 7}).json()
    assert again["state"] == "succeeded" and runner.calls == 1


def test_rejects_other_family_and_bad_horizon(setup):
    client, _ = setup
    assert client.post("/runs", json={"model": "Chronos-Bolt", "horizon": 7}).status_code == 400
    assert client.post("/runs", json={"model": "NBEATS", "horizon": 5}).status_code == 400


def test_failure_is_reported(tmp_path):
    client = TestClient(make_app(StubRunner(fail=True), RunStore(tmp_path)))
    r = client.post("/runs", json={"model": "KAN", "horizon": 14}).json()
    s = wait(client, r["run_id"])
    assert s["state"] == "failed" and "out of memory" in s["error"]
    assert client.get(f"/runs/{r['run_id']}/metrics").status_code == 409


def test_restart_marks_interrupted_runs(tmp_path):
    store = RunStore(tmp_path)
    store.set_status("abc", model="NBEATS", horizon=7, state="running")
    make_app(StubRunner(), store)
    assert store.status("abc")["state"] == "failed"


def test_forecast_needs_training_then_works(setup):
    client, _ = setup
    r = client.post("/forecast", json={"model": "NBEATS", "horizon": 14})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "not_trained"
    run = client.post("/runs", json={"model": "NBEATS", "horizon": 14}).json()
    assert wait(client, run["run_id"])["state"] == "succeeded"
    m = client.get(f"/runs/{run['run_id']}/metrics").json()
    assert m["meta"]["timings"]["training_seconds"] == 12.5
    f = client.post("/forecast", json={"model": "NBEATS", "horizon": 14}).json()
    assert f["origin"] == "2024-08-26" and len(f["rows"]) == 37 * 14
    days = sorted({row["ds"] for row in f["rows"]})
    assert days[0] == "2024-08-27" and days[-1] == "2024-09-09"
    assert f["timings"]["training_seconds"] == 12.5 and f["timings"]["forecast_seconds"] == 0.2


class ZeroShotStub(StubRunner):
    family = "chronos"
    needs_training = False

    def forecast(self, model, horizon, artifacts_dir):
        return stub_forecast(horizon), {"forecast_seconds": 0.1, "cold_start": False}


def test_zero_shot_forecast_needs_no_run(tmp_path):
    client = TestClient(make_app(ZeroShotStub(), RunStore(tmp_path)))
    f = client.post("/forecast", json={"model": "Chronos-Bolt", "horizon": 7})
    assert f.status_code == 200
    assert f.json()["timings"]["training_seconds"] is None


def test_forecast_with_wrong_dates_is_rejected(tmp_path):
    class Bad(ZeroShotStub):
        def forecast(self, model, horizon, artifacts_dir):
            return stub_forecast(horizon, shift=-1), {}
    client = TestClient(make_app(Bad(), RunStore(tmp_path)), raise_server_exceptions=False)
    assert client.post("/forecast", json={"model": "Chronos-Bolt", "horizon": 7}).status_code == 500
