"""parity_check.py reports a match when a store run equals the reference."""
import subprocess
import sys

import pandas as pd

from conftest import FIXTURES, ROOT
from everglades_core.jobs import validate
from everglades_core.scoring import score_predictions
from everglades_core.store import RunStore, run_id


def test_parity_script_matches(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "store")
    for h in (7, 28):
        ref = pd.read_csv(FIXTURES / f"Chronos_predictions_{h}days.csv")
        pred = validate(ref.rename(columns={f"Chronos-{h}": "yhat", "target": "y"}))
        rid = run_id("Chronos-Bolt", h)
        store.save_result(rid, pred, score_predictions(pred), {})
        store.set_status(rid, model="Chronos-Bolt", horizon=h, state="succeeded")
    out = tmp_path / "report.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "parity_check.py"),
                        "--chronos-dir", str(FIXTURES), "--informer-dir", str(FIXTURES),
                        "--store", str(tmp_path / "store"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = pd.read_csv(out)
    got = rep[rep.model == "Chronos-Bolt"].set_index("horizon")["result"]
    assert got[7] == "match" and got[28] == "match"
    assert rep.loc[(rep.model == "Chronos-Bolt") & (rep.horizon == 7), "max_abs_diff_yhat"].iloc[0] == 0
    assert (rep[rep.model == "NBEATS"]["result"] == "no finished run").all()
