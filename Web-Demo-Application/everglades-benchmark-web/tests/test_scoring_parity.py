"""The site's scoring path must reproduce the published table exactly."""
import pandas as pd
import pytest

from conftest import FIXTURES
from everglades_core.jobs import validate
from everglades_core.scoring import score_predictions

TABLE = pd.read_csv(FIXTURES / "results_all_horizons.csv")
TABLE["Model"] = TABLE["Model"].ffill()


def load(model, h):
    if model == "Chronos":
        d = pd.read_csv(FIXTURES / f"Chronos_predictions_{h}days.csv")
        return d.rename(columns={f"Chronos-{h}": "yhat", "target": "y"})
    d = pd.read_csv(FIXTURES / f"Informer-Results-{h}days.csv")
    return d.rename(columns={"Informer": "yhat"})


@pytest.mark.parametrize("model", ["Chronos", "Informer"])
@pytest.mark.parametrize("h", [7, 14, 21, 28])
def test_metrics_match_published(model, h):
    m = score_predictions(validate(load(model, h)))
    o = m["overall"]
    pub = TABLE[(TABLE.Model == model) & (TABLE["Lead Time"] == h)].iloc[0]
    for ours, theirs in [("MAE", "Overall_MAE"), ("RMSE", "Overall_RMSE"),
                         ("NSE", "Overall_NSE"), ("PBIAS", "Overall_PBIAS"),
                         ("R2", "Overall_R2")]:
        assert round(o[ours], 3) == pytest.approx(pub[theirs], abs=1e-9), ours
    for st in m["per_station"]:
        col = f"{st['station']}_MAE"
        assert round(st["MAE"], 3) == pytest.approx(pub[col], abs=1e-9), col
