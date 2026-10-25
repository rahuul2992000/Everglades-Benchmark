"""
Hydrologic model evaluation metrics: NSE, PBIAS, R-squared.

Reference
---------
Moriasi, D.N., Arnold, J.G., Van Liew, M.W., Bingner, R.L., Harmel, R.D.,
Veith, T.L. (2007). Model Evaluation Guidelines for Systematic Quantification
of Accuracy in Watershed Simulations. Transactions of the ASABE 50(3): 885-900.

Formulas are transcribed directly from Moriasi eq. 1 (NSE) and eq. 2 (PBIAS),
and from the standard longhand form of the squared Pearson correlation (R2);
each was verified numerically against a literal implementation of the printed
equation. RSR is also computed since Moriasi recommends it as the third
statistic, but it is optional in the output.

Note Moriasi's caveat on R2: it is "oversensitive to high extreme values
(outliers) and insensitive to additive and proportional differences between
model predictions and measured data." A forecast offset by a constant still
scores R2 = 1.

Performance ratings (Moriasi Table 4, streamflow, monthly time step):

    Rating          RSR              NSE               PBIAS
    Very good       0.00-0.50        0.75-1.00         < +/-10
    Good            0.50-0.60        0.65-0.75         +/-10 to +/-15
    Satisfactory    0.60-0.70        0.50-0.65         +/-15 to +/-25
    Unsatisfactory  > 0.70           < 0.50            > +/-25

Two caveats for forecasting work:
  1. The ratings are for a MONTHLY time step. Moriasi notes that "as the
     evaluation time step increases, a stricter performance rating is
     warranted" - so daily evaluation is judged more leniently, and these
     thresholds are conservative when applied to daily forecasts.
  2. They were developed for calibration/validation of watershed simulation
     models, not for lead-time forecasting. Report the baseline alongside.
"""
import numpy as np
import pandas as pd


def nse(obs, sim):
    """Nash-Sutcliffe efficiency. 1 = perfect; <0 = worse than the observed mean."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if denom == 0 else 1.0 - np.sum((obs - sim) ** 2) / denom


def pbias(obs, sim):
    """Percent bias. 0 = perfect. Positive = model underestimates."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    denom = np.sum(obs)
    return np.nan if denom == 0 else 100.0 * np.sum(obs - sim) / denom


def rsr(obs, sim):
    """RMSE / standard deviation of observations. 0 = perfect."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    sd = np.std(obs, ddof=1)
    return np.nan if sd == 0 else np.sqrt(np.mean((obs - sim) ** 2)) / sd


def r_squared(obs, sim):
    """Squared Pearson correlation. Measures collinearity, not accuracy."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() == 0 or sim.std() == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1] ** 2)


# ---------------------------------------------------------------------------
# Performance ratings
#
# Two published schemes are supported.
#
# "moriasi2015" (default) - Moriasi et al. (2015), as summarised for flow:
#     Very good      NSE > 0.80          |PBIAS| < 5           r2 > 0.85
#     Good           0.70 < NSE <= 0.80  5 <= |PBIAS| < 10     0.75 < r2 <= 0.85
#     Satisfactory   0.50 < NSE <= 0.70  10 <= |PBIAS| < 15    0.60 < r2 <= 0.75
#     Unsatisfactory NSE <= 0.50         |PBIAS| >= 15         r2 <= 0.60
#
# "moriasi2007" - Moriasi et al. (2007) Table 4, streamflow, monthly step:
#     Very good      NSE > 0.75          |PBIAS| < 10          (r2 not rated)
#     Good           0.65 < NSE <= 0.75  10 <= |PBIAS| < 15
#     Satisfactory   0.50 < NSE <= 0.65  15 <= |PBIAS| < 25
#     Unsatisfactory NSE <= 0.50         |PBIAS| >= 25
#
# The 2007 scheme is the source of the commonly quoted "+/-25% PBIAS is
# acceptable" rule. 2015 is the stricter revision and is the only one of the
# two that rates r2, so it is the default here.
#
# Both were developed for calibration/validation of watershed simulation
# models on a monthly time step, not for lead-time forecasting on daily data.
# ---------------------------------------------------------------------------

SCHEMES = {
    "moriasi2015": {
        "NSE":   [(0.80, "very good"), (0.70, "good"), (0.50, "satisfactory")],
        "PBIAS": [(5, "very good"), (10, "good"), (15, "satisfactory")],
        "R2":    [(0.85, "very good"), (0.75, "good"), (0.60, "satisfactory")],
    },
    "moriasi2007": {
        "NSE":   [(0.75, "very good"), (0.65, "good"), (0.50, "satisfactory")],
        "PBIAS": [(10, "very good"), (15, "good"), (25, "satisfactory")],
        "R2":    None,
    },
}
DEFAULT_SCHEME = "moriasi2015"


def _rate_higher_better(v, thresholds):
    if v is None or np.isnan(v):
        return "n/a"
    for cut, label in thresholds:
        if v > cut:
            return label
    return "unsatisfactory"


def _rate_lower_better(v, thresholds):
    if v is None or np.isnan(v):
        return "n/a"
    a = abs(v)
    for cut, label in thresholds:
        if a < cut:
            return label
    return "unsatisfactory"


def rate_nse(v, scheme=DEFAULT_SCHEME):
    return _rate_higher_better(v, SCHEMES[scheme]["NSE"])


def rate_pbias(v, scheme=DEFAULT_SCHEME):
    return _rate_lower_better(v, SCHEMES[scheme]["PBIAS"])


def rate_r2(v, scheme=DEFAULT_SCHEME):
    t = SCHEMES[scheme]["R2"]
    return "n/a" if t is None else _rate_higher_better(v, t)


def rate_rsr(v):
    """Moriasi 2007 only; RSR is not rated in the 2015 flow criteria."""
    if v is None or np.isnan(v):
        return "n/a"
    if v < 0.50:
        return "very good"
    if v < 0.60:
        return "good"
    if v < 0.70:
        return "satisfactory"
    return "unsatisfactory"


def evaluate(obs, sim, include_ratings=True, include_rsr=False,
             scheme=DEFAULT_SCHEME):
    """All metrics for one observed/simulated pair."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    keep = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[keep], sim[keep]
    out = {
        "n": len(obs),
        "MAE": float(np.mean(np.abs(obs - sim))),
        "RMSE": float(np.sqrt(np.mean((obs - sim) ** 2))),
        "NSE": nse(obs, sim),
        "PBIAS": pbias(obs, sim),
        "R2": r_squared(obs, sim),
    }
    if include_rsr:
        out["RSR"] = rsr(obs, sim)
    if include_ratings:
        out["NSE_rating"] = rate_nse(out["NSE"], scheme)
        out["PBIAS_rating"] = rate_pbias(out["PBIAS"], scheme)
        out["R2_rating"] = rate_r2(out["R2"], scheme)
        if include_rsr:
            out["RSR_rating"] = rate_rsr(out["RSR"])
    return out


def evaluate_by_station(df, obs_col, sim_col, station_col="unique_id",
                        stations=None, baseline_col=None, include_rsr=False,
                        scheme=DEFAULT_SCHEME):
    """
    Metrics per station, plus an overall row (mean across stations).
    If baseline_col is given (e.g. persistence), its metrics are added with a
    _baseline suffix so model skill can be judged against it.
    """
    rows = []
    groups = df.groupby(station_col, sort=False)
    for name, g in groups:
        if stations is not None and name not in stations:
            continue
        rec = {"station": name, **evaluate(g[obs_col], g[sim_col], include_rsr=include_rsr,
                                          scheme=scheme)}
        if baseline_col is not None and baseline_col in g:
            base = evaluate(g[obs_col], g[baseline_col], include_ratings=False,
                            include_rsr=include_rsr)
            rec.update({f"{k}_baseline": v for k, v in base.items() if k != "n"})
        rows.append(rec)

    table = pd.DataFrame(rows)
    if len(table):
        numeric = table.select_dtypes(include=[np.number]).columns
        overall = {c: table[c].mean() for c in numeric}
        overall["station"] = "Overall (mean of stations)"
        overall["n"] = int(table["n"].mean())
        overall["NSE_rating"] = rate_nse(overall.get("NSE", np.nan), scheme)
        overall["PBIAS_rating"] = rate_pbias(overall.get("PBIAS", np.nan), scheme)
        overall["R2_rating"] = rate_r2(overall.get("R2", np.nan), scheme)
        if "RSR" in table.columns:
            overall["RSR_rating"] = rate_rsr(overall.get("RSR", np.nan))
        table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    return table
