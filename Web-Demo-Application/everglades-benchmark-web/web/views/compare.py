import pandas as pd
import streamlit as st

import charts
import client
import ui
from everglades_core import config

ALL = list(config.MODELS)
DEFAULT = ["Chronos-Bolt", "NBEATS", "PatchTST", "NLinear"]
METRICS = ["MAE", "RMSE", "NSE", "PBIAS", "R2"]
LOWER_BETTER = {"MAE": True, "RMSE": True, "NSE": False, "R2": False, "PBIAS": True}

st.title("Compare models")
ui.lede("Run several models on the same test period and see how they rank, how "
        "their forecasts track the observed water level, and how accuracy changes "
        "as the horizon grows.")


def submit_batch(key: str, pairs: list[tuple[str, int]]):
    runs, errors = [], []
    for model, h in pairs:
        try:
            s = client.submit(model, h)
            runs.append({"model": model, "horizon": h, "run_id": s["run_id"], "status": s})
        except client.ServiceError as e:
            errors.append(f"{model}, {ui.horizon_label(h)}: {e}")
    st.session_state[key] = runs
    for e in errors:
        st.error(e)


def pending(runs):
    return [r for r in runs if r["status"]["state"] not in client.TERMINAL]


@st.fragment(run_every=3)
def poll(key: str):
    runs = st.session_state.get(key, [])
    for r in pending(runs):
        try:
            r["status"] = client.status(r["model"], r["run_id"])
        except client.ServiceError:
            pass
    if not pending(runs):
        st.rerun()
    done = sum(r["status"]["state"] in client.TERMINAL for r in runs)
    st.progress(done / len(runs), text=f"{done} of {len(runs)} runs finished")
    st.dataframe(pd.DataFrame([{
        "Model": r["model"], "Horizon": ui.horizon_label(r["horizon"]),
        "Status": ui.stage_text(r["status"])} for r in runs]),
        hide_index=True, width="stretch")


@st.cache_data(show_spinner=False, max_entries=256)
def load(model: str, rid: str):
    return client.metrics(model, rid), client.predictions(model, rid)


def finished(runs):
    out = []
    for r in runs:
        if r["status"]["state"] == "succeeded":
            try:
                res, preds = load(r["model"], r["run_id"])
                out.append((r, res, preds))
            except client.ServiceError as e:
                st.warning(f"{r['model']}: {e}")
    failed = [r for r in runs if r["status"]["state"] == "failed"]
    if failed:
        st.warning("These runs didn't finish: " + ", ".join(
            f"{r['model']} ({ui.horizon_label(r['horizon'])})" for r in failed)
            + ". Run them again to retry.")
    return out


# ---------------------------------------------------------------------------
st.subheader("One horizon, several models")
c1, c2 = st.columns([3, 1])
with c1:
    models = st.multiselect("Models", ALL, default=DEFAULT, key="cmp_models")
with c2:
    horizon = st.selectbox("Horizon", config.HORIZONS, format_func=ui.horizon_label,
                           key="cmp_horizon")
if st.button("Run selected models", type="primary", disabled=not models):
    submit_batch("cmp_runs", [(m, horizon) for m in models])

runs = st.session_state.get("cmp_runs", [])
if runs and pending(runs):
    poll("cmp_runs")
elif runs:
    done = finished(runs)
    if done:
        h = done[0][0]["horizon"]
        rank_by = st.segmented_control("Rank by", METRICS, default="NSE",
                                       format_func=lambda k: ui.METRIC_LABEL.get(k, k),
                                       key="cmp_rank") or "NSE"
        table = pd.DataFrame([{"Model": r["model"],
                               **{ui.METRIC_LABEL.get(k, k): (
                                   ui.clean_zero(res["metrics"]["overall"][k]) if k == "PBIAS"
                                   else res["metrics"]["overall"][k]) for k in METRICS},
                               "NSE rating": (res["metrics"]["overall"].get("NSE_rating")
                                              or "").capitalize(),
                               **ui.timing_summary(res["meta"].get("timings"))}
                               #"Evaluation": ui.runtime_text(res["meta"].get("runtime_seconds"))}
                              for r, res, _ in done])
        col = ui.METRIC_LABEL.get(rank_by, rank_by)
        key = table[col].abs() if rank_by == "PBIAS" else table[col]
        table = table.loc[key.sort_values(ascending=LOWER_BETTER[rank_by]).index]
        table.insert(0, "Rank", range(1, len(table) + 1))
        st.markdown(f"**Overall results, {ui.horizon_label(h)}** "
                    f'<span class="eb-note">mean of five stations</span>',
                    unsafe_allow_html=True)
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.3f")
                                    for k in ["MAE", "RMSE", "NSE", "R²"]}
                     | {"PBIAS": st.column_config.NumberColumn(format="%.2f")})

        station = st.segmented_control("Station", config.STATIONS, default=config.STATIONS[0],
                                       format_func=ui.station_label,
                                       key="cmp_station") or config.STATIONS[0]
        frames = {r["model"]: p[p["unique_id"] == station] for r, _, p in done}
        st.altair_chart(charts.overlay(frames), width="stretch")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("Accuracy across horizons")
ui.lede("Runs the selected models at 7, 14, 21, and 28 days. Runs that have "
        "already finished load right away.")
hm = st.multiselect("Models", ALL, default=DEFAULT, key="hz_models")
n = len(hm) * len(config.HORIZONS)
if st.button(f"Run {n} configurations", disabled=not hm):
    submit_batch("hz_runs", [(m, h) for m in hm for h in config.HORIZONS])

hz = st.session_state.get("hz_runs", [])
if hz and pending(hz):
    poll("hz_runs")
elif hz:
    done = finished(hz)
    if done:
        metric = st.segmented_control("Metric", METRICS, default="NSE",
                                      format_func=lambda k: ui.METRIC_LABEL.get(k, k),
                                      key="hz_metric") or "NSE"
        df = pd.DataFrame([{"model": r["model"], "horizon": r["horizon"],
                            metric: res["metrics"]["overall"][metric]}
                           for r, res, _ in done])
        st.altair_chart(charts.by_horizon(df, metric, ui.METRIC_LABEL.get(metric, metric)),
                        width="stretch")
        st.markdown(f'<p class="eb-note">{ui.METRIC_HELP[metric]}</p>',
                    unsafe_allow_html=True)
