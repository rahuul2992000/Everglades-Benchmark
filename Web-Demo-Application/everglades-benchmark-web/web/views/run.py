import pandas as pd
import streamlit as st

import charts
import client
import ui
from everglades_core import config

TASK = [m for m, s in config.MODELS.items() if s.category == "Task-specific"]
FOUNDATION = [m for m, s in config.MODELS.items() if s.category == "Foundation model"]

st.title("Evaluate a model")
ui.lede("Pick a model and a forecast horizon. The model is evaluated on the "
        "benchmark's test period, January 29 to August 26, 2024, and scored against "
        "observed water levels at five Everglades stations.")

left, right = st.columns([1.35, 3], gap="large")

with left:
    kind = st.segmented_control("Model type", ["Task-specific", "Foundation model"],
                                default="Task-specific", key="run_kind")
    options = FOUNDATION if kind == "Foundation model" else TASK
    model = st.selectbox("Model", options, key=f"run_model_{kind}")
    horizon = st.segmented_control("Horizon (days)", list(config.HORIZONS), default=7,
                                   format_func=ui.horizon_short, key="run_horizon")
    clicked = st.button("Run evaluation", type="primary", width="stretch",
                        disabled=model is None or horizon is None)
    st.markdown('<p class="eb-note">Training and evaluation can take a few minutes. '
                'A configuration that has already run shows its results right away.</p>',
                unsafe_allow_html=True)

if clicked:
    try:
        s = client.submit(model, horizon)
        st.session_state["run"] = {"model": model, "horizon": horizon,
                                   "run_id": s["run_id"], "status": s}
    except client.ServiceError as e:
        st.session_state["run"] = None
        with right:
            st.error(str(e))

run = st.session_state.get("run")


@st.fragment(run_every=2)
def poll(run: dict):
    try:
        s = client.status(run["model"], run["run_id"])
    except client.ServiceError as e:
        st.warning(str(e))
        return
    if s["state"] in client.TERMINAL:
        run["status"] = s
        st.rerun()
    st.markdown(f'<div class="eb-run-title">{run["model"]}, '
                f'{ui.horizon_label(run["horizon"])}</div>', unsafe_allow_html=True)
    frac = s.get("progress")
    st.progress(frac if frac is not None else 0.0, text=ui.stage_text(s))
    if frac is None and s["state"] == "running":
        st.caption("This step doesn't report progress. The page updates when it finishes.")


@st.cache_data(show_spinner=False, max_entries=64)
def load_result(model: str, rid: str):
    return client.metrics(model, rid), client.predictions(model, rid)


with right:
    if not run:
        st.info("Choose a model and horizon, then select Run evaluation.")
    elif run["status"]["state"] in ("queued", "running"):
        poll(run)
    elif run["status"]["state"] == "failed":
        st.markdown(f'<div class="eb-run-title">{run["model"]}, '
                    f'{ui.horizon_label(run["horizon"])}</div>', unsafe_allow_html=True)
        st.error("This run didn't finish. Select Run evaluation to try again.")
        with st.expander("Details"):
            st.code(run["status"].get("error") or "No error message was recorded.")
    else:
        try:
            result, preds = load_result(run["model"], run["run_id"])
        except client.ServiceError as e:
            st.error(str(e))
            st.stop()
        m, meta = result["metrics"], result["meta"]
        st.markdown(
            f'<div class="eb-run-title">{run["model"]}, {ui.horizon_label(run["horizon"])}</div>'
            f'<div class="eb-run-sub">Evaluation took {ui.runtime_text(meta.get("runtime_seconds"))}'
            f' on {meta["environment"].get("gpu") or "CPU"}'
            + "".join(f", {k.lower()}: {v}" for k, v in ui.timing_summary(meta.get("timings")).items() if v)
            + '</div>',
            unsafe_allow_html=True)

        station = st.segmented_control("Station", config.STATIONS,
                                       default=config.STATIONS[0],
                                       format_func=ui.station_label, key="run_station")
        station = station or config.STATIONS[0]
        st.altair_chart(charts.hydrograph(preds[preds["unique_id"] == station]),
                        width="stretch")

        st.markdown("**Across all five stations**")
        ui.metric_cards(m["overall"])
        st.markdown('<p class="eb-note">Overall values are the mean of the five '
                    'stations. Ratings follow Moriasi et al. (2015).</p>',
                    unsafe_allow_html=True)

        rows = [{"Station": ui.station_label(r["station"]),
                 **{ui.METRIC_LABEL.get(k, k): r[k] for k in ["MAE", "RMSE", "NSE", "R2"]},
                 "PBIAS": ui.clean_zero(r["PBIAS"]),
                 "NSE rating": (r.get("NSE_rating") or "").capitalize()}
                for r in m["per_station"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.3f")
                                    for k in ["MAE", "RMSE", "NSE", "R²"]}
                     | {"PBIAS": st.column_config.NumberColumn(format="%.2f")})

        c1, c2 = st.columns([1, 2])
        with c1:
            try:
                st.download_button(
                    "Download predictions (CSV)",
                    client.predictions_csv(run["model"], run["run_id"]),
                    file_name=f"{run['model']}_{run['horizon']}days_predictions.csv",
                    mime="text/csv")
            except client.ServiceError as e:
                st.caption(str(e))
        with st.expander("Run details"):
            st.json({"model settings": meta["model_config"],
                     "test period": meta["test_period"],
                     "data file": meta["data_file"],
                     "data sha256": meta["data_sha256"],
                     "software": {k: v for k, v in meta["environment"].items() if k != "host"},
                     "run id": meta["run_id"]}, expanded=False)
