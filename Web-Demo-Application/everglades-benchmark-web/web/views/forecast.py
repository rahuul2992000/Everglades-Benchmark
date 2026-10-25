import pandas as pd
import streamlit as st

import charts
import client
import ui
from everglades_core import config, data

TASK = [m for m, s in config.MODELS.items() if s.category == "Task-specific"]
FOUNDATION = [m for m, s in config.MODELS.items() if s.category == "Foundation model"]
ORIGIN = pd.Timestamp(config.TEST_END)


def history_days(horizon: int) -> int:
    """Enough observed days for context while keeping the forecast readable."""
    return max(4 * horizon, 42)

st.title("Forecast")
ui.lede("Forecast the next 7 to 28 days from the most recent data in the benchmark, "
        "August 26, 2024. Forecasts show what each model predicts and how long one "
        "forecast takes to compute. There are no observed values to compare with.")


@st.cache_data(show_spinner=False)
def recent_history(days: int):
    long = data.load_long()
    return long[long["ds"] > ORIGIN - pd.Timedelta(days=days)]


left, right = st.columns([1.35, 3], gap="large")
with left:
    kind = st.segmented_control("Model type", ["Task-specific", "Foundation model"],
                                default="Foundation model", key="fc_kind")
    options = FOUNDATION if kind == "Foundation model" else TASK
    model = st.selectbox("Model", options, key=f"fc_model_{kind}")
    horizon = st.segmented_control("Horizon (days)", list(config.HORIZONS), default=7,
                                   format_func=ui.horizon_short, key="fc_horizon")
    clicked = st.button("Make forecast", type="primary", width="stretch",
                        disabled=model is None or horizon is None)
    st.markdown('<p class="eb-note">Task-specific models are trained once, the first '
                'time they are evaluated. After that, a forecast takes seconds. '
                'Chronos-Bolt needs no training.</p>', unsafe_allow_html=True)


def request_forecast(model, horizon):
    try:
        st.session_state["fc"] = client.forecast(model, horizon)
        st.session_state["fc_train"] = None
    except client.NotTrained as e:
        st.session_state["fc"] = None
        st.session_state["fc_train"] = {"model": model, "horizon": horizon,
                                        "message": str(e), "run": None}
    except client.ServiceError as e:
        st.session_state["fc"] = None
        st.session_state["fc_error"] = str(e)


if clicked:
    st.session_state["fc_error"] = None
    with right, st.spinner("Forecasting..."):
        request_forecast(model, horizon)


@st.fragment(run_every=3)
def poll_training(pending: dict):
    run = pending["run"]
    try:
        s = client.status(pending["model"], run["run_id"])
    except client.ServiceError as e:
        st.warning(str(e))
        return
    run["status"] = s
    if s["state"] == "succeeded":
        request_forecast(pending["model"], pending["horizon"])
        st.rerun()
    if s["state"] == "failed":
        st.rerun()
    frac = s.get("progress")
    st.progress(frac if frac is not None else 0.0, text=ui.stage_text(s))


with right:
    pending = st.session_state.get("fc_train")
    res = st.session_state.get("fc")
    if st.session_state.get("fc_error"):
        st.error(st.session_state["fc_error"])
    elif pending:
        st.markdown(f'<div class="eb-run-title">{pending["model"]}, '
                    f'{ui.horizon_label(pending["horizon"])}</div>', unsafe_allow_html=True)
        run = pending.get("run")
        if run is None:
            st.info(f'{pending["message"]} Training runs the model\'s full evaluation '
                    'once and saves the trained model. It can take a few minutes.')
            if st.button("Train model"):
                try:
                    s = client.submit(pending["model"], pending["horizon"])
                    pending["run"] = {"run_id": s["run_id"], "status": s}
                    st.rerun()
                except client.ServiceError as e:
                    st.error(str(e))
        elif run["status"]["state"] == "failed":
            st.error("Training didn't finish. Select Train model to try again.")
            with st.expander("Details"):
                st.code(run["status"].get("error") or "No error message was recorded.")
            pending["run"] = None
        else:
            poll_training(pending)
    elif not res:
        st.info("Choose a model and horizon, then select Make forecast.")
    else:
        fc, t = res["frame"], res["timings"]
        st.markdown(
            f'<div class="eb-run-title">{res["model"]}, next {res["horizon"]} days</div>'
            f'<div class="eb-run-sub">From the latest data in the benchmark, '
            f'{ORIGIN.strftime("%B %-d, %Y")}. Forecast covers '
            f'{fc["ds"].min().strftime("%B %-d")} to {fc["ds"].max().strftime("%B %-d, %Y")}.'
            f'</div>', unsafe_allow_html=True)

        station = st.segmented_control("Station", config.STATIONS, default=config.STATIONS[0],
                                       format_func=ui.station_label,
                                       key="fc_station") or config.STATIONS[0]
        hist = recent_history(history_days(res["horizon"]))
        st.altair_chart(charts.forecast_chart(hist[hist["unique_id"] == station],
                                              fc[fc["unique_id"] == station], ORIGIN),
                        width="stretch")

        gpu = res["environment"].get("gpu") or "CPU"
        training = t.get("training_seconds")
        cells = [("One forecast", ui.runtime_text(t.get("forecast_seconds")),
                  f"median of {t.get('repeats', 5)} runs"),
                 ("Training", "None" if not t.get("needs_training") else
                  ui.runtime_text(training) or "n/a",
                  "zero-shot model" if not t.get("needs_training") else "one-time, when first evaluated")]
        if t.get("cold_start") and t.get("load_seconds") is not None:
            cells.append(("Model load", ui.runtime_text(t["load_seconds"]),
                          "first forecast after the service starts"))
        ui.timing_cards(cells)
        shared = (" The GPU is shared with other jobs, so times vary slightly between runs."
                  if gpu != "CPU" else "")
        st.markdown(f'<p class="eb-note">Measured on {gpu}. One forecast is the median of '
                    f'{t.get("repeats", 5)} repeated forecasts after a warm-up call.{shared}</p>',
                    unsafe_allow_html=True)

        table = fc[fc["unique_id"] == station][["ds", "yhat"]].rename(
            columns={"ds": "Date", "yhat": "Forecast water level (ft)"})
        table["Date"] = table["Date"].dt.date
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Forecast water level (ft)":
                                    st.column_config.NumberColumn(format="%.3f")})
        out = fc.assign(ds=fc["ds"].dt.strftime("%Y-%m-%d")).rename(columns={"yhat": "forecast"})
        st.download_button("Download forecast (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"{res['model']}_{res['horizon']}days_forecast.csv",
                           mime="text/csv")
