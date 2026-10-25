import os

import pandas as pd
import streamlit as st

import client
import ui
from everglades_core import config

ANONYMOUS = os.environ.get("ANONYMOUS", "1") == "1"
PAPER_URL = os.environ.get("PAPER_URL", "")
CODE_URL = os.environ.get("CODE_URL", "")

st.title("About this benchmark")
ui.lede("This site runs eleven task-specific forecasting models and one time "
        "series foundation model on Everglades water-level data, using the same "
        "data, settings, and scoring as the accompanying study.")

st.subheader("How runs are evaluated")
st.markdown(f"""
The data are daily series from October 16, 2020 to August 26, 2024. The final
{config.TEST_SIZE} days, {config.TEST_START} to {config.TEST_END}, are the test period.

**Task-specific models** are trained with NeuralForecast on the first
{1411 - config.TEST_SIZE} days, of which the last {config.VAL_SIZE} are used for
validation and early stopping. They are then evaluated with rolling
cross-validation over the test period.

**Chronos-Bolt** is used zero-shot, without training. For each day in the test
period it forecasts from the preceding 100 of history

**Scoring.** MAE, RMSE, NSE, PBIAS, and R² are computed for each of the five
stations ({", ".join(config.STATION_LABELS.values())}), and the overall value is
their mean. Ratings follow Moriasi et al. (2015).
""")

st.subheader("Forecasts and timing")
st.markdown("""
The Forecast page predicts the 7 to 28 days after the last day in the data,
August 26, 2024, using the same settings as the evaluation. Task-specific models
use the model trained during their evaluation; Chronos-Bolt needs no training.
Forecasts have no observed values to compare with, so no accuracy metrics are shown.

**Timing.** Three times are reported, because they answer different questions:

- **One forecast**: how long a single forecast takes, which is what someone using a
  model day to day waits for. It is the median of five repeated forecasts after a
  warm-up call.
- **Training**: the one-time cost of fitting a task-specific model. Chronos-Bolt is
  zero-shot and has none.

Times are measured on the server's GPU, which is shared with other jobs, so they
vary slightly between runs.
""")

st.subheader("Model settings")
common = ", ".join(f"{k}={v}" for k, v in config.NF_COMMON.items())
st.markdown(f"All task-specific models use {common}, NeuralForecast's default "
            f"random seed (1), and forecast horizon h equal to the selected horizon. "
            f"Chronos-Bolt uses the {config.CHRONOS_PRESET} preset with a "
            f"history window of 100 days.")

st.subheader("Software")
for fam, name in [("nf", "Task-specific model service"), ("chronos", "Chronos-Bolt service")]:
    h = client.health(fam)
    if h is None:
        st.markdown(f"**{name}**: offline")
    else:
        env = ", ".join(f"{k} {v}" for k, v in h["environment"].items() if v and k != "host")
        st.markdown(f"**{name}**: {env}")

if not ANONYMOUS and (PAPER_URL or CODE_URL):
    st.subheader("Paper and code")
    if PAPER_URL:
        st.markdown(f"[Read the paper]({PAPER_URL})")
    if CODE_URL:
        st.markdown(f"[View the code]({CODE_URL})")
