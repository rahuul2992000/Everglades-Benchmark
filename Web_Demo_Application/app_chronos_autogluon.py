import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ======================================================================
# CONFIG
# ======================================================================

DATA_FILEPATH = "/aul/homes/rrang016/Downloads/final_concatenated_data.csv"
HISTORY_WINDOW_SIZE = 107       # same as your script
PREDICTION_LENGTH = 7           # next 7 days
FIRST_DATE = pd.to_datetime("2024-01-29")  # same as your script

# ======================================================================
# DATA LOADING (same logic as your VSCode code)
# ======================================================================

@st.cache_data
def load_long_data(csv_path: str):
    data = pd.read_csv(csv_path)

    # Ensure datetime
    data["date"] = pd.to_datetime(data["date"])

    # Melt to long format
    long_df = pd.melt(
        data,
        id_vars=["date"],
        value_vars=[col for col in data.columns if col != "date"],
        var_name="unique_id",
        value_name="y",
    )
    long_df.rename(columns={"date": "ds"}, inplace=True)
    long_df = long_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    # Chronos expects 'target'
    Y_df = long_df.rename(columns={"y": "target"}).reset_index(drop=True)

    station_uids = sorted(Y_df["unique_id"].unique())
    all_dates = np.sort(Y_df["ds"].unique())

    return Y_df, station_uids, all_dates.min(), all_dates.max()


# ======================================================================
# CHRONOS (AutoGluon) – TRAIN ONCE
# ======================================================================

@st.cache_resource
def train_chronos_predictor(Y_df: pd.DataFrame, first_date: pd.Timestamp):
    """
    Exactly your logic:
      - Use data up to first_date
      - Take last HISTORY_WINDOW_SIZE per series
      - Fit TimeSeriesPredictor with presets='bolt_small'
    """
    initial_train_data = Y_df[Y_df["ds"] <= first_date].copy()
    initial_train_data = (
        initial_train_data
        .groupby("unique_id")
        .tail(HISTORY_WINDOW_SIZE)
    )

    initial_train_ts = TimeSeriesDataFrame.from_data_frame(
        initial_train_data,
        id_column="unique_id",
        timestamp_column="ds",
    )

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH
    ).fit(
        initial_train_ts,
        presets="bolt_small",
        # hyperparameters={"enable_gpu": True},  # if available
    )

    return predictor


# ======================================================================
# STREAMLIT APP
# ======================================================================

st.set_page_config(layout="wide")
st.title("Chronos-Bolt Zero-Shot 7-Day Forecast")

with st.spinner("Loading data..."):
    Y_df, station_uids, global_min_date, global_max_date = load_long_data(DATA_FILEPATH)

if len(station_uids) == 0:
    st.error("No series found in data. Check DATA_FILEPATH.")
    st.stop()

with st.spinner("Training Chronos-Bolt once (bolt_small)..."):
    predictor = train_chronos_predictor(Y_df, FIRST_DATE)

st.success("Data loaded and Chronos model trained!")

# ----------------------------------------------------------------------
# Sidebar configuration
# ----------------------------------------------------------------------
st.sidebar.header("Configuration")

selected_station = st.sidebar.selectbox(
    "Select Station (unique_id):",
    station_uids,
    index=station_uids.index("NP205_stage") if "NP205_stage" in station_uids else 0,
)

# Compute valid date range for cutoff:
# Need:
#   - at least HISTORY_WINDOW_SIZE days before cutoff
#   - at least PREDICTION_LENGTH days after cutoff
station_dates = (
    Y_df[Y_df["unique_id"] == selected_station]["ds"]
    .sort_values()
    .unique()
)
if len(station_dates) < HISTORY_WINDOW_SIZE + PREDICTION_LENGTH + 1:
    st.error(f"Not enough data for station {selected_station} to support "
             f"{HISTORY_WINDOW_SIZE}-day history + {PREDICTION_LENGTH}-day forecast.")
    st.stop()

min_idx = HISTORY_WINDOW_SIZE
max_idx = len(station_dates) - PREDICTION_LENGTH - 1

min_cutoff_date = station_dates[min_idx]
max_cutoff_date = station_dates[max_idx]

selected_date = st.sidebar.date_input(
    "Select Forecast Cutoff Date:",
    value=min_cutoff_date,
    min_value=min_cutoff_date,
    max_value=max_cutoff_date,
    help=(
        f"Chronos will see the last {HISTORY_WINDOW_SIZE} days ending on this date "
        f"and forecast the next {PREDICTION_LENGTH} days."
    ),
)
selected_date = pd.to_datetime(selected_date)

# ----------------------------------------------------------------------
# RUN FORECAST
# ----------------------------------------------------------------------
if st.button("Run 7-Day Forecast"):
    # 1. Build the rolling window data up to selected_date (like your loop)
    train_data = Y_df[Y_df["ds"] <= selected_date].copy()
    train_data = (
        train_data
        .groupby("unique_id")
        .tail(HISTORY_WINDOW_SIZE)
    )

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="unique_id",
        timestamp_column="ds",
    )

    with st.spinner("Predicting with Chronos-Bolt..."):
        forecast_ts = predictor.predict(train_ts)  # TimeSeriesDataFrame

    # 2. Extract forecast for the selected station
    # forecast_ts index: (item_id, timestamp), columns: 'mean', quantiles...
    fc_station = forecast_ts.loc[selected_station]
    fc_station = fc_station.reset_index()  # columns: ['item_id', 'timestamp', 'mean', ...]
    fc_station.rename(columns={"timestamp": "ds", "mean": "Chronos"}, inplace=True)

    # 3. History last 107 days and ground truth next 7 days
    station_hist = Y_df[Y_df["unique_id"] == selected_station].copy()
    station_hist = station_hist.sort_values("ds")

    # Get all dates for this station
    station_dates_full = station_hist["ds"].values
    # index of cutoff
    cutoff_idx = np.where(station_dates_full == selected_date)[0][0]

    # History window
    hist_start_idx = cutoff_idx - HISTORY_WINDOW_SIZE + 1
    hist_dates = station_dates_full[hist_start_idx:cutoff_idx + 1]
    hist_vals = station_hist.iloc[hist_start_idx:cutoff_idx + 1]["target"].values

    # Ground truth for forecast horizon
    true_mask = (station_hist["ds"] > selected_date) & (
        station_hist["ds"] <= selected_date + pd.Timedelta(days=PREDICTION_LENGTH)
    )
    truth_future = station_hist[true_mask].copy()
    # Align truth to forecast dates by merging on ds
    truth_future = truth_future[["ds", "target"]]

    # 4. Build DataFrames for plotting
    df_hist = pd.DataFrame(
        {"ds": hist_dates, "Value": hist_vals, "Type": "History"}
    )

    df_fc = pd.DataFrame(
        {"ds": fc_station["ds"], "Value": fc_station["Chronos"], "Type": "Chronos Forecast"}
    )

    df_truth = pd.DataFrame(
        {"ds": truth_future["ds"], "Value": truth_future["target"], "Type": "Actual"}
    )

    plot_df = pd.concat([df_hist, df_fc, df_truth], ignore_index=True)

    # 5. Plot
    st.subheader(f"Station: {selected_station} – Forecast from {selected_date.date()}")

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("ds:T", title="Date"),
            y=alt.Y("Value:Q", title="Stage / Water Level"),
            color=alt.Color("Type:N", title="Series"),
            tooltip=["ds", "Type", "Value"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # 6. Optional: show 7-day MAE/RMSE for this window
    merged = pd.merge(
        df_fc[df_fc["Type"] == "Chronos Forecast"],
        df_truth[df_truth["Type"] == "Actual"] if "Type" in df_truth.columns else df_truth,
        on="ds",
        how="inner",
        suffixes=("_pred", "_true"),
    )

    if len(merged) > 0:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
            # ---- Display Actual vs Predicted Table ----
        debug_table = merged[["ds", "Value_true", "Value_pred"]].copy()
        debug_table.rename(
            columns={
                "ds": "Date",
                "Value_true": "Actual (Observed)",
                "Value_pred": "Predicted (Chronos)"
            },
            inplace=True,
        )

        st.subheader("Actual vs Predicted Values (7-Day Forecast Window)")
        st.dataframe(debug_table.style.format({"Actual (Observed)": "{:.4f}",
                                            "Predicted (Chronos)": "{:.4f}"}))

        mae_val = mean_absolute_error(merged["Value_true"], merged["Value_pred"])
        rmse_val = np.sqrt(mean_squared_error(merged["Value_true"], merged["Value_pred"]))

        st.markdown(
            f"**7-day window metrics (this station, this cutoff):**  \n"
            f"MAE = `{mae_val:.4f}`,  RMSE = `{rmse_val:.4f}`"
        )
    else:
        st.info("No overlapping truth found for the forecast horizon (edge of dataset?).")
