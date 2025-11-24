# app_neuralforecast_everglades.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from neuralforecast import NeuralForecast
from neuralforecast.models import (
    NBEATS, Informer, PatchTST, TimeMixer, TSMixer, TSMixerx,
    iTransformer, RMoK, NLinear, DLinear, KAN
)
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

# ======================================================================
# CONFIG
# ======================================================================

DATA_FILEPATH = "D:/Downloads/Everglades-Code/data/final_concatenated_data.csv"

HORIZON = 7
INPUT_SIZE = 100

STAGES_OF_INTEREST = ['NP205_stage', 'P33_stage', 'G620_water_level', 'NESRS1', 'NESRS2']

# ======================================================================
# DATA LOADING & PREPROCESSING
# ======================================================================

@st.cache_data
def load_long_data(csv_path: str):
    """Load wide CSV and convert to long format Y_df with train/val/test splits."""
    data = pd.read_csv(csv_path)

    # Drop unwanted index column if present
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    data['date'] = pd.to_datetime(data['date'])
    df = data.copy()

    # Melt to long format: unique_id, ds, y
    long_df = pd.melt(
        df,
        id_vars=['date'],
        value_vars=[col for col in df.columns if col != 'date'],
        var_name='unique_id',
        value_name='y'
    )
    long_df.rename(columns={'date': 'ds'}, inplace=True)
    long_df = long_df[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds'])
    Y_df = long_df.reset_index(drop=True)

    # Train/val/test splitting (last 422 days = val+test, last 211 = test)
    ds_sorted = np.sort(Y_df['ds'].unique())
    if len(ds_sorted) < 422:
        raise ValueError("Not enough time steps for 422-day (val+test) split (need at least 422 days).")

    train_val_cutoff = ds_sorted[-422]   # first day of val
    test_cutoff      = ds_sorted[-211]   # first day of test

    Y_train_df = Y_df[Y_df['ds'] < train_val_cutoff]
    Y_val_df   = Y_df[(Y_df['ds'] >= train_val_cutoff) & (Y_df['ds'] < test_cutoff)]
    Y_test_df  = Y_df[Y_df['ds'] >= test_cutoff].reset_index(drop=True)

    train_size = len(np.sort(Y_train_df['ds'].unique()))
    val_size   = len(np.sort(Y_val_df['ds'].unique()))
    test_size  = len(np.sort(Y_test_df['ds'].unique()))

    # Fit uses train+val with val_size parameter
    Y_fit_df = pd.concat([Y_train_df, Y_val_df]).reset_index(drop=True)

    n_series = Y_df['unique_id'].nunique()
    station_uids = sorted(Y_df['unique_id'].unique())

    split_info = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'train_val_cutoff': train_val_cutoff,
        'test_cutoff': test_cutoff,
        'n_series': n_series
    }

    return Y_df, Y_fit_df, Y_test_df, station_uids, split_info


def build_nf_model(model_name: str, h: int, input_size: int, n_series: int):
    """Factory to create a NeuralForecast model instance by name."""
    if model_name == 'NBEATS':
        return NBEATS(h=h, input_size=input_size,
                      max_steps=1000, val_check_steps=100,
                      early_stop_patience_steps=50)
    elif model_name == 'Informer':
        return Informer(h=h, input_size=input_size,
                        max_steps=1000, val_check_steps=100,
                        early_stop_patience_steps=50)
    elif model_name == 'PatchTST':
        return PatchTST(h=h, input_size=input_size,
                        max_steps=1000, val_check_steps=100,
                        early_stop_patience_steps=50)
    elif model_name == 'TimeMixer':
        return TimeMixer(h=h, input_size=input_size, n_series=n_series,
                         max_steps=1000, val_check_steps=100,
                         early_stop_patience_steps=50)
    elif model_name == 'TSMixer':
        return TSMixer(h=h, input_size=input_size, n_series=n_series,
                       max_steps=1000, val_check_steps=100,
                       early_stop_patience_steps=50)
    elif model_name == 'TSMixerx':
        return TSMixerx(h=h, input_size=input_size, n_series=n_series,
                        max_steps=1000, val_check_steps=100,
                        early_stop_patience_steps=50)
    elif model_name == 'iTransformer':
        return iTransformer(h=h, input_size=input_size, n_series=n_series,
                            max_steps=1000, val_check_steps=100,
                            early_stop_patience_steps=50)
    elif model_name == 'RMoK':
        return RMoK(h=h, input_size=input_size, n_series=n_series,
                    max_steps=1000, val_check_steps=100,
                    early_stop_patience_steps=50)
    elif model_name == 'NLinear':
        return NLinear(h=h, input_size=input_size,
                       max_steps=1000, val_check_steps=100,
                       early_stop_patience_steps=50)
    elif model_name == 'DLinear':
        return DLinear(h=h, input_size=input_size,
                       max_steps=1000, val_check_steps=100,
                       early_stop_patience_steps=50)
    elif model_name == 'KAN':
        return KAN(h=h, input_size=input_size,
                   max_steps=1000, val_check_steps=100,
                   early_stop_patience_steps=50)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@st.cache_resource
def train_neuralforecast(model_name: str, Y_fit_df: pd.DataFrame, val_size: int, n_series: int):
    """Train a single NeuralForecast model and return the fitted NF object."""
    model = build_nf_model(model_name, HORIZON, INPUT_SIZE, n_series)
    nf = NeuralForecast(models=[model], freq='D')  # daily data
    nf.fit(df=Y_fit_df, val_size=val_size)
    return nf


@st.cache_data
def get_cv_predictions(model_name: str, Y_df: pd.DataFrame,
                       Y_fit_df: pd.DataFrame,
                       val_size: int, test_size: int, n_series: int):
    """Run cross_validation for the selected model and return aggregated predictions."""
    nf = train_neuralforecast(model_name, Y_fit_df=Y_fit_df, val_size=val_size, n_series=n_series)
    Y_hat_df = nf.cross_validation(
        df=Y_df,
        val_size=val_size,
        test_size=test_size,
        n_windows=None
    )
    Y_hat_df = Y_hat_df.reset_index()

    # Column name for predictions is typically the model name
    model_col = model_name

    # Aggregate (in case there are multiple samples)
    Y_hat_df_agg = (
        Y_hat_df
        .groupby(['unique_id', 'ds'], as_index=False)
        .agg({model_col: 'mean', 'y': 'mean'})
    )

    return Y_hat_df_agg


@st.cache_data
def get_future_forecast(model_name: str,
                        Y_df: pd.DataFrame,
                        Y_fit_df: pd.DataFrame,
                        val_size: int,
                        n_series: int,
                        horizon: int,
                        test_size: int):
    """Get future forecast beyond the last date for the selected model."""
    nf = train_neuralforecast(model_name, Y_fit_df=Y_fit_df, val_size=val_size, n_series=n_series)

    # Out-of-sample forecast: next 'horizon' points beyond the last ds
    # Depending on NF version, this may be h=... or test_size=...
    future_fc = nf.predict(df=Y_df, test_size=test_size)  # if this errors, try test_size=horizon
    future_fc = future_fc.reset_index()

    return future_fc

# ======================================================================
# CHRONOS-BOLT (AutoGluon) HELPERS – ALIGNED WITH NF TEST PERIOD
# ======================================================================

HISTORY_WINDOW_SIZE = 107      # from your notebook
CHRONOS_PRED_LENGTH = 7        # prediction_length


@st.cache_resource
def load_chronos_predictor(
    Y_df: pd.DataFrame,
    first_date: pd.Timestamp,
) -> TimeSeriesPredictor:
    """
    One-time Chronos-Bolt fit using your initial setup logic,
    aligned so that `first_date` = NF test_start.
    """
    # Your notebook: Y_df with columns ['unique_id', 'ds', 'y'] -> rename to 'target'
    Y_df_chronos = Y_df.rename(columns={'y': 'target'}).copy()

    # Use NF test_start as the cutoff for initial training
    initial_train_data = Y_df_chronos[Y_df_chronos['ds'] <= first_date]
    initial_train_data = initial_train_data.groupby('unique_id').tail(HISTORY_WINDOW_SIZE)

    initial_train_ts = TimeSeriesDataFrame.from_data_frame(
        initial_train_data,
        id_column="unique_id",
        timestamp_column="ds",
    )

    predictor = TimeSeriesPredictor(
        prediction_length=CHRONOS_PRED_LENGTH
    ).fit(
        initial_train_ts,
        presets="bolt_small",
        # hyperparameters={"enable_gpu": True},  # if you use GPU
    )

    return predictor


@st.cache_data
def chronos_full_predictions(
    Y_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp
) -> pd.DataFrame:
    """
    Run your rolling-window Chronos logic over the NF test period:
    ds in [test_start, test_end].
    Returns results_df with ['unique_id', 'ds', 'target', 'Chronos'].
    """
    predictor = load_chronos_predictor(Y_df, first_date=test_start)

    Y_df_chronos = Y_df.rename(columns={'y': 'target'}).copy()

    # Align to NF test window
    date_range = pd.date_range(test_start, test_end, freq='D')
    predictions_list = []

    for current_date in date_range:
        # Same as your notebook:
        train_data = Y_df_chronos[Y_df_chronos['ds'] <= current_date]
        train_data = train_data.groupby('unique_id').tail(HISTORY_WINDOW_SIZE)

        train_data_ts = TimeSeriesDataFrame.from_data_frame(
            train_data,
            id_column="unique_id",
            timestamp_column="ds",
        )

        forecast = predictor.predict(train_data_ts)

        # Extract 7th-day mean forecast per unique_id
        seventh_day_forecasts = (
            forecast['mean']
            .groupby(level='item_id')
            .last()              # last of the 7 days -> your “7th-day” logic
            .reset_index()
        )
        seventh_day_forecasts['ds'] = current_date
        seventh_day_forecasts.rename(
            columns={'item_id': 'unique_id', 'mean': 'Chronos'},
            inplace=True
        )

        predictions_list.append(seventh_day_forecasts)

    df_predictions = pd.concat(predictions_list, ignore_index=True)

    # Merge with ground truth like in your notebook
    results_df = pd.merge(
        df_predictions,
        Y_df_chronos,
        on=['unique_id', 'ds'],
        how='inner'
    )

    return results_df


# ======================================================================
# STREAMLIT UI
# ======================================================================

st.set_page_config(layout="wide")
st.title("Everglades Water Level Forecasting with Deep Learning Models")

with st.spinner("Loading data and preparing splits..."):
    Y_df, Y_fit_df, Y_test_df, station_uids, split_info = load_long_data(DATA_FILEPATH)

st.success(
    f"Loaded {len(station_uids)} series. "
    f"Train={split_info['train_size']} days, "
    f"Val={split_info['val_size']} days, "
    f"Test={split_info['test_size']} days."
)

# Sidebar configuration
st.sidebar.header("Configuration")

model_options = [
    'NBEATS', 'Informer', 'PatchTST', 'TimeMixer', 'TSMixer',
    'TSMixerx', 'iTransformer', 'RMoK', 'NLinear', 'DLinear', 'KAN'
]

selected_model = st.sidebar.selectbox("Select Model:", model_options, index=0)

selected_station = st.sidebar.selectbox(
    "Select Station (unique_id):",
    station_uids,
    index=station_uids.index('NP205_stage') if 'NP205_stage' in station_uids else 0
)

show_metrics_for_all = st.sidebar.checkbox(
    "Show MAE/RMSE summary for 5 key stations",
    value=True
)

# Tabs: Evaluation vs Future Forecast
tab_eval, tab_future = st.tabs(["📊 Evaluation (Historical)", "🔮 Future Forecast (Next 7 Days)"])

# ----------------------------------------------------------------------
# TAB 1: EVALUATION
# ----------------------------------------------------------------------
with tab_eval:
    st.subheader(f"Evaluation on Historical Test Period – {selected_model}")

    if st.button("Run Evaluation", key="eval_button"):
        with st.spinner(f"Training {selected_model} and running cross-validation..."):
            Y_hat_df_agg = get_cv_predictions(
                model_name=selected_model,
                Y_df=Y_df,
                Y_fit_df=Y_fit_df,
                val_size=split_info['val_size'],
                test_size=split_info['test_size'],
                n_series=split_info['n_series']
            )

        st.success("Evaluation forecasts generated!")

        model_col = selected_model
        test_cutoff = split_info['test_cutoff']

        # Plot: Forecast vs Actual on test window for selected station
        st.markdown(f"### Forecast vs Actual (Test Period) – Station: `{selected_station}`")

        station_data = Y_hat_df_agg[
            (Y_hat_df_agg['unique_id'] == selected_station) &
            (Y_hat_df_agg['ds'] >= test_cutoff)
        ].copy()

        if station_data.empty:
            st.warning("No data available for this station in the test period.")
        else:
            station_long = pd.concat([
                station_data.assign(Type="Actual",  Value=lambda df: df['y']),
                station_data.assign(Type="Forecast", Value=lambda df: df[model_col])
            ], ignore_index=True)

            chart = (
                alt.Chart(station_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X('ds:T', title='Date'),
                    y=alt.Y('Value:Q', title='Stage / Water Level'),
                    color=alt.Color('Type:N', title='Series'),
                    tooltip=['ds', 'Type', 'Value']
                )
                .properties(height=400)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # Metrics summary
        if show_metrics_for_all:
            st.markdown(f"### MAE / RMSE for 5 Key Stations – Model: `{selected_model}`")

            metrics_rows = []

            for stage in STAGES_OF_INTEREST:
                stage_data = Y_hat_df_agg[Y_hat_df_agg['unique_id'] == stage]
                if stage_data.empty:
                    continue

                true = stage_data['y'].values
                pred = stage_data[model_col].values

                mae_val = mae(true, pred)
                rmse_val = np.sqrt(mse(true, pred))

                metrics_rows.append({
                    'Station': stage,
                    'MAE': mae_val,
                    'RMSE': rmse_val
                })

            if metrics_rows:
                metrics_df = pd.DataFrame(metrics_rows)
                overall_mae = metrics_df['MAE'].mean()
                overall_rmse = metrics_df['RMSE'].mean()

                st.dataframe(metrics_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}'}))

                st.markdown(
                    f"**Overall (mean over {len(metrics_rows)} stations)**  \n"
                    f"MAE = `{overall_mae:.4f}`,  RMSE = `{overall_rmse:.4f}`"
                )
            else:
                st.info("No data found for the specified stages in predictions.")

        # =========================
        # Chronos-Bolt comparison
        # =========================
        st.markdown("### Chronos-Bolt (AutoGluon) – Aligned with NF Test Period")

        if st.button("Run Chronos-Bolt on Test Period", key="chronos_button"):
            with st.spinner("Running Chronos-Bolt zero-shot evaluation over NF test window..."):
                chronos_results = chronos_full_predictions(
                    Y_df=Y_df,
                    test_start=split_info['test_cutoff'],
                    test_end=Y_df['ds'].max()
                )

            # --- Plot Chronos vs Actual for selected_station ---
            station_chronos = chronos_results[
                chronos_results['unique_id'] == selected_station
            ].copy()

            if station_chronos.empty:
                st.warning(
                    f"No Chronos results found for station `{selected_station}` in the test window."
                )
            else:
                chronos_long = pd.concat([
                    station_chronos.assign(Type="Actual",  Value=lambda df: df['target']),
                    station_chronos.assign(Type="Chronos", Value=lambda df: df['Chronos']),
                ], ignore_index=True)

                chronos_chart = (
                    alt.Chart(chronos_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X('ds:T', title='Date'),
                        y=alt.Y('Value:Q', title='Stage / Water Level'),
                        color=alt.Color('Type:N', title='Series'),
                        tooltip=['ds', 'Type', 'Value']
                    )
                    .properties(height=400)
                    .interactive()
                )
                st.altair_chart(chronos_chart, use_container_width=True)

            # --- Chronos metrics for 5 key stages ---
            st.markdown("#### MAE / RMSE for 5 Key Stations – Chronos-Bolt")

            chronos_metrics_rows = []
            for stage in STAGES_OF_INTEREST:
                stage_data = chronos_results[chronos_results['unique_id'] == stage]
                if stage_data.empty:
                    continue

                true_c = stage_data['target'].values
                pred_c = stage_data['Chronos'].values

                mae_val_c = mae(true_c, pred_c)
                rmse_val_c = np.sqrt(mse(true_c, pred_c))

                chronos_metrics_rows.append({
                    'Station': stage,
                    'MAE': mae_val_c,
                    'RMSE': rmse_val_c
                })

            if chronos_metrics_rows:
                chronos_metrics_df = pd.DataFrame(chronos_metrics_rows)
                overall_mae_c = chronos_metrics_df['MAE'].mean()
                overall_rmse_c = chronos_metrics_df['RMSE'].mean()

                st.dataframe(chronos_metrics_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}'}))

                st.markdown(
                    f"**Chronos Overall (mean over {len(chronos_metrics_rows)} stations)**  \n"
                    f"MAE = `{overall_mae_c:.4f}`,  RMSE = `{overall_rmse_c:.4f}`"
                )
            else:
                st.info("No Chronos data found for the specified stages in the test window.")


# ----------------------------------------------------------------------
# TAB 2: FUTURE FORECAST
# ----------------------------------------------------------------------
with tab_future:
    st.subheader(f"Future Forecast (Next {HORIZON} Days) – {selected_model}")

    st.markdown(
        "This uses the **full history up to the last available date** in `Y_df` and "
        f"forecasts the next **{HORIZON} days** beyond it."
    )

    if st.button("Forecast Next 7 Days", key="future_button"):
        with st.spinner(f"Training {selected_model} and generating future forecast..."):
            future_fc = get_future_forecast(
                model_name=selected_model,
                Y_df=Y_df,
                Y_fit_df=Y_fit_df,
                val_size=split_info['val_size'],
                n_series=split_info['n_series'],
                horizon=HORIZON,
                test_size=split_info['test_size']
            )

        model_col = selected_model
        last_date = Y_df['ds'].max()

        station_future = future_fc[future_fc['unique_id'] == selected_station].copy()
        # Keep only dates strictly beyond last observed date, if any
        station_future = station_future[station_future['ds'] > last_date]

        if station_future.empty:
            st.warning(
                "No future predictions found for this station beyond the last date. "
                "If your NeuralForecast version uses `test_size` instead of `h`, "
                "replace `h=horizon` with `test_size=horizon` in `get_future_forecast`."
            )
        else:
            st.markdown(f"### Station: `{selected_station}` – Future Forecast")

            chart_future = (
                alt.Chart(station_future)
                .mark_line(point=True)
                .encode(
                    x=alt.X("ds:T", title="Future Date"),
                    y=alt.Y(f"{model_col}:Q", title="Forecasted Stage / Water Level"),
                    tooltip=["ds", model_col]
                )
                .properties(height=350)
            )
            st.altair_chart(chart_future, use_container_width=True)

            st.caption(
                f"Forecast horizon: {HORIZON} days after {last_date.date()}. "
                "No ground truth is available yet for these dates."
            )
