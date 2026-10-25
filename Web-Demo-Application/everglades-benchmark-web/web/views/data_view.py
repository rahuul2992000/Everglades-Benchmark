import pandas as pd
import streamlit as st

import charts
import ui
from everglades_core import config, data

st.title("Data")
ui.lede("Daily observations from October 2020 to August 2024: water levels, "
        "structure flows, rainfall, and evapotranspiration across 37 series. "
        "Models are scored on water level at five stations.")


@st.cache_data(show_spinner=False)
def load():
    long = data.load_long()
    return data.load_wide(), long, data.split(long)


wide, long, s = load()
periods = [
    {"name": "Training", "start": wide["date"].min(), "end": s.train_end},
    {"name": "Validation", "start": s.val_start, "end": s.test_start - pd.Timedelta(days=1)},
    {"name": "Test", "start": s.test_start, "end": s.test_end},
]

c = st.columns(4)
c[0].metric("Series", long["unique_id"].nunique())
c[1].metric("Days", wide["date"].nunique())
c[2].metric("Training and validation", f"{s.Y_fit_df['ds'].nunique()} days")
c[3].metric("Test", f"{s.test_size} days")

targets = [c for c in config.STATIONS]
others = sorted(c for c in wide.columns if c not in targets + ["date"])
choice = st.multiselect("Series", targets + others, default=targets,
                        format_func=ui.station_label)
if choice:
    plot = long[long["unique_id"].isin(choice)].rename(columns={"y": "value"})
    plot = plot.assign(series=plot["unique_id"].map(ui.station_label))
    st.altair_chart(charts.data_explorer(plot, periods), width="stretch")
    st.markdown('<p class="eb-note">Series have different units, so plot similar '
                'series together. Drag to pan, scroll to zoom.</p>', unsafe_allow_html=True)

st.markdown("**Periods**")
st.dataframe(pd.DataFrame([{"Period": p["name"], "From": p["start"].date(),
                            "To": p["end"].date(),
                            "Days": (p["end"] - p["start"]).days + 1} for p in periods]),
             hide_index=True, width="stretch")
