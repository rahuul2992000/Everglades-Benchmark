"""Everglades forecasting benchmark website.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))

import streamlit as st  # noqa: E402

import ui  # noqa: E402

st.set_page_config(page_title="Everglades forecasting benchmark", page_icon="💧",
                   layout="wide")
ui.inject_css()

pages = [
    st.Page(HERE / "views" / "run.py", title="Evaluate", default=True),
    st.Page(HERE / "views" / "compare.py", title="Compare", url_path="compare"),
    st.Page(HERE / "views" / "forecast.py", title="Forecast", url_path="forecast"),
    st.Page(HERE / "views" / "data_view.py", title="Data", url_path="data"),
    st.Page(HERE / "views" / "about.py", title="About", url_path="about"),
]
st.navigation(pages, position="top").run()
