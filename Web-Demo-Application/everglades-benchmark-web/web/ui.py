"""Shared styling and small UI components."""
from __future__ import annotations

import html

import streamlit as st

from everglades_core import config

INK = "#17252A"
WATER = "#1B6E8A"      # predictions, accent
OBSERVED = "#B3BCB8"   # observed values: wide light line under predictions
MARL = "#EEF1ED"       # panels
SAWGRASS = "#8A8F3C"   # test period marker
RATING_COLORS = {
    "very good": ("#E3EFE4", "#24552D"),
    "good": ("#ECF0DC", "#4A6120"),
    "satisfactory": ("#F6ECD6", "#7A540D"),
    "unsatisfactory": ("#F6E1DC", "#8A2F21"),
}
METRIC_HELP = {
    "MAE": "Mean absolute error, in feet. Lower is better.",
    "RMSE": "Root mean squared error, in feet. Lower is better.",
    "NSE": "Nash-Sutcliffe efficiency. 1 is perfect; below 0 is worse than "
           "predicting the observed mean.",
    "PBIAS": "Percent bias. 0 is perfect; positive means the model underestimates.",
    "R2": "Squared correlation between observed and predicted values.",
}
METRIC_LABEL = {"R2": "R²"}

CSS = f"""
<style>
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
h1, h2, h3 {{ font-weight: 500 !important; letter-spacing: -0.01em; color: {INK}; }}
h1 {{ font-size: 1.9rem !important; }}
h2 {{ font-size: 1.35rem !important; }}
h3 {{ font-size: 1.15rem !important; }}
.eb-lede {{ color: #46524F; font-size: 1.02rem; max-width: 62ch; margin: -0.4rem 0 1.4rem; line-height: 1.55; }}
.eb-metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 0.4rem 0 0.8rem; }}
.eb-card {{ background: {MARL}; border-radius: 6px; padding: 12px 14px; }}
.eb-card .l {{ font-size: 0.8rem; color: #46524F; }}
.eb-card .v {{ font-size: 1.55rem; font-weight: 500; color: {INK}; font-variant-numeric: tabular-nums; line-height: 1.3; }}
.eb-chip {{ display: inline-block; font-size: 0.74rem; padding: 1px 8px; border-radius: 4px; margin-top: 2px; }}
.eb-run-title {{ font-size: 1.15rem; font-weight: 500; color: {INK}; margin: 0.2rem 0 0.1rem; }}
.eb-run-sub {{ font-size: 0.86rem; color: #5A6563; margin-bottom: 0.6rem; }}
.eb-note {{ font-size: 0.84rem; color: #5A6563; }}
@media (max-width: 760px) {{ .eb-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
"""


def timing_cards(cells: list[tuple[str, str, str]]):
    """cells: (label, value, note)."""
    html_cells = "".join(
        f'<div class="eb-card"><div class="l">{html.escape(l)}</div>'
        f'<div class="v">{html.escape(v)}</div>'
        f'<div class="l" style="margin-top:2px">{html.escape(n)}</div></div>'
        for l, v, n in cells)
    st.markdown(f'<div class="eb-metrics" style="grid-template-columns:repeat({len(cells)},'
                f'minmax(0,1fr))">{html_cells}</div>', unsafe_allow_html=True)


def timing_summary(timings: dict) -> dict:
    """Training, one forecast, and evaluation as display strings."""
    t = timings or {}
    train = t.get("training_seconds")
    return {
        "Training": "None (zero-shot)" if "training_seconds" in t and train is None
                    else runtime_text(train),
        "One forecast": runtime_text(t.get("forecast_seconds")),
    }


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def lede(text: str):
    st.markdown(f'<p class="eb-lede">{html.escape(text)}</p>', unsafe_allow_html=True)


def chip(rating: str | None) -> str:
    if not rating or rating == "n/a":
        return ""
    bg, fg = RATING_COLORS.get(rating, (MARL, INK))
    return (f'<span class="eb-chip" style="background:{bg};color:{fg}">'
            f'{html.escape(rating.capitalize())}</span>')


def fmt(metric: str, v) -> str:
    if v is None:
        return "n/a"
    if metric == "PBIAS":
        return "0.00" if abs(v) < 0.005 else f"{v:.2f}"
    return f"{v:.3f}"


def metric_cards(overall: dict):
    cells = []
    for m in ["MAE", "RMSE", "NSE", "PBIAS", "R2"]:
        rating = overall.get(f"{m}_rating") if m in ("NSE", "PBIAS", "R2") else None
        cells.append(
            f'<div class="eb-card" title="{html.escape(METRIC_HELP[m])}">'
            f'<div class="l">{METRIC_LABEL.get(m, m)}</div>'
            f'<div class="v">{fmt(m, overall.get(m))}</div>{chip(rating)}</div>')
    st.markdown(f'<div class="eb-metrics">{"".join(cells)}</div>', unsafe_allow_html=True)


def clean_zero(v, places: int = 2):
    """Avoid showing -0.00 for tiny negative values."""
    return 0.0 if v is not None and abs(v) < 0.5 * 10 ** -places else v


def station_label(uid: str) -> str:
    return config.STATION_LABELS.get(uid, uid)


def horizon_label(h: int) -> str:
    return f"{h} days"


def horizon_short(h: int) -> str:
    return str(h)


def stage_text(s: dict) -> str:
    state = s.get("state")
    if state == "queued":
        pos = s.get("queue_position")
        return ("Waiting to start" if not pos else
                f"Waiting to start, {pos} run{'s' if pos != 1 else ''} ahead")
    return s.get("stage") or state or ""


def runtime_text(seconds) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.2f} s" if seconds >= 0.1 else f"{seconds * 1000:.0f} ms"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m} min {s} s" if m else f"{s} s"
