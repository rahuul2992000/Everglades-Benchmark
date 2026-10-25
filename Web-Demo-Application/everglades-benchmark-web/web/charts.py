"""Altair charts. Each plots exactly the rows it is given; nothing is shifted."""
from __future__ import annotations

import altair as alt
import pandas as pd

from ui import INK, OBSERVED, SAWGRASS, WATER

FONT = "Public Sans, system-ui, sans-serif"
MODEL_PALETTE = ["#1B6E8A", "#B5652B", "#6C4F8C", "#3F7F4F", "#A2486B",
                 "#2E8FA3", "#8A8F3C", "#7A5A3A", "#4E6FB0", "#B08A2B",
                 "#5C8F85", "#9A4A3A"]


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height)
            .configure(font=FONT)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#46524F", titleColor="#46524F",
                            gridColor="#E3E7E4", domainColor="#C9CFCB",
                            tickColor="#C9CFCB", labelFontSize=12, titleFontSize=12,
                            titleFontWeight=400)
            .configure_legend(orient="top", labelFontSize=12, title=None,
                              labelColor=INK, symbolStrokeWidth=2.5))


def _time_x():
    return alt.X("ds:T", title=None, axis=alt.Axis(
        format="%b %Y", labelAngle=0, tickCount={"interval": "month", "step": 1}))


def hydrograph(df: pd.DataFrame, height: int = 340) -> alt.Chart:
    """df: ds, y, yhat for one station. Observed is a wide light line drawn
    underneath; the prediction is a thinner line on top."""
    long = pd.concat([
        pd.DataFrame({"ds": df["ds"], "value": df["y"], "series": "Observed"}),
        pd.DataFrame({"ds": df["ds"], "value": df["yhat"], "series": "Predicted"}),
    ])
    lines = alt.Chart(long).mark_line().encode(
        x=_time_x(),
        y=alt.Y("value:Q", title="Water level (ft)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Observed", "Predicted"], range=[OBSERVED, WATER])),
        strokeWidth=alt.condition(alt.datum.series == "Observed",
                                  alt.value(4), alt.value(1.8)),
        order=alt.Order("series:N"))
    hover = alt.selection_point(fields=["ds"], nearest=True, on="pointerover",
                                clear="pointerout", empty=False)
    wide = df.assign(Observed=df["y"], Predicted=df["yhat"])
    rule = (alt.Chart(wide).mark_rule(color="#9AA3A0").encode(
        x="ds:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
                 alt.Tooltip("Observed:Q", format=".3f"),
                 alt.Tooltip("Predicted:Q", format=".3f")])
        .add_params(hover))
    return _style(alt.layer(lines, rule), height).interactive(bind_y=False)


def forecast_chart(history: pd.DataFrame, fc: pd.DataFrame, origin,
                   height: int = 340) -> alt.Chart:
    """history: ds, y (observed, up to origin). fc: ds, yhat (after origin)."""
    long = pd.concat([
        pd.DataFrame({"ds": history["ds"], "value": history["y"], "series": "Observed"}),
        pd.DataFrame({"ds": fc["ds"], "value": fc["yhat"], "series": "Forecast"}),
    ])
    span = (long["ds"].max() - long["ds"].min()).days
    x = (_time_x() if span > 90 else alt.X("ds:T", title=None, axis=alt.Axis(
        format="%b %d", labelAngle=0, tickCount={"interval": "week", "step": 1})))
    lines = alt.Chart(long).mark_line().encode(
        x=x,
        y=alt.Y("value:Q", title="Water level (ft)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Observed", "Forecast"], range=[OBSERVED, WATER])),
        strokeWidth=alt.condition(alt.datum.series == "Observed",
                                  alt.value(4), alt.value(2.2)),
        tooltip=[alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="Water level", format=".3f")])
    dots = alt.Chart(fc).mark_circle(size=28, color=WATER).encode(x="ds:T", y="yhat:Q")
    o = pd.DataFrame({"ds": [pd.Timestamp(origin)], "label": ["Forecast start"]})
    rule = alt.Chart(o).mark_rule(strokeDash=[4, 3], color="#8A938F").encode(x="ds:T")
    label = alt.Chart(o).mark_text(align="right", dx=-6, dy=10, fontSize=12,
                                   color="#46524F").encode(x="ds:T", y=alt.value(0), text="label")
    return _style(alt.layer(lines, dots, rule, label), height).interactive(bind_y=False)


def overlay(frames: dict[str, pd.DataFrame], height: int = 360) -> alt.Chart:
    """frames: model -> df with ds, y, yhat for one station."""
    first = next(iter(frames.values()))
    parts = [pd.DataFrame({"ds": first["ds"], "value": first["y"], "series": "Observed"})]
    for model, df in frames.items():
        parts.append(pd.DataFrame({"ds": df["ds"], "value": df["yhat"], "series": model}))
    long = pd.concat(parts)
    long["rank"] = (long["series"] != "Observed").astype(int)
    domain = ["Observed", *frames.keys()]
    rng = [OBSERVED, *MODEL_PALETTE[:len(frames)]]
    chart = alt.Chart(long).mark_line().encode(
        x=_time_x(),
        y=alt.Y("value:Q", title="Water level (ft)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=rng)),
        strokeWidth=alt.condition(alt.datum.series == "Observed",
                                  alt.value(4), alt.value(1.5)),
        order=alt.Order("rank:Q"),
        tooltip=[alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="Water level", format=".3f")])
    return _style(chart, height).interactive(bind_y=False)


def by_horizon(df: pd.DataFrame, metric: str, label: str, height: int = 320) -> alt.Chart:
    """df: model, horizon, <metric>."""
    models = list(dict.fromkeys(df["model"]))
    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=55, filled=True),
                                    strokeWidth=2).encode(
        x=alt.X("horizon:O", title="Forecast horizon (days)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=label, scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=alt.Scale(
            domain=models, range=MODEL_PALETTE[:len(models)])),
        tooltip=["model", "horizon", alt.Tooltip(f"{metric}:Q", format=".3f")])
    return _style(chart, height)


def data_explorer(long: pd.DataFrame, periods: list[dict], height: int = 380) -> alt.Chart:
    """long: ds, value, series. periods: [{"name", "start", "end"}]."""
    p = pd.DataFrame(periods)
    shade = alt.Chart(p).mark_rect(opacity=0.5).encode(
        x="start:T", x2="end:T",
        color=alt.Color("name:N", title=None, scale=alt.Scale(
            domain=["Training", "Validation", "Test"],
            range=["#FFFFFF", "#EEF1ED", "#E4E8D2"]),
            legend=alt.Legend(orient="top", symbolType="square", symbolStrokeColor="#B8BFBB")))
    labels = alt.Chart(p).mark_text(align="left", dx=6, dy=10, fontSize=12,
                                    color="#46524F").encode(
        x="start:T", y=alt.value(0), text="name:N")
    series = list(dict.fromkeys(long["series"]))
    lines = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
        x=_time_x(),
        y=alt.Y("value:Q", title="Value", scale=alt.Scale(zero=False)),
        strokeDash=alt.value([1, 0]),
        detail="series:N",
        tooltip=[alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"),
                 "series:N", alt.Tooltip("value:Q", format=".3f")])
    lines = lines.encode(color=alt.value(WATER)) if len(series) == 1 else lines.encode(
        stroke=alt.Stroke("series:N", title=None, scale=alt.Scale(
            domain=series, range=MODEL_PALETTE[:len(series)])))
    return _style(alt.layer(shade, lines, labels).resolve_scale(color="independent"),
                  height).interactive(bind_y=False)


TEST_COLOR = SAWGRASS
