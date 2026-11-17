"""Plotly figure builders. They take result tables and return figures; no Streamlit here."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    AMOUNT_MISMATCH,
    MATCHED,
    MISSING_FROM_BANK,
    MISSING_FROM_PLATFORM,
    NOT_SETTLEABLE,
    PENDING_SETTLEMENT,
    ROUNDING_DIFFERENCE,
    STATUS_CONFLICT,
)

INK = "#17233B"
TEAL = "#0F766E"
AMBER = "#B7791F"
BRICK = "#B23A48"
SLATE = "#64748B"
SKY = "#3B82A0"

STATUS_COLORS = {
    MATCHED: TEAL,
    ROUNDING_DIFFERENCE: AMBER,
    PENDING_SETTLEMENT: SKY,
    AMOUNT_MISMATCH: BRICK,
    MISSING_FROM_BANK: "#7F1D2D",
    MISSING_FROM_PLATFORM: "#9A5B13",
    STATUS_CONFLICT: "#6B4E9B",
    NOT_SETTLEABLE: "#A8B3C2",
}
SEVERITY_COLORS = {"High": BRICK, "Medium": AMBER, "Low": SKY}

_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
)


def _empty(title: str, message: str = "No data to show") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=SLATE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def status_breakdown_chart(matches: pd.DataFrame) -> go.Figure:
    """Matched vs unmatched: how many transactions fall into each match status."""
    title = "Matched vs unmatched transactions"
    if matches.empty:
        return _empty(title)
    counts = matches["match_status"].value_counts()
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values), hole=0.55,
        marker=dict(colors=[STATUS_COLORS.get(s, SLATE) for s in counts.index]),
        textinfo="value", sort=False,
    ))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def discrepancy_distribution_chart(discrepancies: pd.DataFrame) -> go.Figure:
    title = "Discrepancies by type"
    if discrepancies.empty:
        return _empty(title, "No discrepancies found")
    grouped = discrepancies.groupby(["category", "severity"]).size().reset_index(name="count")
    fig = go.Figure()
    for severity in ("High", "Medium", "Low"):
        part = grouped[grouped["severity"] == severity]
        if part.empty:
            continue
        fig.add_bar(y=part["category"], x=part["count"], name=f"{severity} severity", orientation="h",
                    marker_color=SEVERITY_COLORS[severity])
    fig.update_layout(title=title, barmode="stack", yaxis=dict(autorange="reversed"), **_LAYOUT)
    fig.update_xaxes(title_text="Records", dtick=1)
    return fig


def platform_vs_bank_chart(bridge: pd.DataFrame) -> go.Figure:
    title = "Platform vs bank settlement totals"
    if bridge.empty:
        return _empty(title)
    fig = go.Figure()
    fig.add_bar(x=bridge["currency"], y=[float(v) for v in bridge["platform_total"]], name="Platform", marker_color=INK)
    fig.add_bar(x=bridge["currency"], y=[float(v) for v in bridge["bank_total"]], name="Bank", marker_color=TEAL)
    fig.update_layout(title=title, barmode="group", **_LAYOUT)
    fig.update_yaxes(title_text="Net amount")
    return fig


def daily_trend_chart(daily: pd.DataFrame, currency: str) -> go.Figure:
    title = f"Daily platform records vs bank settlements ({currency})"
    data = daily[daily["currency"] == currency].copy()
    if data.empty:
        return _empty(title)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=data["date"], y=data["platform_amount"], name="Platform (by transaction date)", mode="lines+markers", line=dict(color=INK))
    fig.add_scatter(x=data["date"], y=data["bank_amount"], name="Bank (by settlement date)", mode="lines+markers", line=dict(color=TEAL, dash="dot"))
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Net amount")
    return fig


def rounding_chart(details: pd.DataFrame) -> go.Figure:
    """Each rounding gap as a small bar, with the running total as a line.

    Shows why tiny individual differences matter: the line keeps climbing.
    """
    title = "Rounding differences: individual vs cumulative"
    if details.empty:
        return _empty(title, "No rounding differences found")
    data = details.sort_values(["bank_date", "transaction_id"]).reset_index(drop=True)
    diffs = [float(v) for v in data["amount_difference"]]
    running, total = [], 0.0
    for d in diffs:
        total += d
        running.append(total)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=data["transaction_id"], y=diffs, name="Individual difference", marker_color=AMBER, secondary_y=False)
    fig.add_scatter(x=data["transaction_id"], y=running, name="Cumulative difference", mode="lines+markers", line=dict(color=BRICK), secondary_y=True)
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Individual", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative", secondary_y=True)
    fig.update_xaxes(showticklabels=False, title_text="Transactions in settlement order")
    return fig


def bridge_chart(bridge_row: dict) -> go.Figure:
    """Waterfall of the pieces that add up to the platform-minus-bank gap for one currency.

    A positive bar means the platform shows more money than the bank for that reason.
    The last bar is the total gap, and 'Unexplained' should be zero.
    """
    steps = [
        ("Amount differences", "amount_differences"), ("Currency mismatch", "currency_mismatch"),
        ("Pending settlement", "pending_settlement"), ("Missing from bank", "missing_from_bank"),
        ("Missing from platform", "missing_from_platform"), ("Status conflict", "status_conflict"),
        ("Platform duplicates", "platform_duplicates"), ("Bank duplicates", "bank_duplicates"),
        ("Unexplained", "unexplained"),
    ]
    used = [(name, float(bridge_row[key])) for name, key in steps if float(bridge_row[key]) != 0.0]
    if not used:
        return _empty(f"Gap between platform and bank ({bridge_row['currency']})", "Platform and bank totals agree exactly")
    labels = [name for name, _ in used] + ["Total gap"]
    values = [value for _, value in used] + [0.0]
    measures = ["relative"] * len(used) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing=dict(marker=dict(color=INK)), decreasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=AMBER)), connector=dict(line=dict(color=SLATE, width=1)),
    ))
    layout = {k: v for k, v in _LAYOUT.items() if k != "legend"}
    fig.update_layout(title=f"What explains the platform vs bank gap ({bridge_row['currency']})", showlegend=False, **layout)
    fig.update_yaxes(title_text="Platform minus bank")
    return fig