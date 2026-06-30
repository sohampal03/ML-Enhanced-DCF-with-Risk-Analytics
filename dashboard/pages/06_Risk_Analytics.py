"""
Page 6: Risk Analytics Center — Monte Carlo & Sensitivity Analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from dashboard.components.cards import metric_card, probability_bar, section_divider, info_box
from dashboard.components.charts import (
    monte_carlo_histogram,
    tornado_chart,
    violin_plot,
    COLORS,
)
from src.utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Risk Analytics | AlphaForge", layout="wide", page_icon="🎲")

CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first from the home page.")
    st.stop()

report = st.session_state.report
mc = report.monte_carlo
dcf = report.dcf_result

st.markdown(
    """
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">
        🎲 Risk Analytics Center
    </h1>
    <div style="color:#8b9cc8;font-size:14px;margin-bottom:28px;">
        Monte Carlo simulation with 10,000+ paths quantifying valuation uncertainty
    </div>
    """,
    unsafe_allow_html=True,
)

if not mc:
    info_box(
        "Monte Carlo simulation was not run or failed. Re-run analysis with Monte Carlo enabled.",
        "warning",
    )
    if dcf:
        st.markdown("### Base DCF Scenario Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(
                "Bear Case (−30%)", format_currency(dcf.intrinsic_value_per_share * 0.70), icon="🐻"
            )
        with col2:
            metric_card("Base Case", format_currency(dcf.intrinsic_value_per_share), icon="📊")
        with col3:
            metric_card(
                "Bull Case (+30%)", format_currency(dcf.intrinsic_value_per_share * 1.30), icon="🐂"
            )
    st.stop()

current_price = mc.current_price

# ── KPI Cards ────────────────────────────────────────────────────────────────
section_divider("Simulation Statistics", badge=f"{mc.n_simulations:,} Paths")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Mean Value", format_currency(mc.mean_value), icon="📊")
with c2:
    metric_card("Median Value", format_currency(mc.median_value), icon="📍")
with c3:
    metric_card("Std Deviation", format_currency(mc.std_value), icon="📉")
with c4:
    metric_card("Skewness", f"{mc.skewness:.2f}", icon="📐", subtitle="Positive = right tail")
with c5:
    metric_card("Kurtosis", f"{mc.kurtosis:.2f}", icon="📏", subtitle="Fat tails > 3")

st.markdown("<br>", unsafe_allow_html=True)

# ── Probability bars ─────────────────────────────────────────────────────────
col_prob, col_hist = st.columns([1, 2])

with col_prob:
    section_divider("Probability Assessment")
    probability_bar("P(Undervalued) — IV > Current", mc.prob_undervalued, COLORS["bull"])
    probability_bar("P(Fairly Valued) — Within ±15%", mc.prob_in_range, COLORS["gold"])
    probability_bar("P(Overvalued) — IV < Current", mc.prob_overvalued, COLORS["bear"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Percentile table
    section_divider("Percentile Bands")
    pct_df = mc.to_percentile_table()
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

with col_hist:
    section_divider("Intrinsic Value Distribution")
    fig_hist = monte_carlo_histogram(
        intrinsic_values=mc.intrinsic_values,
        current_price=current_price,
        ticker=report.ticker,
        p25=mc.p25,
        p50=mc.p50,
        p75=mc.p75,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Violin + Tornado Row ─────────────────────────────────────────────────────
st.markdown("---")
col_violin, col_tornado = st.columns([1, 1])

with col_violin:
    section_divider("Distribution Shape (Violin)")
    fig_violin = violin_plot(mc.intrinsic_values, current_price, report.ticker)
    st.plotly_chart(fig_violin, use_container_width=True)

with col_tornado:
    section_divider("Parameter Sensitivity (Tornado)")
    if mc.tornado_data:
        fig_tornado = tornado_chart(mc.tornado_data, report.ticker)
        st.plotly_chart(fig_tornado, use_container_width=True)
    else:
        info_box("Tornado data not available", "warning")

# ── Scenario Analysis ────────────────────────────────────────────────────────
st.markdown("---")
section_divider("Scenario Analysis (Bear / Base / Bull)")

sc1, sc2, sc3 = st.columns(3)
scenarios = [
    ("🐻 Bear Case (5th Pct)", mc.p5, COLORS["bear"]),
    ("📊 Base Case (50th Pct)", mc.p50, COLORS["primary"]),
    ("🐂 Bull Case (95th Pct)", mc.p95, COLORS["bull"]),
]
for col, (label, val, color) in zip([sc1, sc2, sc3], scenarios):
    with col:
        upside = (val - current_price) / current_price * 100 if current_price > 0 else 0
        st.markdown(
            f"""
            <div style="
                background:#111827;border:1px solid #1e2d4a;border-top:3px solid {color};
                border-radius:12px;padding:24px;text-align:center;
            ">
                <div style="color:#8b9cc8;font-size:12px;margin-bottom:8px;">{label}</div>
                <div style="color:{color};font-size:28px;font-weight:800;">{format_currency(val)}</div>
                <div style="color:{'#00c851' if upside > 0 else '#ff4444'};
                     font-size:14px;margin-top:8px;">
                     {"+" if upside > 0 else ""}{upside:.1f}% vs current
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Fan Chart ────────────────────────────────────────────────────────────────
st.markdown("---")
section_divider("Fan Chart — Forecast Uncertainty Over Time")

if report.revenue_forecast:
    rf = report.revenue_forecast
    years = list(rf.historical_years) + list(rf.forecast_years)
    hist_vals = rf.historical_values
    forecast_vals = rf.forecast_values
    lower = rf.confidence_lower
    upper = rf.confidence_upper

    fig_fan = go.Figure()

    # Historical
    fig_fan.add_trace(
        go.Scatter(
            x=rf.historical_years,
            y=hist_vals,
            name="Historical",
            line=dict(color=COLORS["primary"], width=2.5),
        )
    )

    # Fan bands at different CIs
    for alpha, label in [(0.15, "70% CI"), (0.05, "90% CI")]:
        spread = [(u - l) * alpha for u, l in zip(upper, lower)]
        fig_fan.add_trace(
            go.Scatter(
                x=rf.forecast_years + rf.forecast_years[::-1],
                y=[(f + s) for f, s in zip(forecast_vals, spread)]
                + [(f - s) for f, s in zip(forecast_vals, spread)][::-1],
                fill="toself",
                fillcolor=f"rgba(26,108,245,{0.08 if alpha == 0.15 else 0.15})",
                line=dict(color="rgba(0,0,0,0)"),
                name=label,
            )
        )

    # Central forecast
    fig_fan.add_trace(
        go.Scatter(
            x=rf.forecast_years,
            y=forecast_vals,
            name="Forecast",
            line=dict(color=COLORS["gold"], width=2, dash="dash"),
        )
    )

    fig_fan.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0)",
        title=f"{report.ticker} — Revenue Fan Chart",
        height=380,
    )
    st.plotly_chart(fig_fan, use_container_width=True)
