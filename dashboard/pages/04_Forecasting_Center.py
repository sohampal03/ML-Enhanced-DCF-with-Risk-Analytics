"""
Page 4: Forecasting Center — ML Model Comparison & Revenue Forecasts
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.cards import section_divider, metric_card, info_box
from dashboard.components.charts import revenue_forecast_chart, margin_trend_chart, COLORS
from src.utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Forecasting Center | AlphaForge", layout="wide", page_icon="🤖")

CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first from the home page.")
    st.stop()

report = st.session_state.report

st.markdown("""
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">🤖 Forecasting Center</h1>
    <div style="color:#8b9cc8;font-size:14px;margin-bottom:28px;">
        Machine learning revenue & margin forecasting with model leaderboard
    </div>
""", unsafe_allow_html=True)

# ── Revenue Forecast ─────────────────────────────────────────────────────────
rf = report.revenue_forecast
if rf:
    section_divider("Revenue Forecast", badge=rf.best_model_name)

    col_chart, col_scores = st.columns([2, 1])
    with col_chart:
        fig = revenue_forecast_chart(
            historical_years=rf.historical_years,
            historical_values=rf.historical_values,
            forecast_years=rf.forecast_years,
            forecast_values=rf.forecast_values,
            lower_ci=rf.confidence_lower,
            upper_ci=rf.confidence_upper,
            ticker=report.ticker,
            label="Revenue ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_scores:
        section_divider("Model Leaderboard", badge="Best = Lowest MAPE")
        if rf.model_leaderboard:
            lb_df = rf.to_leaderboard_df()
            st.dataframe(lb_df.drop(columns=["Rank"]), use_container_width=True, hide_index=True)
        
        st.markdown("**Feature Importances**")
        if rf.feature_importances:
            import pandas as pd
            fi = pd.DataFrame(list(rf.feature_importances.items()), columns=["Feature", "Importance"])
            fi = fi.head(8)
            fig_fi = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"],
                orientation="h",
                marker=dict(color=COLORS["primary"]),
            ))
            fig_fi.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

else:
    info_box("Revenue forecast not available — run analysis with ML enabled.", "warning")

# ── Margin Forecasts ─────────────────────────────────────────────────────────
st.markdown("---")
section_divider("Margin Forecasts")

mf = report.margin_forecasts
if mf:
    tabs = st.tabs(["EBIT Margin", "Gross Margin", "Net Margin"])
    margin_keys = ["ebit_margin", "gross_margin", "net_margin"]
    
    for tab, key in zip(tabs, margin_keys):
        with tab:
            if key in mf:
                mfr = mf[key]
                fig = revenue_forecast_chart(
                    historical_years=mfr.historical_years,
                    historical_values=[v * 100 for v in mfr.historical_values],
                    forecast_years=mfr.forecast_years,
                    forecast_values=[v * 100 for v in mfr.forecast_values],
                    lower_ci=[v * 100 for v in mfr.confidence_lower],
                    upper_ci=[v * 100 for v in mfr.confidence_upper],
                    ticker=report.ticker,
                    label=key.replace("_", " ").title() + " (%)",
                )
                st.plotly_chart(fig, use_container_width=True)
                info_box(f"Best model: {mfr.best_model_name}", "info")
else:
    info_box("Margin forecasts not available.", "warning")

# ── Historical Margin Trends ─────────────────────────────────────────────────
st.markdown("---")
section_divider("Historical Margin Trends")

income_df = report.financial_data.income_df() if report.financial_data else None
if income_df is not None and not income_df.empty:
    income_df = income_df.sort_values("fiscal_year")
    years = income_df["fiscal_year"].tolist()
    
    fig_margins = margin_trend_chart(
        years=years,
        gross_margins=income_df["gross_margin"].fillna(0).tolist(),
        ebit_margins=income_df["ebit_margin"].fillna(0).tolist(),
        net_margins=income_df["net_margin"].fillna(0).tolist(),
        ticker=report.ticker,
    )
    st.plotly_chart(fig_margins, use_container_width=True)
