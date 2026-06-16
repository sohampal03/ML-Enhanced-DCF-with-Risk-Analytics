"""
Page 2: Company Explorer — Business overview, price history, peer comparison
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.components.cards import metric_card, section_divider, info_box
from dashboard.components.charts import candlestick_chart, peer_radar_chart, financial_bar_chart, COLORS
from src.utils.helpers import format_currency, format_percentage, format_multiple

st.set_page_config(page_title="Company Explorer | AlphaForge", layout="wide", page_icon="🏢")

CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first from the home page.")
    st.stop()

report = st.session_state.report
data = report.financial_data
info = data.company_info

st.markdown(f"""
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:4px;">🏢 {info.name}</h1>
    <div style="color:#8b9cc8;font-size:14px;margin-bottom:28px;">
        {info.sector or 'N/A'} · {info.industry or 'N/A'} · {info.country or 'N/A'} · {info.exchange or 'N/A'}
    </div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    ("Market Cap", format_currency(info.market_cap or 0), "🏢"),
    ("Current Price", format_currency(info.current_price or 0), "💰"),
    ("P/E Ratio", format_multiple(info.pe_ratio or 0), "📊"),
    ("Beta", f"{info.beta:.2f}" if info.beta else "N/A", "📐"),
    ("ROE", format_percentage(info.return_on_equity or 0), "📈"),
    ("Employees", f"{info.employees:,}" if info.employees else "N/A", "👥"),
]
for col, (label, val, icon) in zip([c1, c2, c3, c4, c5, c6], kpis):
    with col:
        metric_card(label, val, icon=icon)

# ── Description ───────────────────────────────────────────────────────────────
if info.description:
    st.markdown("---")
    section_divider("Business Description")
    st.markdown(
        f'<div style="color:#8b9cc8;font-size:14px;line-height:1.8;max-width:900px;">{info.description[:800]}{"..." if len(info.description) > 800 else ""}</div>',
        unsafe_allow_html=True,
    )

# ── Price History ─────────────────────────────────────────────────────────────
st.markdown("---")
section_divider("Price History")

if data.price_history:
    try:
        price_df = pd.DataFrame(data.price_history)
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(list(price_df.keys())[:len(price_df)])

        period_opts = {"1Y": -252, "3Y": -756, "5Y": -1260, "All": 0}
        period = st.radio("Period", list(period_opts.keys()), horizontal=True, index=2)
        cutoff = period_opts[period]
        display_df = price_df.iloc[cutoff:] if cutoff else price_df

        fig = candlestick_chart(display_df, info.ticker, show_volume=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        info_box(f"Price chart unavailable: {e}", "warning")
else:
    info_box("Price history not available", "warning")

# ── Revenue & Earnings Trend ──────────────────────────────────────────────────
st.markdown("---")
income_df = data.income_df().sort_values("fiscal_year")
if not income_df.empty:
    col_rev, col_ni = st.columns(2)
    with col_rev:
        section_divider("Revenue Trend")
        fig = financial_bar_chart(
            income_df["fiscal_year"].tolist(),
            income_df["revenue"].fillna(0).tolist(),
            "Revenue",
            info.ticker,
            color=COLORS["primary"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_ni:
        section_divider("Net Income Trend")
        fig = financial_bar_chart(
            income_df["fiscal_year"].tolist(),
            income_df["net_income"].fillna(0).tolist(),
            "Net Income",
            info.ticker,
            color=COLORS["bull"],
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Key Ratios Table ──────────────────────────────────────────────────────────
st.markdown("---")
section_divider("Key Financial Ratios")
ratios_data = {
    "Metric": ["P/E Ratio", "Forward P/E", "P/B Ratio", "EV/EBITDA", "Debt/Equity", "ROE", "ROA", "Net Margin", "Revenue Growth", "Dividend Yield"],
    "Value": [
        format_multiple(info.pe_ratio),
        format_multiple(info.forward_pe),
        format_multiple(info.pb_ratio),
        format_multiple(info.ev_ebitda),
        format_multiple(info.debt_to_equity, "x", 2),
        format_percentage(info.return_on_equity or 0),
        format_percentage(info.return_on_assets or 0),
        format_percentage(info.profit_margin or 0),
        format_percentage(info.revenue_growth or 0),
        format_percentage(info.dividend_yield or 0),
    ],
}
st.dataframe(pd.DataFrame(ratios_data), use_container_width=True, hide_index=True)

# ── Comps Peer Table ──────────────────────────────────────────────────────────
if report.comps_result and report.comps_result.company_multiples:
    st.markdown("---")
    section_divider("Peer Comparison", badge=f"{len(report.comps_result.company_multiples)} companies")
    
    comps_df = report.comps_result.to_dataframe()
    # Highlight target row
    def highlight_target(row):
        if row.get("Is Target", False):
            return ["background-color: rgba(26,108,245,0.1)"] * len(row)
        return [""] * len(row)
    
    display_df = comps_df.drop(columns=["Is Target"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
