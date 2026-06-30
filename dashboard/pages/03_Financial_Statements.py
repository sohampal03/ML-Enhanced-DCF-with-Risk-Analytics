"""Page 3: Financial Statements Explorer"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.components.cards import section_divider, info_box
from dashboard.components.charts import financial_bar_chart, margin_trend_chart, COLORS
from src.utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Financial Statements | AlphaForge", layout="wide", page_icon="📋")
CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first.")
    st.stop()

report = st.session_state.report
data = report.financial_data

st.markdown(
    """
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">📋 Financial Statements</h1>
    <div style="color:#8b9cc8;margin-bottom:28px;">Historical income, balance sheet, and cash flow analysis</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📈 Income Statement", "🏦 Balance Sheet", "💸 Cash Flow"])


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format numbers in display DataFrame."""
    display = df.copy()
    for col in display.columns:
        if col in ("fiscal_year", "ticker"):
            continue
        try:
            display[col] = display[col].apply(
                lambda x: (
                    format_currency(x)
                    if isinstance(x, float) and abs(x) > 1e4
                    else (
                        format_percentage(x)
                        if isinstance(x, float) and abs(x) <= 2
                        else (f"{x:.2f}" if isinstance(x, float) else x)
                    )
                )
            )
        except Exception:
            pass
    return display


with tab1:
    income_df = data.income_df()
    if not income_df.empty:
        section_divider("Income Statement", badge=f"{len(income_df)} Years")
        st.dataframe(
            fmt_df(income_df.drop(columns=["ticker"], errors="ignore")),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        income_sorted = income_df.sort_values("fiscal_year")
        col1, col2 = st.columns(2)
        with col1:
            fig = financial_bar_chart(
                income_sorted["fiscal_year"].tolist(),
                income_sorted["revenue"].fillna(0).tolist(),
                "Revenue",
                report.ticker,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = margin_trend_chart(
                income_sorted["fiscal_year"].tolist(),
                income_sorted["gross_margin"].fillna(0).tolist(),
                income_sorted["ebit_margin"].fillna(0).tolist(),
                income_sorted["net_margin"].fillna(0).tolist(),
                report.ticker,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        info_box("Income statement data not available", "warning")

with tab2:
    balance_df = data.balance_df()
    if not balance_df.empty:
        section_divider("Balance Sheet", badge=f"{len(balance_df)} Years")
        st.dataframe(
            fmt_df(balance_df.drop(columns=["ticker"], errors="ignore")),
            use_container_width=True,
            hide_index=True,
        )

        balance_sorted = balance_df.sort_values("fiscal_year")
        col1, col2 = st.columns(2)
        with col1:
            fig = financial_bar_chart(
                balance_sorted["fiscal_year"].tolist(),
                balance_sorted["total_assets"].fillna(0).tolist(),
                "Total Assets",
                report.ticker,
                color=COLORS["gold"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = financial_bar_chart(
                balance_sorted["fiscal_year"].tolist(),
                balance_sorted["total_debt"].fillna(0).tolist(),
                "Total Debt",
                report.ticker,
                color=COLORS["bear"],
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        info_box("Balance sheet data not available", "warning")

with tab3:
    cf_df = data.cashflow_df()
    if not cf_df.empty:
        section_divider("Cash Flow Statement", badge=f"{len(cf_df)} Years")
        st.dataframe(
            fmt_df(cf_df.drop(columns=["ticker"], errors="ignore")),
            use_container_width=True,
            hide_index=True,
        )

        cf_sorted = cf_df.sort_values("fiscal_year")
        col1, col2 = st.columns(2)
        with col1:
            fig = financial_bar_chart(
                cf_sorted["fiscal_year"].tolist(),
                cf_sorted["operating_cash_flow"].fillna(0).tolist(),
                "Operating Cash Flow",
                report.ticker,
                color=COLORS["primary"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = financial_bar_chart(
                cf_sorted["fiscal_year"].tolist(),
                cf_sorted["free_cash_flow"].fillna(0).tolist(),
                "Free Cash Flow",
                report.ticker,
                color=COLORS["bull"],
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        info_box("Cash flow data not available", "warning")
