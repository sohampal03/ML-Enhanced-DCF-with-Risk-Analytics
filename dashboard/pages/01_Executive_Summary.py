"""Page 1: Executive Summary — full overview after analysis"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from dashboard.components.cards import (
    metric_card, recommendation_badge, section_divider,
    traffic_light, valuation_summary_banner, probability_bar, info_box,
)
from dashboard.components.charts import dcf_waterfall_chart, valuation_gauge, COLORS
from src.utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Executive Summary | AlphaForge", layout="wide", page_icon="📊")
CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first from the home page."); st.stop()

report = st.session_state.report
if not report.is_complete:
    st.error("Analysis incomplete. Please re-run."); st.stop()

dcf = report.dcf_result
mc = report.monte_carlo
data = report.financial_data
info = data.company_info
classification = report.classification

st.markdown("""
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">📊 Executive Summary</h1>
    <div style="color:#8b9cc8;margin-bottom:28px;">Complete valuation overview and investment recommendation</div>
""", unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────
valuation_summary_banner(
    ticker=report.ticker,
    company_name=report.company_name,
    current_price=dcf.current_price,
    intrinsic_value=dcf.intrinsic_value_per_share,
    recommendation=dcf.recommendation,
    margin_of_safety=dcf.margin_of_safety,
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: metric_card("Market Cap", format_currency(info.market_cap or 0), icon="🏢")
with k2: metric_card("Enterprise Value", format_currency(dcf.enterprise_value), icon="🏗️")
with k3: metric_card("WACC", format_percentage(dcf.wacc), icon="⚖️")
with k4: metric_card("Terminal Growth", format_percentage(dcf.terminal_growth_rate), icon="∞")
with k5: metric_card("Beta", f"{info.beta:.2f}" if info.beta else "N/A", icon="📐")
with k6: metric_card("Shares Out.", f"{info.shares_outstanding / 1e9:.2f}B" if info.shares_outstanding else "N/A", icon="📋")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main 3-col layout ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    section_divider("Investment Verdict")
    recommendation_badge(dcf.recommendation)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if classification:
        st.markdown(f"""
            <div style="background:#0f1628;border:1px solid #1e2d4a;border-radius:10px;padding:16px;margin-top:12px;">
                <div style="color:#4a5a7e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">ML Classification</div>
                <div style="color:{classification.predicted_color};font-size:18px;font-weight:700;">{classification.predicted_label}</div>
                <div style="color:#8b9cc8;font-size:12px;margin-top:4px;">Confidence: {classification.confidence:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    section_divider("Traffic Lights")
    
    mos = dcf.margin_of_safety
    traffic_light("Valuation Safety", "green" if mos > 0.20 else "yellow" if mos > 0 else "red", f"{mos*100:.1f}% MOS")
    
    pe_ok = info.pe_ratio and info.pe_ratio < 30 if info.pe_ratio else None
    traffic_light("P/E Ratio", "green" if pe_ok else "yellow" if pe_ok is None else "red",
                  f"{info.pe_ratio:.1f}x" if info.pe_ratio else "N/A")
    
    growth_ok = info.revenue_growth and info.revenue_growth > 0.05
    traffic_light("Revenue Growth", "green" if growth_ok else "red",
                  format_percentage(info.revenue_growth or 0))
    
    de = info.debt_to_equity
    traffic_light("Debt Level", "green" if de and de < 1 else "yellow" if de and de < 2 else "red",
                  f"{de:.2f}x" if de else "N/A")

with col2:
    section_divider("Value Bridge")
    fig_wf = dcf_waterfall_chart(
        pv_fcff=dcf.pv_fcff_sum,
        pv_terminal=dcf.pv_terminal_value,
        enterprise_value=dcf.enterprise_value,
        total_debt=dcf.total_debt,
        cash=dcf.cash,
        equity_value=dcf.equity_value,
        intrinsic_per_share=dcf.intrinsic_value_per_share,
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    
    if mc:
        section_divider("Valuation Gauge")
        fig_gauge = valuation_gauge(
            current_price=dcf.current_price,
            intrinsic_value=dcf.intrinsic_value_per_share,
            p5=mc.p5, p95=mc.p95, ticker=report.ticker,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    section_divider("Risk Probabilities")
    if mc:
        probability_bar("P(Undervalued)", mc.prob_undervalued, COLORS["bull"])
        probability_bar("P(Fairly Valued)", mc.prob_in_range, COLORS["gold"])
        probability_bar("P(Overvalued)", mc.prob_overvalued, COLORS["bear"])
        st.markdown("<br>", unsafe_allow_html=True)
        section_divider("Monte Carlo Range")
        scenarios = [
            ("Bear (P5)", mc.p5, COLORS["bear"]),
            ("Median (P50)", mc.p50, COLORS["primary"]),
            ("Bull (P95)", mc.p95, COLORS["bull"]),
        ]
        for label, val, color in scenarios:
            st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     border-bottom:1px solid #1e2d4a;padding:10px 0;">
                    <span style="color:#8b9cc8;font-size:13px;">{label}</span>
                    <span style="color:{color};font-size:15px;font-weight:700;">{format_currency(val)}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        info_box("Run analysis with Monte Carlo enabled for probability estimates", "info")
    
    st.markdown("<br>", unsafe_allow_html=True)
    section_divider("Key Assumptions")
    assumptions = [
        ("Forecast Years", str(dcf.forecast_years)),
        ("WACC", format_percentage(dcf.wacc)),
        ("Terminal Growth", format_percentage(dcf.terminal_growth_rate)),
        ("EBIT Margin", format_percentage(dcf.ebit_margin_forecast)),
        ("Tax Rate", format_percentage(dcf.tax_rate)),
    ]
    for label, val in assumptions:
        st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;
                 border-bottom:1px solid #1e2d4a;">
                <span style="color:#4a5a7e;font-size:12px;">{label}</span>
                <span style="color:#e8edf8;font-size:12px;font-weight:600;">{val}</span>
            </div>
        """, unsafe_allow_html=True)

# ── Company Description ───────────────────────────────────────────────────────
if info.description:
    st.markdown("---")
    section_divider("Investment Thesis")
    st.markdown(f'<div style="color:#8b9cc8;font-size:14px;line-height:1.8;max-width:1000px;">{info.description[:600]}...</div>', unsafe_allow_html=True)

# ── Errors ────────────────────────────────────────────────────────────────────
if report.errors:
    st.markdown("---")
    with st.expander(f"⚠️ Analysis Notes ({len(report.errors)} items)", expanded=False):
        for module, error in report.errors.items():
            info_box(f"{module}: {error}", "warning")
