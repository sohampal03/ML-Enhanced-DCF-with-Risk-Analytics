"""
Page 5: DCF Lab — Interactive DCF with live recalculation.

The MOST IMPORTANT page. Users can adjust all DCF assumptions via sliders
and see intrinsic value update in real-time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from dashboard.components.cards import (
    info_box,
    metric_card,
    recommendation_badge,
    section_divider,
    valuation_summary_banner,
)
from dashboard.components.charts import (
    dcf_waterfall_chart,
    sensitivity_heatmap,
    revenue_forecast_chart,
    COLORS,
)
from src.finance.engines.dcf_engine import DCFEngine
from src.utils.helpers import format_currency, format_percentage
import plotly.graph_objects as go

st.set_page_config(page_title="DCF Lab | AlphaForge", layout="wide", page_icon="🔬")

# Load CSS
CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Check for report ──────────────────────────────────────────────────────────
if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first from the home page.")
    st.stop()

report = st.session_state.report
if not report.dcf_result or not report.financial_data:
    st.error("DCF data not available.")
    st.stop()

dcf = report.dcf_result
data = report.financial_data
wacc_result = report.wacc_result

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:24px;">
        <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin:0;">
            🔬 DCF Lab
        </h1>
        <div style="color:#8b9cc8;font-size:14px;margin-top:6px;">
            Adjust assumptions and watch intrinsic value update in real-time
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Formula Education Box ─────────────────────────────────────────────────────
with st.expander("📚 How DCF Works — Formula & Theory", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Discounted Cash Flow (DCF) Formula:**
        
        ```
        Enterprise Value = Σ [FCFF_t / (1+WACC)^t] + TV / (1+WACC)^n
        
        Where:
          FCFF = EBIT × (1-Tax) + D&A − ΔNWC − CAPEX
          TV   = FCFF_n × (1+g) / (WACC − g)       [Gordon Growth]
          WACC = (E/V)×Ke + (D/V)×Kd×(1-T)
          Ke   = Rf + β × (Rm − Rf)                 [CAPM]
        ```
        
        **Equity Value = Enterprise Value − Debt + Cash**
        
        **Intrinsic Value / Share = Equity Value / Shares Outstanding**
        """)
    with c2:
        st.markdown("""
        **Key Assumptions & Their Impact:**
        
        | Assumption | ↑ Increases | ↓ Decreases |
        |---|---|---|
        | Revenue Growth | Intrinsic Value | — |
        | WACC | — | Intrinsic Value |
        | Terminal Growth | Intrinsic Value | — |
        | EBIT Margin | Intrinsic Value | — |
        | CAPEX % | — | Intrinsic Value |
        
        > **Rule of thumb:** A 1% change in WACC typically changes intrinsic value by 10–20%.
        > Terminal value often represents 60–80% of total value in high-growth companies.
        """)

st.markdown("---")

# ── Left: Assumption Sliders | Right: Live DCF Output ────────────────────────
col_sliders, col_results = st.columns([1, 2])

with col_sliders:
    st.markdown(
        '<div style="color:#8b9cc8;font-size:11px;font-weight:600;letter-spacing:1.5px;'
        'text-transform:uppercase;margin-bottom:16px;">⚙ ADJUST ASSUMPTIONS</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Revenue Growth**")
    rev_growth_y1 = (
        st.slider(
            "Year 1 Growth (%)",
            -20,
            60,
            int((dcf.revenue_growth_rates[0] if dcf.revenue_growth_rates else 0.08) * 100),
            key="rev_y1",
        )
        / 100
    )
    rev_growth_terminal = (
        st.slider(
            "Terminal Growth Rate (%)",
            1,
            6,
            int(dcf.terminal_growth_rate * 100),
            key="rev_terminal",
        )
        / 100
    )

    st.markdown("**Margins & Costs**")
    ebit_margin = (
        st.slider("EBIT Margin (%)", 1, 60, int(dcf.ebit_margin_forecast * 100), key="ebit_margin")
        / 100
    )
    tax_rate = st.slider("Tax Rate (%)", 5, 40, int(dcf.tax_rate * 100), key="tax_rate") / 100
    capex_pct = st.slider("CAPEX (% of Revenue)", 1, 30, 5, key="capex_pct") / 100
    da_pct = st.slider("D&A (% of Revenue)", 1, 20, 4, key="da_pct") / 100

    st.markdown("**Discount Rate**")
    wacc_val = (
        st.slider(
            "WACC (%)",
            4.0,
            25.0,
            float(round(dcf.wacc * 100, 1)),
            step=0.1,
            key="wacc_slider",
        )
        / 100
    )

    st.markdown("**Forecast Period**")
    n_years = st.slider("Forecast Years", 5, 15, dcf.forecast_years, key="n_years")

    st.markdown("---")
    recalc = st.button("🔄 Recalculate", use_container_width=True, type="primary")

# ── Live DCF Calculation ──────────────────────────────────────────────────────
# Build growth schedule from Y1 → terminal
growth_rates = list(np.linspace(rev_growth_y1, max(rev_growth_terminal, 0.01), n_years))

# Recompute DCF with current slider values
engine = DCFEngine()
try:
    live_dcf = engine.compute(
        data=data,
        fcff_result=report.fcff_result,
        wacc_result=wacc_result,
        revenue_growth_rates=growth_rates,
        ebit_margin=ebit_margin,
        wacc_override=wacc_val,
        terminal_growth_rate=rev_growth_terminal,
        forecast_years=n_years,
        tax_rate_override=tax_rate,
        capex_pct_revenue=capex_pct,
        da_pct_revenue=da_pct,
        build_sensitivity=True,
    )
    live_ok = True
except Exception as e:
    live_dcf = dcf  # Fallback to original
    live_ok = False
    st.warning(f"Recalculation warning: {e}")

with col_results:
    section_divider("Live Valuation Output", badge="AUTO-REFRESH")

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card(
            "Intrinsic Value", format_currency(live_dcf.intrinsic_value_per_share), icon="💎"
        )
    with k2:
        metric_card("Market Price", format_currency(live_dcf.current_price), icon="📈")
    with k3:
        mos_pct = live_dcf.margin_of_safety * 100
        metric_card(
            "Margin of Safety",
            f"{mos_pct:.1f}%",
            delta=f"{'Undervalued' if mos_pct > 15 else 'Overvalued' if mos_pct < -15 else 'Fair'}",
            delta_positive=mos_pct > 15,
            icon="🛡️",
        )
    with k4:
        upside = live_dcf.upside_potential * 100
        metric_card(
            "Upside/Downside",
            f"{upside:+.1f}%",
            delta_positive=upside > 0,
            icon="🎯",
        )

    # Recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    recommendation_badge(live_dcf.recommendation)
    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall
    section_divider("Value Bridge")
    fig_wf = dcf_waterfall_chart(
        pv_fcff=live_dcf.pv_fcff_sum,
        pv_terminal=live_dcf.pv_terminal_value,
        enterprise_value=live_dcf.enterprise_value,
        total_debt=live_dcf.total_debt,
        cash=live_dcf.cash,
        equity_value=live_dcf.equity_value,
        intrinsic_per_share=live_dcf.intrinsic_value_per_share,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ── DCF Projection Table ──────────────────────────────────────────────────────
st.markdown("---")
section_divider("Year-by-Year DCF Projections", badge=f"{n_years} Years Explicit")

proj_df = __import__("pandas").DataFrame([p.to_dict() for p in live_dcf.projections])
# Format currency columns
for col in ["Revenue", "EBIT", "NOPAT", "D&A", "CAPEX", "ΔNWC", "FCFF", "PV of FCFF"]:
    if col in proj_df.columns:
        proj_df[col] = proj_df[col].apply(lambda x: format_currency(x))
# Format percentage columns
for col in ["Revenue Growth", "EBIT Margin"]:
    if col in proj_df.columns:
        proj_df[col] = proj_df[col].apply(
            lambda x: format_percentage(x) if isinstance(x, float) else x
        )
# Format discount factor
if "Discount Factor" in proj_df.columns:
    proj_df["Discount Factor"] = proj_df["Discount Factor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) else x
    )

st.dataframe(
    proj_df,
    use_container_width=True,
    hide_index=True,
)

# ── Sensitivity Matrix ────────────────────────────────────────────────────────
if live_dcf.sensitivity_matrix is not None:
    st.markdown("---")
    section_divider("WACC × Terminal Growth Sensitivity Matrix")
    st.caption(
        f"Each cell shows the estimated intrinsic value per share. "
        f"Current price: **{format_currency(live_dcf.current_price)}** (highlighted cells = above current)"
    )
    fig_heat = sensitivity_heatmap(
        sensitivity_df=live_dcf.sensitivity_matrix,
        current_price=live_dcf.current_price,
        ticker=report.ticker,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── WACC Breakdown ────────────────────────────────────────────────────────────
if wacc_result:
    st.markdown("---")
    section_divider("WACC Decomposition")
    info_box(wacc_result.capm_formula, "info")
    info_box(wacc_result.wacc_formula, "success")
    wacc_df = __import__("pandas").DataFrame(wacc_result.to_dict())
    st.dataframe(wacc_df, use_container_width=True, hide_index=True)
