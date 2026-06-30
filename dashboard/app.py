"""
AlphaForge Dashboard — Main Entry Point

Bloomberg Terminal × Modern SaaS Design
Powered by AI-enhanced DCF Valuation
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AlphaForge — Intelligent Business Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/yourusername/ai-business-valuation",
        "Report a bug": "https://github.com/yourusername/ai-business-valuation/issues",
        "About": "AlphaForge — AI-Powered Business Valuation Platform v1.0",
    },
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
CSS_FILE = Path(__file__).parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from configs.settings import settings
from src.utils.logging import configure_logging
from src.valuation.orchestrator import FullValuationReport, ValuationOrchestrator

configure_logging(log_level=settings.log_level)

# ── Session State Defaults ────────────────────────────────────────────────────
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "report" not in st.session_state:
    st.session_state.report = None
if "loading" not in st.session_state:
    st.session_state.loading = False
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = ValuationOrchestrator()

# ── Brand Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #060d1f 0%, #0f1628 100%);
        border-bottom: 1px solid #1e2d4a;
        padding: 16px 32px;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="
                font-size:26px;font-weight:900;letter-spacing:-1px;
                background:linear-gradient(135deg,#1a6cf5,#f0b429);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            ">AlphaForge</div>
            <div style="
                background:rgba(26,108,245,0.12);color:#1a6cf5;
                border:1px solid rgba(26,108,245,0.3);border-radius:20px;
                padding:3px 10px;font-size:11px;font-weight:600;
            ">INSTITUTIONAL GRADE</div>
        </div>
        <div style="color:#4a5a7e;font-size:12px;font-family:'JetBrains Mono',monospace;">
            AI-Powered DCF · Monte Carlo · Explainable AI
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0 24px;">
            <div style="font-size:28px;font-weight:900;
                background:linear-gradient(135deg,#1a6cf5,#f0b429);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                AlphaForge
            </div>
            <div style="color:#4a5a7e;font-size:11px;margin-top:4px;">
                Business Valuation Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Ticker Search ─────────────────────────────────────────────────────────
    st.markdown(
        '<div style="color:#8b9cc8;font-size:11px;font-weight:600;'
        'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">TICKER SEARCH</div>',
        unsafe_allow_html=True,
    )

    ticker_input = (
        st.text_input(
            "Enter Ticker Symbol",
            value=st.session_state.ticker,
            placeholder="e.g. AAPL, MSFT, INFY.NS",
            label_visibility="collapsed",
            key="ticker_input_main",
        )
        .upper()
        .strip()
    )

    # Quick picks
    st.markdown(
        '<div style="color:#4a5a7e;font-size:11px;margin-bottom:8px;">Quick picks:</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    quick_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN"]
    for i, qt in enumerate(quick_tickers):
        with cols[i % 3]:
            if st.button(qt, key=f"quick_{qt}", use_container_width=True):
                ticker_input = qt

    analyze_clicked = st.button(
        "🔍 Analyze",
        use_container_width=True,
        type="primary",
        key="analyze_btn_main",
    )

    st.markdown("---")

    # ── Analysis Options ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="color:#8b9cc8;font-size:11px;font-weight:600;'
        'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">ANALYSIS OPTIONS</div>',
        unsafe_allow_html=True,
    )

    run_ml = st.checkbox("🤖 ML Forecasting", value=True, key="opt_ml")
    run_mc = st.checkbox("🎲 Monte Carlo (10K sims)", value=True, key="opt_mc")
    run_comps = st.checkbox("📊 Comparable Analysis", value=True, key="opt_comps")
    forecast_years = st.slider("Forecast Horizon", 5, 15, 10, key="opt_years")

    st.markdown("---")

    # ── Status Panel ──────────────────────────────────────────────────────────
    if st.session_state.report:
        report: FullValuationReport = st.session_state.report
        st.markdown(
            '<div style="color:#8b9cc8;font-size:11px;font-weight:600;'
            'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">CURRENT ANALYSIS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:#1a6cf5;font-size:16px;font-weight:700;">{report.ticker}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:#4a5a7e;font-size:12px;">{report.company_name}</div>',
            unsafe_allow_html=True,
        )

        if report.dcf_result:
            rec = report.dcf_result.recommendation
            rec_color = {
                "STRONG BUY": "#00c851",
                "BUY": "#00c851",
                "HOLD": "#f0b429",
                "SELL": "#ff4444",
                "STRONG SELL": "#ff4444",
            }.get(rec, "#8b9cc8")
            st.markdown(
                f'<div style="color:{rec_color};font-size:13px;font-weight:700;margin-top:8px;">{rec}</div>',
                unsafe_allow_html=True,
            )

        if report.errors:
            st.markdown(
                f'<div style="color:#f0b429;font-size:11px;margin-top:8px;">⚠ {len(report.errors)} module(s) had issues</div>',
                unsafe_allow_html=True,
            )

# ── Main Analysis Logic ───────────────────────────────────────────────────────
if analyze_clicked and ticker_input:
    st.session_state.ticker = ticker_input
    st.session_state.loading = True

    with st.spinner(f"⏳ Running full valuation analysis for **{ticker_input}**..."):
        try:
            orchestrator: ValuationOrchestrator = st.session_state.orchestrator
            report = orchestrator.analyze(
                ticker=ticker_input,
                run_ml=run_ml,
                run_monte_carlo=run_mc,
                run_comps=run_comps,
                forecast_years=forecast_years,
            )
            st.session_state.report = report

            if report.is_complete:
                st.success(
                    f"✅ Analysis complete for **{report.company_name}** ({ticker_input}). "
                    f"Navigate using the sidebar pages."
                )
            else:
                st.warning(
                    f"⚠️ Partial analysis completed. Some modules encountered errors: "
                    f"{', '.join(report.errors.keys())}"
                )
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.loading = False

    st.session_state.loading = False

# ── Landing / Welcome Screen ──────────────────────────────────────────────────
if not st.session_state.report:
    # Hero section
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px 40px;">
            <div style="
                font-size:54px;font-weight:900;letter-spacing:-2px;
                background:linear-gradient(135deg,#e8edf8 0%,#8b9cc8 40%,#1a6cf5 70%,#f0b429 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;
                line-height:1.1;
                margin-bottom:20px;
            ">
                AI-Powered Business<br>Valuation
            </div>
            <div style="color:#8b9cc8;font-size:18px;max-width:600px;margin:0 auto 40px;line-height:1.6;">
                Institutional-grade DCF valuation enhanced with Machine Learning forecasting,
                Monte Carlo risk analysis, and Explainable AI — for any publicly traded company.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📈", "DCF Valuation", "FCFF-based two-stage DCF with Gordon Growth Model terminal value"),
        (
            "🤖",
            "ML Forecasting",
            "XGBoost, LightGBM, CatBoost revenue & margin models with leaderboard",
        ),
        ("🎲", "Monte Carlo", "10,000+ simulations quantifying valuation uncertainty"),
        ("🔍", "Explainable AI", "SHAP values explaining every prediction in plain English"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#111827;border:1px solid #1e2d4a;
                    border-radius:16px;padding:24px;text-align:center;
                    transition:all 0.3s;height:180px;
                ">
                    <div style="font-size:36px;margin-bottom:12px;">{icon}</div>
                    <div style="color:#e8edf8;font-weight:700;font-size:15px;margin-bottom:8px;">{title}</div>
                    <div style="color:#4a5a7e;font-size:12px;line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;color:#4a5a7e;font-size:13px;padding:16px;">
            Enter a ticker symbol in the sidebar and click <strong style="color:#1a6cf5;">Analyze</strong>
            to generate a complete institutional valuation report.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif st.session_state.report and st.session_state.report.is_complete:
    report = st.session_state.report

    # Show executive summary on home page
    from dashboard.components.cards import (
        metric_card,
        recommendation_badge,
        section_divider,
        traffic_light,
        valuation_summary_banner,
    )
    from dashboard.components.charts import dcf_waterfall_chart
    from src.utils.helpers import format_currency, format_percentage

    dcf = report.dcf_result
    info = report.financial_data.company_info

    valuation_summary_banner(
        ticker=report.ticker,
        company_name=report.company_name,
        current_price=dcf.current_price,
        intrinsic_value=dcf.intrinsic_value_per_share,
        recommendation=dcf.recommendation,
        margin_of_safety=dcf.margin_of_safety,
    )

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Market Cap", format_currency(info.market_cap or 0), icon="🏢")
    with c2:
        metric_card("Current Price", format_currency(dcf.current_price), icon="💰")
    with c3:
        mos = dcf.margin_of_safety * 100
        metric_card(
            "Margin of Safety",
            f"{mos:.1f}%",
            delta=f"{'Safe' if mos > 20 else 'Caution'}",
            delta_positive=mos > 20,
            icon="🛡️",
        )
    with c4:
        metric_card("WACC", format_percentage(dcf.wacc), icon="📊")
    with c5:
        metric_card("Forecast Years", str(dcf.forecast_years), icon="📅")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        section_divider("DCF Value Bridge")
        fig = dcf_waterfall_chart(
            pv_fcff=dcf.pv_fcff_sum,
            pv_terminal=dcf.pv_terminal_value,
            enterprise_value=dcf.enterprise_value,
            total_debt=dcf.total_debt,
            cash=dcf.cash,
            equity_value=dcf.equity_value,
            intrinsic_per_share=dcf.intrinsic_value_per_share,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_divider("Valuation Assessment")
        recommendation_badge(dcf.recommendation)
        st.markdown("<br>", unsafe_allow_html=True)

        # Monte Carlo probability
        if report.monte_carlo:
            mc = report.monte_carlo
            from dashboard.components.cards import probability_bar

            probability_bar("P(Undervalued)", mc.prob_undervalued, "#00c851")
            probability_bar("P(Fairly Valued)", mc.prob_in_range, "#f0b429")
            probability_bar("P(Overvalued)", mc.prob_overvalued, "#ff4444")
        else:
            traffic_light(
                "Valuation",
                (
                    "green"
                    if dcf.margin_of_safety > 0.15
                    else "red" if dcf.margin_of_safety < -0.15 else "yellow"
                ),
                f"{dcf.margin_of_safety * 100:.1f}% MOS",
            )

    st.markdown(
        """
        <div style="text-align:center;color:#4a5a7e;font-size:12px;padding:24px 0 8px;">
            📌 Use the <strong style="color:#1a6cf5;">sidebar navigation</strong>
            to explore DCF Lab, Forecasting Center, Risk Analytics, and more.
        </div>
        """,
        unsafe_allow_html=True,
    )
