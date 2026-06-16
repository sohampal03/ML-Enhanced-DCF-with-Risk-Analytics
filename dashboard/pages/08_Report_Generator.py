"""Page 9: Report Generator — Export PDF/HTML institutional reports"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import io
import datetime
import streamlit as st
import pandas as pd

from dashboard.components.cards import section_divider, info_box, metric_card
from src.utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Report Generator | AlphaForge", layout="wide", page_icon="📄")
CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first."); st.stop()

report = st.session_state.report
dcf = report.dcf_result
info = report.financial_data.company_info if report.financial_data else None


def generate_html_report(report) -> str:
    """Generate a complete HTML investment report."""
    dcf = report.dcf_result
    mc = report.monte_carlo
    info = report.financial_data.company_info if report.financial_data else None
    ts = datetime.datetime.now().strftime("%B %d, %Y")

    rec_color = {
        "STRONG BUY": "#00c851", "BUY": "#00c851",
        "HOLD": "#f0b429", "SELL": "#ff4444", "STRONG SELL": "#ff4444",
    }.get(dcf.recommendation if dcf else "HOLD", "#8b9cc8")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaForge Investment Report — {report.ticker}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Arial', sans-serif; background: #f8f9fa; color: #1a1a2e; }}
  .cover {{ background: linear-gradient(135deg, #0a0e1a, #1a2540);
            color: white; padding: 60px 48px; min-height: 260px; }}
  .cover h1 {{ font-size: 36px; font-weight: 900; margin-bottom: 8px; }}
  .cover .subtitle {{ color: #8b9cc8; font-size: 16px; }}
  .badge {{ display: inline-block; background: rgba(26,108,245,0.2); color: #5b9cf5;
             border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 700;
             margin-top: 12px; }}
  .section {{ padding: 32px 48px; background: white; margin-bottom: 2px; }}
  .section-title {{ font-size: 20px; font-weight: 800; color: #0a0e1a;
                    border-bottom: 2px solid #1a6cf5; padding-bottom: 12px; margin-bottom: 20px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px;
           padding: 16px; border-top: 3px solid #1a6cf5; }}
  .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                color: #666; font-weight: 600; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 22px; font-weight: 800; color: #0a0e1a; }}
  .rec-badge {{ display: inline-block; padding: 12px 32px; border-radius: 8px;
                font-size: 24px; font-weight: 900; color: {rec_color};
                background: rgba(0,0,0,0.04); border: 2px solid {rec_color}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #0a0e1a; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #f0f0f0; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .footer {{ background: #0a0e1a; color: #4a5a7e; padding: 20px 48px;
             font-size: 11px; text-align: center; }}
  .highlight {{ background: rgba(26,108,245,0.05); border-left: 3px solid #1a6cf5;
                padding: 12px 16px; border-radius: 4px; margin: 12px 0; }}
</style>
</head>
<body>

<div class="cover">
  <div style="font-size:13px;color:#f0b429;font-weight:700;letter-spacing:2px;margin-bottom:16px;">
    VALUATION RESEARCH REPORT — CONFIDENTIAL
  </div>
  <h1>{report.company_name}</h1>
  <div class="subtitle">{report.ticker} · {info.sector if info else 'N/A'} · {info.industry if info else 'N/A'}</div>
  <div class="badge">AI-ENHANCED DCF VALUATION</div>
  <div style="margin-top:24px;font-size:12px;color:#4a5a7e;">Generated: {ts} · AlphaForge Platform v1.0</div>
</div>

<div class="section">
  <div class="section-title">Executive Summary</div>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Current Price</div>
      <div class="kpi-value">{format_currency(dcf.current_price) if dcf else 'N/A'}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Intrinsic Value</div>
      <div class="kpi-value" style="color:#1a6cf5;">{format_currency(dcf.intrinsic_value_per_share) if dcf else 'N/A'}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Margin of Safety</div>
      <div class="kpi-value">{format_percentage(dcf.margin_of_safety) if dcf else 'N/A'}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Verdict</div>
      <div class="kpi-value" style="color:{rec_color};">{dcf.recommendation if dcf else 'N/A'}</div>
    </div>
  </div>
  
  <div class="rec-badge">{dcf.recommendation if dcf else 'N/A'}</div>
  
  {'<div class="highlight">Monte Carlo: ' + f'{report.monte_carlo.n_simulations:,} simulations → P(Undervalued) = {report.monte_carlo.prob_undervalued:.1%}</div>' if report.monte_carlo else ''}
</div>

<div class="section">
  <div class="section-title">DCF Valuation Summary</div>
  {''.join(f'''<table><tr><th>Metric</th><th>Value</th></tr>
  {''.join(f"<tr><td>{r}</td><td>{v}</td></tr>" for r, v in zip(dcf.to_summary_dict()['Metric'], dcf.to_summary_dict()['Value']))}
  </table>''' if dcf else [''])}
</div>

{'<div class="section"><div class="section-title">Year-by-Year Projections</div>' + 
  '<table><tr>' + ''.join(f"<th>{k}</th>" for k in ["Year","Revenue","EBIT","FCFF","Discount Factor","PV FCFF"]) + '</tr>' +
  ''.join('<tr>' + ''.join(f'<td>{p.to_dict().get(k, "")}</td>' for k in ["Year","projected_revenue","ebit","fcff","discount_factor","pv_fcff"]) + '</tr>' for p in dcf.projections) +
  '</table></div>' if dcf else ''}

<div class="footer">
  <p>This report was generated by AlphaForge — an AI-powered valuation platform. This is not financial advice.</p>
  <p>All valuations are based on publicly available financial data and ML-based forecasts. Past performance is not indicative of future results.</p>
</div>

</body>
</html>"""
    return html


# ── Page UI ───────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">📄 Report Generator</h1>
    <div style="color:#8b9cc8;margin-bottom:28px;">Export institutional-quality investment reports</div>
""", unsafe_allow_html=True)

col_opts, col_preview = st.columns([1, 2])

with col_opts:
    section_divider("Report Options")
    
    include_dcf = st.checkbox("Include DCF Analysis", value=True)
    include_mc = st.checkbox("Include Monte Carlo", value=True)
    include_comps = st.checkbox("Include Peer Comparison", value=True)
    include_forecasts = st.checkbox("Include ML Forecasts", value=True)
    
    st.markdown("---")
    
    report_format = st.selectbox("Export Format", ["HTML Report", "CSV Data Export"])
    
    st.markdown("---")
    
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating institutional report..."):
            if report_format == "HTML Report":
                html_content = generate_html_report(report)
                st.download_button(
                    "⬇️ Download HTML Report",
                    data=html_content,
                    file_name=f"AlphaForge_{report.ticker}_{datetime.datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            elif report_format == "CSV Data Export":
                # Bundle key data as CSV
                if report.dcf_result:
                    proj_df = pd.DataFrame([p.to_dict() for p in report.dcf_result.projections])
                    csv = proj_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv,
                        file_name=f"AlphaForge_{report.ticker}_projections.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

with col_preview:
    section_divider("Report Preview")
    
    if dcf and info:
        st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d4a;border-radius:12px;padding:28px;">
                <div style="font-size:11px;color:#f0b429;font-weight:700;letter-spacing:2px;margin-bottom:16px;">
                    VALUATION RESEARCH REPORT
                </div>
                <h2 style="color:#e8edf8;font-size:24px;font-weight:800;margin-bottom:8px;">{report.company_name}</h2>
                <div style="color:#8b9cc8;font-size:13px;margin-bottom:24px;">
                    {report.ticker} · {info.sector or 'N/A'} · AI-Enhanced DCF
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
                    <div>
                        <div style="color:#4a5a7e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Current Price</div>
                        <div style="color:#e8edf8;font-size:20px;font-weight:700;">{format_currency(dcf.current_price)}</div>
                    </div>
                    <div>
                        <div style="color:#4a5a7e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Intrinsic Value</div>
                        <div style="color:#1a6cf5;font-size:20px;font-weight:700;">{format_currency(dcf.intrinsic_value_per_share)}</div>
                    </div>
                    <div>
                        <div style="color:#4a5a7e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Margin of Safety</div>
                        <div style="color:#e8edf8;font-size:20px;font-weight:700;">{format_percentage(dcf.margin_of_safety)}</div>
                    </div>
                    <div>
                        <div style="color:#4a5a7e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Recommendation</div>
                        <div style="font-size:18px;font-weight:800;">{dcf.recommendation}</div>
                    </div>
                </div>
                {"<div style='background:rgba(26,108,245,0.1);border-radius:8px;padding:12px;color:#8b9cc8;font-size:12px;'>" + f"Monte Carlo: {report.monte_carlo.n_simulations:,} sims · P(Undervalued) = {report.monte_carlo.prob_undervalued:.1%} · Median IV = {format_currency(report.monte_carlo.median_value)}" + "</div>" if report.monte_carlo else ""}
            </div>
        """, unsafe_allow_html=True)
