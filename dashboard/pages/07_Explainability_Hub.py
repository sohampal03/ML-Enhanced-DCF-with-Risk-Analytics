"""Page 7: Explainability Hub — SHAP & natural language explanations"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import info_box, section_divider
from dashboard.components.charts import COLORS

st.set_page_config(page_title="Explainability Hub | AlphaForge", layout="wide", page_icon="🔍")
CSS_FILE = Path(__file__).parent.parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "report" not in st.session_state or not st.session_state.report:
    st.warning("⚠️ Please run an analysis first.")
    st.stop()

report = st.session_state.report
dcf = report.dcf_result
classification = report.classification
rf = report.revenue_forecast

st.markdown(
    """
    <h1 style="font-size:32px;font-weight:900;color:#e8edf8;margin-bottom:8px;">🔍 Explainability Hub</h1>
    <div style="color:#8b9cc8;margin-bottom:28px;">
        Understanding why the model predicted what it did — SHAP values, feature importances, and natural language insights
    </div>
""",
    unsafe_allow_html=True,
)

# ── Valuation Classification Explanation ─────────────────────────────────────
if classification:
    section_divider("Valuation Classification", badge=classification.predicted_label)

    col_class, col_probs = st.columns([1, 1])
    with col_class:
        rec_color = classification.predicted_color
        st.markdown(
            f"""
            <div style="background:rgba(26,108,245,0.05);border:1px solid #1e2d4a;
                 border-top:4px solid {rec_color};border-radius:12px;padding:24px;text-align:center;">
                <div style="font-size:36px;margin-bottom:12px;">{"🟢" if "Under" in classification.predicted_label else "🔴" if "Over" in classification.predicted_label else "🟡"}</div>
                <div style="color:{rec_color};font-size:24px;font-weight:800;">{classification.predicted_label}</div>
                <div style="color:#8b9cc8;font-size:13px;margin-top:8px;">Confidence: {classification.confidence:.1%}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        for note in classification.notes:
            info_box(note, "info")

    with col_probs:
        # Probability pie chart
        probs = classification.probabilities
        fig = go.Figure(
            go.Pie(
                labels=list(probs.keys()),
                values=list(probs.values()),
                hole=0.6,
                marker=dict(colors=[COLORS["bull"], COLORS["gold"], COLORS["bear"]]),
                texttemplate="%{label}<br>%{percent}",
                textfont=dict(color="white"),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            showlegend=False,
            annotations=[
                dict(
                    text="P(class)",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=13, color="#8b9cc8"),
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance bar
    if classification.feature_importances:
        st.markdown("---")
        section_divider("Signal Attribution — What Drove the Classification")
        fi = pd.DataFrame(
            list(classification.feature_importances.items()), columns=["Signal", "Strength"]
        )
        fi = fi.sort_values("Strength", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=fi["Strength"],
                y=fi["Signal"],
                orientation="h",
                marker=dict(color=COLORS["primary"], opacity=0.8),
                text=[f"{v:.2f}" for v in fi["Strength"]],
                textposition="outside",
                textfont=dict(color=COLORS["primary"]),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            showlegend=False,
            xaxis_title="Signal Strength",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── ML Model Feature Importance ───────────────────────────────────────────────
if rf and rf.feature_importances:
    st.markdown("---")
    section_divider("Revenue Forecast — Feature Importances", badge=rf.best_model_name)

    fi_items = sorted(rf.feature_importances.items(), key=lambda x: x[1], reverse=True)[:12]
    features, values = zip(*fi_items) if fi_items else ([], [])

    col_fi, col_nl = st.columns([1, 1])
    with col_fi:
        fig = go.Figure(
            go.Bar(
                x=list(values),
                y=list(features),
                orientation="h",
                marker=dict(
                    color=list(values),
                    colorscale=[[0, COLORS["primary"]], [1, COLORS["gold"]]],
                    showscale=False,
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            xaxis_title="Importance Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_nl:
        section_divider("Natural Language Explanation")
        if fi_items:
            top = fi_items[0]
            st.markdown(
                """
                <div style="background:#0f1628;border:1px solid #1e2d4a;border-radius:12px;padding:20px;">
                    <div style="color:#1a6cf5;font-size:13px;font-weight:600;text-transform:uppercase;
                         letter-spacing:1px;margin-bottom:16px;">Why this forecast?</div>
            """,
                unsafe_allow_html=True,
            )

            for i, (feat, imp) in enumerate(fi_items[:5]):
                pct = imp / sum(v for _, v in fi_items) * 100
                bar_color = COLORS["bull"] if imp > 0 else COLORS["bear"]
                st.markdown(
                    f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="color:#e8edf8;font-size:13px;">{feat.replace('_', ' ').title()}</span>
                            <span style="color:{COLORS['gold']};font-size:13px;font-weight:600;">+{pct:.1f}%</span>
                        </div>
                        <div style="background:#1e2d4a;border-radius:3px;height:6px;">
                            <div style="background:{bar_color};width:{min(pct*3, 100)}%;height:100%;border-radius:3px;"></div>
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

# ── DCF Assumption Attribution ────────────────────────────────────────────────
if dcf and report.monte_carlo and report.monte_carlo.tornado_data:
    st.markdown("---")
    section_divider("DCF Assumption Impact (SHAP-style)")

    tornado = report.monte_carlo.tornado_data
    items = sorted(tornado.items(), key=lambda x: x[1]["impact"], reverse=True)

    for name, data_item in items:
        impact = data_item["impact"]
        corr = data_item["correlation"]
        direction = "positive" if corr > 0 else "negative"
        bar_color = COLORS["bull"] if corr > 0 else COLORS["bear"]

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin:10px 0;">
                <div style="width:160px;text-align:right;color:#8b9cc8;font-size:13px;">
                    {data_item['label']}
                </div>
                <div style="flex:1;background:#1e2d4a;border-radius:4px;height:8px;position:relative;">
                    <div style="background:{bar_color};
                         width:{min(abs(corr) * 100, 100)}%;height:100%;
                         border-radius:4px;opacity:0.8;">
                    </div>
                </div>
                <div style="width:80px;color:{bar_color};font-size:12px;font-weight:600;">
                    {corr:+.2f} corr
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
