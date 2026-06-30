"""
Reusable UI components: metric cards, traffic lights, badges.
Renders HTML via st.markdown with unsafe_allow_html=True.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src.utils.helpers import format_currency, format_percentage


def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
    icon: str = "",
    subtitle: Optional[str] = None,
) -> None:
    """Render a styled KPI metric card."""
    delta_color = (
        "#00c851"
        if delta_positive
        else "#ff4444" if delta_positive is not None and not delta_positive else "#8b9cc8"
    )
    delta_html = (
        f'<div class="metric-delta" style="color:{delta_color};">'
        f'{"▲" if delta_positive else "▼" if delta_positive is not None else "●"} {delta}'
        f"</div>"
        if delta
        else ""
    )
    subtitle_html = (
        f'<div style="font-size:11px;color:#4a5a7e;margin-top:4px;">{subtitle}</div>'
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div class="metric-card animate-fade-in">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def traffic_light(label: str, color: str = "yellow", value: str = "") -> None:
    """Render a traffic light indicator."""
    valid_colors = {"green", "red", "yellow"}
    if color not in valid_colors:
        color = "yellow"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;padding:8px 0;">
            <span class="traffic-light {color}"></span>
            <span style="color:#e8edf8;font-size:14px;font-weight:500;">{label}</span>
            {f'<span style="margin-left:auto;color:#8b9cc8;font-size:13px;">{value}</span>' if value else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_badge(recommendation: str) -> None:
    """Render a large recommendation badge."""
    color_map = {
        "STRONG BUY": ("#00c851", "rgba(0,200,81,0.12)"),
        "BUY": ("#00c851", "rgba(0,200,81,0.08)"),
        "HOLD": ("#f0b429", "rgba(240,180,41,0.12)"),
        "SELL": ("#ff4444", "rgba(255,68,68,0.08)"),
        "STRONG SELL": ("#ff4444", "rgba(255,68,68,0.12)"),
    }
    color, bg = color_map.get(recommendation.upper(), ("#8b9cc8", "rgba(139,156,200,0.1)"))

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:{bg};
            color:{color};
            border:2px solid {color};
            border-radius:12px;
            padding:12px 32px;
            font-size:22px;
            font-weight:900;
            letter-spacing:2px;
            text-align:center;
            width:100%;
        ">{recommendation}</div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, level: str = "info") -> None:
    """Render a colored info/warning/success box."""
    colors = {
        "info": ("#00bcd4", "rgba(0,188,212,0.08)", "ℹ"),
        "warning": ("#f0b429", "rgba(240,180,41,0.08)", "⚠"),
        "success": ("#00c851", "rgba(0,200,81,0.08)", "✓"),
        "danger": ("#ff4444", "rgba(255,68,68,0.08)", "✕"),
    }
    c, bg, icon = colors.get(level, colors["info"])

    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:3px solid {c};
            border-radius:6px;
            padding:12px 16px;
            color:{c};
            font-size:13px;
            margin:8px 0;
        ">{icon} {text}</div>
        """,
        unsafe_allow_html=True,
    )


def section_divider(title: str, badge: Optional[str] = None) -> None:
    """Render a styled section header with optional badge."""
    badge_html = (
        f'<span style="background:rgba(26,108,245,0.15);color:#1a6cf5;'
        f'border-radius:20px;padding:3px 10px;font-size:11px;font-weight:600;margin-left:12px;">'
        f"{badge}</span>"
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div class="section-header" style="display:flex;align-items:center;
             border-bottom:1px solid #1e2d4a;padding-bottom:14px;margin-bottom:20px;">
            <h2 style="font-size:18px;font-weight:700;color:#e8edf8;margin:0;">
                {title}
            </h2>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def valuation_summary_banner(
    ticker: str,
    company_name: str,
    current_price: float,
    intrinsic_value: float,
    recommendation: str,
    margin_of_safety: float,
) -> None:
    """Renders the top-of-page executive banner."""
    upside = (intrinsic_value - current_price) / current_price * 100 if current_price > 0 else 0
    mos_pct = margin_of_safety * 100
    rec_color = {
        "STRONG BUY": "#00c851",
        "BUY": "#00c851",
        "HOLD": "#f0b429",
        "SELL": "#ff4444",
        "STRONG SELL": "#ff4444",
    }.get(recommendation.upper(), "#8b9cc8")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f1628 0%, #151d35 100%);
            border: 1px solid #1e2d4a;
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        ">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;
                 background:linear-gradient(90deg,#1a6cf5,#f0b429);"></div>
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
                <div>
                    <div style="font-size:13px;color:#8b9cc8;letter-spacing:1.5px;
                         text-transform:uppercase;margin-bottom:4px;">{ticker}</div>
                    <div style="font-size:26px;font-weight:800;color:#e8edf8;">{company_name}</div>
                </div>
                <div style="display:flex;gap:32px;flex-wrap:wrap;">
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:#8b9cc8;text-transform:uppercase;letter-spacing:1px;">Current Price</div>
                        <div style="font-size:24px;font-weight:700;color:#e8edf8;">{format_currency(current_price)}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:#8b9cc8;text-transform:uppercase;letter-spacing:1px;">Intrinsic Value</div>
                        <div style="font-size:24px;font-weight:700;color:#1a6cf5;">{format_currency(intrinsic_value)}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:#8b9cc8;text-transform:uppercase;letter-spacing:1px;">Upside</div>
                        <div style="font-size:24px;font-weight:700;
                             color:{'#00c851' if upside > 0 else '#ff4444'};">
                             {"+" if upside > 0 else ""}{upside:.1f}%
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px;color:#8b9cc8;text-transform:uppercase;letter-spacing:1px;">Verdict</div>
                        <div style="font-size:20px;font-weight:800;color:{rec_color};">{recommendation}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bar(label: str, probability: float, color: str = "#1a6cf5") -> None:
    """Render an animated probability bar."""
    pct = probability * 100
    st.markdown(
        f"""
        <div style="margin:10px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#8b9cc8;font-size:13px;">{label}</span>
                <span style="color:#e8edf8;font-size:13px;font-weight:600;">{pct:.1f}%</span>
            </div>
            <div style="background:#1e2d4a;border-radius:4px;height:8px;overflow:hidden;">
                <div style="
                    background:linear-gradient(90deg,{color},{color}88);
                    width:{pct}%;height:100%;border-radius:4px;
                    transition:width 0.8s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "metric_card",
    "traffic_light",
    "recommendation_badge",
    "info_box",
    "section_divider",
    "valuation_summary_banner",
    "probability_bar",
]
