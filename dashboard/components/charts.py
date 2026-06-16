"""
Plotly Chart Factory — 20+ chart types for the AlphaForge dashboard.

All charts use the Bloomberg-inspired dark theme with consistent styling.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Dark Theme Template ──────────────────────────────────────────────────────

DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#e8edf8", size=12),
        title=dict(font=dict(size=16, color="#e8edf8", family="Inter")),
        xaxis=dict(
            gridcolor="rgba(30,45,74,0.8)",
            linecolor="#1e2d4a",
            tickcolor="#1e2d4a",
            tickfont=dict(color="#8b9cc8"),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(30,45,74,0.8)",
            linecolor="#1e2d4a",
            tickcolor="#1e2d4a",
            tickfont=dict(color="#8b9cc8"),
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(15,22,40,0.9)",
            bordercolor="#1e2d4a",
            font=dict(color="#8b9cc8"),
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#1e2d4a",
            font=dict(color="#e8edf8"),
        ),
        colorway=["#1a6cf5", "#f0b429", "#00c851", "#ff4444", "#00bcd4", "#9c27b0", "#ff9800"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

# Color palette
COLORS = {
    "primary": "#1a6cf5",
    "gold": "#f0b429",
    "bull": "#00c851",
    "bear": "#ff4444",
    "info": "#00bcd4",
    "purple": "#9c27b0",
    "orange": "#ff9800",
    "gradient_blue": ["#0d47a1", "#1565c0", "#1976d2", "#1a6cf5", "#42a5f5", "#90caf9"],
}


def _fig_defaults(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply standard styling to any figure."""
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True,
    )
    return fig


# ── Price Charts ─────────────────────────────────────────────────────────────


def candlestick_chart(
    price_df: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
) -> go.Figure:
    """Candlestick chart with volume bars."""
    if show_volume and "Volume" in price_df.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()
        show_volume = False

    candle = go.Candlestick(
        x=price_df.index,
        open=price_df.get("Open", price_df.get("open")),
        high=price_df.get("High", price_df.get("high")),
        low=price_df.get("Low", price_df.get("low")),
        close=price_df.get("Close", price_df.get("close")),
        name=ticker,
        increasing=dict(fillcolor=COLORS["bull"], line=dict(color=COLORS["bull"])),
        decreasing=dict(fillcolor=COLORS["bear"], line=dict(color=COLORS["bear"])),
    )

    if show_volume:
        vol_colors = [
            COLORS["bull"] if row.get("Close", 0) >= row.get("Open", 0) else COLORS["bear"]
            for _, row in price_df.iterrows()
        ]
        vol = go.Bar(
            x=price_df.index,
            y=price_df.get("Volume", price_df.get("volume", pd.Series())),
            marker_color=vol_colors,
            marker_opacity=0.5,
            name="Volume",
            showlegend=False,
        )
        fig.add_trace(candle, row=1, col=1)
        fig.add_trace(vol, row=2, col=1)
    else:
        fig.add_trace(candle)

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{ticker} — Price History",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    return fig


# ── Revenue / Forecast Charts ─────────────────────────────────────────────────


def revenue_forecast_chart(
    historical_years: list[int],
    historical_values: list[float],
    forecast_years: list[int],
    forecast_values: list[float],
    lower_ci: list[float],
    upper_ci: list[float],
    ticker: str,
    label: str = "Revenue",
) -> go.Figure:
    """Area chart with historical + forecast + confidence band."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_years,
        y=historical_values,
        mode="lines+markers",
        name="Historical",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=6, color=COLORS["primary"]),
    ))

    # Confidence band (add upper first, then lower with fill)
    fig.add_trace(go.Scatter(
        x=forecast_years + forecast_years[::-1],
        y=upper_ci + lower_ci[::-1],
        fill="toself",
        fillcolor="rgba(26,108,245,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast",
        line=dict(color=COLORS["gold"], width=2.5, dash="dash"),
        marker=dict(size=7, color=COLORS["gold"], symbol="diamond"),
    ))

    # Vertical separator
    split_year = historical_years[-1] if historical_years else forecast_years[0] - 1
    fig.add_vline(
        x=split_year,
        line_dash="dot",
        line_color="#4a5a7e",
        annotation_text="Forecast →",
        annotation_font=dict(color="#8b9cc8", size=11),
    )

    _fig_defaults(fig, height=420)
    fig.update_layout(
        title=f"{ticker} — {label} Forecast",
        yaxis_title=label,
        xaxis_title="Fiscal Year",
    )
    return fig


# ── DCF Waterfall ─────────────────────────────────────────────────────────────


def dcf_waterfall_chart(
    pv_fcff: float,
    pv_terminal: float,
    enterprise_value: float,
    total_debt: float,
    cash: float,
    equity_value: float,
    intrinsic_per_share: float,
) -> go.Figure:
    """Bloomberg-style DCF waterfall chart."""
    categories = [
        "PV of FCFFs",
        "PV Terminal Value",
        "Enterprise Value",
        "Less: Debt",
        "Add: Cash",
        "Equity Value",
    ]

    values = [pv_fcff, pv_terminal, 0, -total_debt, cash, 0]
    # Equity value is the sum
    measure = ["relative", "relative", "total", "relative", "relative", "total"]

    # Colors per bar
    bar_colors = [COLORS["primary"], COLORS["gold"], "#2a5a9c", COLORS["bear"], COLORS["bull"], "#1557cc"]

    fig = go.Figure(go.Waterfall(
        name="DCF Breakdown",
        orientation="v",
        measure=measure,
        x=categories,
        y=values,
        connector=dict(line=dict(color="#1e2d4a", width=1.5)),
        increasing=dict(marker=dict(color=COLORS["bull"])),
        decreasing=dict(marker=dict(color=COLORS["bear"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
        text=[
            f"${pv_fcff / 1e9:.1f}B",
            f"${pv_terminal / 1e9:.1f}B",
            f"${enterprise_value / 1e9:.1f}B",
            f"-${total_debt / 1e9:.1f}B",
            f"+${cash / 1e9:.1f}B",
            f"${equity_value / 1e9:.1f}B",
        ],
        textposition="outside",
        textfont=dict(color="#e8edf8"),
    ))

    _fig_defaults(fig, height=450)
    fig.update_layout(
        title="DCF Value Bridge",
        yaxis_title="Value ($B)",
        showlegend=False,
    )
    return fig


# ── Sensitivity Heatmap ───────────────────────────────────────────────────────


def sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    current_price: float,
    ticker: str,
) -> go.Figure:
    """WACC × Terminal Growth Rate sensitivity heatmap."""
    z = sensitivity_df.values.astype(float)

    # Color scale: red (below current) → white → green (above current)
    colorscale = [
        [0.0, COLORS["bear"]],
        [0.3, "#8b0000"],
        [0.5, "#1a2540"],
        [0.7, "#003300"],
        [1.0, COLORS["bull"]],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=sensitivity_df.columns.tolist(),
        y=sensitivity_df.index.tolist(),
        colorscale=colorscale,
        zmid=current_price,
        text=[[f"${v:.0f}" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="WACC: %{y}<br>TGR: %{x}<br>Intrinsic: $%{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Intrinsic Value ($)", side="right"),
            tickfont=dict(color="#8b9cc8"),
            bgcolor="rgba(15,22,40,0.9)",
            bordercolor="#1e2d4a",
        ),
    ))

    _fig_defaults(fig, height=380)
    fig.update_layout(
        title=f"{ticker} — Intrinsic Value Sensitivity (WACC × Terminal Growth)",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
    )
    return fig


# ── Monte Carlo Distribution ─────────────────────────────────────────────────


def monte_carlo_histogram(
    intrinsic_values: np.ndarray,
    current_price: float,
    ticker: str,
    p25: float,
    p50: float,
    p75: float,
) -> go.Figure:
    """Monte Carlo intrinsic value distribution with key percentile bands."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=intrinsic_values,
        nbinsx=80,
        name="Simulated Values",
        marker=dict(
            color=COLORS["primary"],
            opacity=0.7,
            line=dict(color=COLORS["primary"], width=0.2),
        ),
    ))

    # Current price line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color=COLORS["bear"],
        line_width=2.5,
        annotation_text=f"Current: ${current_price:.0f}",
        annotation_font=dict(color=COLORS["bear"], size=11),
        annotation_position="top right",
    )

    # Percentile lines
    for pct_val, label, color in [
        (p25, "P25", "#8b9cc8"),
        (p50, "Median", COLORS["gold"]),
        (p75, "P75", COLORS["bull"]),
    ]:
        fig.add_vline(
            x=pct_val,
            line_dash="dot",
            line_color=color,
            line_width=1.5,
            annotation_text=f"{label}: ${pct_val:.0f}",
            annotation_font=dict(color=color, size=10),
        )

    _fig_defaults(fig, height=420)
    fig.update_layout(
        title=f"{ticker} — Monte Carlo: Intrinsic Value Distribution",
        xaxis_title="Intrinsic Value per Share ($)",
        yaxis_title="Frequency",
        bargap=0.05,
        showlegend=False,
    )
    return fig


def tornado_chart(tornado_data: dict, ticker: str) -> go.Figure:
    """Tornado sensitivity chart — impact of each parameter on IV."""
    if not tornado_data:
        return go.Figure()

    # Sort by impact (descending)
    sorted_params = sorted(tornado_data.items(), key=lambda x: x[1]["impact"], reverse=True)[:8]

    labels = [d["label"] for _, d in sorted_params]
    low_vals = [d["iv_low_param"] for _, d in sorted_params]
    high_vals = [d["iv_high_param"] for _, d in sorted_params]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="High (90th pct)",
        y=labels,
        x=[h - l for h, l in zip(high_vals, low_vals)],
        orientation="h",
        base=low_vals,
        marker_color=COLORS["bull"],
        marker_opacity=0.8,
        text=[f"${h:,.0f}" for h in high_vals],
        textposition="outside",
        textfont=dict(color=COLORS["bull"]),
    ))

    fig.add_trace(go.Bar(
        name="Low (10th pct)",
        y=labels,
        x=[l - l for l in low_vals],  # zero width (base already set)
        orientation="h",
        base=[0] * len(low_vals),
        marker_color=COLORS["bear"],
        text=[f"${l:,.0f}" for l in low_vals],
        textposition="inside",
        textfont=dict(color="white"),
        showlegend=False,
    ))

    _fig_defaults(fig, height=400)
    fig.update_layout(
        title=f"{ticker} — Parameter Sensitivity (Tornado Chart)",
        xaxis_title="Impact on Intrinsic Value ($)",
        barmode="overlay",
        showlegend=True,
    )
    return fig


# ── Radar Chart (Comps) ───────────────────────────────────────────────────────


def peer_radar_chart(
    categories: list[str],
    target_values: list[float],
    peer_values: list[float],
    target_name: str,
    peer_label: str = "Peer Median",
) -> go.Figure:
    """Radar chart comparing target vs peer median across metrics."""
    cats = categories + [categories[0]]  # Close the polygon
    t_vals = target_values + [target_values[0]]
    p_vals = peer_values + [peer_values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=p_vals, theta=cats, fill="toself",
        name=peer_label,
        fillcolor="rgba(240,180,41,0.1)",
        line=dict(color=COLORS["gold"], width=2),
    ))

    fig.add_trace(go.Scatterpolar(
        r=t_vals, theta=cats, fill="toself",
        name=target_name,
        fillcolor="rgba(26,108,245,0.15)",
        line=dict(color=COLORS["primary"], width=2.5),
    ))

    _fig_defaults(fig, height=380)
    fig.update_layout(
        title=f"{target_name} vs Peer Median",
        polar=dict(
            bgcolor="rgba(10,14,26,0)",
            radialaxis=dict(visible=True, gridcolor="#1e2d4a", tickfont=dict(color="#8b9cc8")),
            angularaxis=dict(gridcolor="#1e2d4a", tickfont=dict(color="#e8edf8")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


# ── Gauge / Confidence Chart ──────────────────────────────────────────────────


def valuation_gauge(
    current_price: float,
    intrinsic_value: float,
    p5: float,
    p95: float,
    ticker: str,
) -> go.Figure:
    """Gauge showing where current price sits in the valuation range."""
    # Normalize
    range_min = min(current_price * 0.3, p5 * 0.8)
    range_max = max(current_price * 2.0, p95 * 1.2)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_price,
        delta=dict(
            reference=intrinsic_value,
            valueformat="$,.2f",
            increasing=dict(color=COLORS["bear"]),   # Higher price = overvalued
            decreasing=dict(color=COLORS["bull"]),   # Lower price = undervalued
        ),
        gauge=dict(
            axis=dict(
                range=[range_min, range_max],
                tickfont=dict(color="#8b9cc8"),
                tickformat="$,.0f",
            ),
            bar=dict(color=COLORS["primary"], thickness=0.3),
            bgcolor="rgba(15,22,40,0.9)",
            bordercolor="#1e2d4a",
            steps=[
                dict(range=[range_min, p5], color="rgba(0,200,81,0.25)"),
                dict(range=[p5, intrinsic_value * 0.85], color="rgba(0,200,81,0.12)"),
                dict(range=[intrinsic_value * 0.85, intrinsic_value * 1.15], color="rgba(240,180,41,0.15)"),
                dict(range=[intrinsic_value * 1.15, p95], color="rgba(255,68,68,0.12)"),
                dict(range=[p95, range_max], color="rgba(255,68,68,0.25)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["gold"], width=4),
                thickness=0.85,
                value=intrinsic_value,
            ),
        ),
        title=dict(text=f"{ticker} — Current Price vs Intrinsic Value", font=dict(color="#e8edf8")),
        number=dict(prefix="$", valueformat=",.2f", font=dict(color="#e8edf8")),
    ))

    _fig_defaults(fig, height=350)
    fig.update_layout(
        paper_bgcolor="rgba(10,14,26,0)",
        font=dict(color="#e8edf8"),
    )
    return fig


# ── Financial Trend Charts ────────────────────────────────────────────────────


def financial_bar_chart(
    years: list,
    values: list[float],
    label: str,
    ticker: str,
    show_growth: bool = True,
    color: str = COLORS["primary"],
) -> go.Figure:
    """Bar chart for financial metrics with growth rate overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=years,
        y=values,
        name=label,
        marker=dict(
            color=color,
            opacity=0.85,
            line=dict(color=color, width=0.5),
        ),
        text=[f"${v / 1e9:.1f}B" if abs(v) > 1e8 else f"${v / 1e6:.0f}M" for v in values],
        textposition="auto",
        textfont=dict(color="white"),
    ), secondary_y=False)

    if show_growth and len(values) > 1:
        growth_rates = [0] + [
            (values[i] - values[i - 1]) / abs(values[i - 1]) * 100
            if values[i - 1] != 0 else 0
            for i in range(1, len(values))
        ]
        fig.add_trace(go.Scatter(
            x=years,
            y=growth_rates,
            name="YoY Growth %",
            mode="lines+markers",
            line=dict(color=COLORS["gold"], width=2, dash="dot"),
            marker=dict(size=6, color=COLORS["gold"]),
        ), secondary_y=True)

        fig.update_yaxes(title_text="Growth Rate (%)", secondary_y=True,
                         ticksuffix="%", tickfont=dict(color="#8b9cc8"))

    _fig_defaults(fig, height=380)
    fig.update_layout(
        title=f"{ticker} — {label}",
        xaxis_title="Fiscal Year",
        yaxis_title=label,
        bargap=0.3,
    )
    return fig


def margin_trend_chart(
    years: list,
    gross_margins: list[float],
    ebit_margins: list[float],
    net_margins: list[float],
    ticker: str,
) -> go.Figure:
    """Multi-line margin trend chart."""
    fig = go.Figure()

    for margins, name, color in [
        (gross_margins, "Gross Margin", COLORS["primary"]),
        (ebit_margins, "EBIT Margin", COLORS["gold"]),
        (net_margins, "Net Margin", COLORS["bull"]),
    ]:
        clean = [m * 100 if m and abs(m) <= 1 else m for m in margins]
        fig.add_trace(go.Scatter(
            x=years,
            y=clean,
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.04,)}",
        ))

    _fig_defaults(fig, height=380)
    fig.update_layout(
        title=f"{ticker} — Margin Trends",
        yaxis=dict(ticksuffix="%"),
        yaxis_title="Margin (%)",
        xaxis_title="Fiscal Year",
    )
    return fig


def violin_plot(values: np.ndarray, current_price: float, ticker: str) -> go.Figure:
    """Violin + box plot for Monte Carlo distribution."""
    fig = go.Figure()

    fig.add_trace(go.Violin(
        y=values,
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(26,108,245,0.15)",
        line_color=COLORS["primary"],
        name="Intrinsic Value Distribution",
    ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color=COLORS["bear"],
        line_width=2,
        annotation_text=f"Current: ${current_price:.0f}",
        annotation_font=dict(color=COLORS["bear"]),
    )

    _fig_defaults(fig, height=400)
    fig.update_layout(
        title=f"{ticker} — Intrinsic Value Distribution (Violin)",
        yaxis_title="Intrinsic Value ($)",
        showlegend=False,
    )
    return fig


__all__ = [
    "candlestick_chart",
    "revenue_forecast_chart",
    "dcf_waterfall_chart",
    "sensitivity_heatmap",
    "monte_carlo_histogram",
    "tornado_chart",
    "peer_radar_chart",
    "valuation_gauge",
    "financial_bar_chart",
    "margin_trend_chart",
    "violin_plot",
    "COLORS",
    "DARK_TEMPLATE",
]
