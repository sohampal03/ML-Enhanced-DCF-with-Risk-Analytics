"""
DCF (Discounted Cash Flow) Engine.

Implements a two-stage FCFF-based DCF model:
  Stage 1 — Explicit forecast period (5–10 years)
  Stage 2 — Terminal value (Gordon Growth Model)

Enterprise Value = PV(FCFF forecast) + PV(Terminal Value)
Equity Value     = Enterprise Value − Total Debt + Cash
Intrinsic Value  = Equity Value / Shares Outstanding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.schemas.financial_schemas import FinancialData
from src.finance.engines.fcff_engine import FCFFEngine, FCFFResult
from src.finance.engines.wacc_engine import WACCBreakdown, WACCEngine
from src.utils.helpers import format_currency, format_percentage


@dataclass
class DCFYearProjection:
    """Single-year projection in the DCF forecast."""

    year: int
    projected_revenue: float
    revenue_growth_rate: float
    ebit_margin: float
    ebit: float
    tax_rate: float
    nopat: float
    da_rate: float          # D&A as % of revenue
    da: float
    capex_rate: float       # CAPEX as % of revenue
    capex: float
    nwc_rate: float         # NWC as % of revenue
    delta_nwc: float
    fcff: float
    discount_factor: float
    pv_fcff: float

    def to_dict(self) -> dict:
        return {
            "Year": f"Y{self.year}",
            "Revenue": self.projected_revenue,
            "Revenue Growth": self.revenue_growth_rate,
            "EBIT Margin": self.ebit_margin,
            "EBIT": self.ebit,
            "NOPAT": self.nopat,
            "D&A": self.da,
            "CAPEX": self.capex,
            "ΔNWC": self.delta_nwc,
            "FCFF": self.fcff,
            "Discount Factor": self.discount_factor,
            "PV of FCFF": self.pv_fcff,
        }


@dataclass
class DCFResult:
    """Complete DCF valuation output."""

    ticker: str
    company_name: str
    current_price: float

    # Projections
    projections: list[DCFYearProjection] = field(default_factory=list)

    # Assumptions
    wacc: float = 0.10
    terminal_growth_rate: float = 0.025
    forecast_years: int = 10
    base_revenue: float = 0.0
    revenue_growth_rates: list[float] = field(default_factory=list)
    ebit_margin_forecast: float = 0.15
    tax_rate: float = 0.21

    # Value components
    pv_fcff_sum: float = 0.0
    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0
    enterprise_value: float = 0.0
    total_debt: float = 0.0
    cash: float = 0.0
    equity_value: float = 0.0
    shares_outstanding: float = 0.0
    intrinsic_value_per_share: float = 0.0

    # Margin of Safety
    margin_of_safety: float = 0.0
    upside_potential: float = 0.0   # (intrinsic - price) / price

    # Recommendation
    recommendation: str = "HOLD"
    recommendation_color: str = "gray"
    confidence_score: float = 0.5

    # Sensitivity matrix
    sensitivity_matrix: Optional[pd.DataFrame] = None

    def to_summary_dict(self) -> dict:
        return {
            "Metric": [
                "Current Market Price",
                "Intrinsic Value / Share",
                "Margin of Safety",
                "Upside / Downside",
                "Enterprise Value",
                "Equity Value",
                "Sum PV FCFF",
                "Terminal Value",
                "PV Terminal Value",
                "Recommendation",
            ],
            "Value": [
                format_currency(self.current_price),
                format_currency(self.intrinsic_value_per_share),
                format_percentage(self.margin_of_safety),
                format_percentage(self.upside_potential),
                format_currency(self.enterprise_value),
                format_currency(self.equity_value),
                format_currency(self.pv_fcff_sum),
                format_currency(self.terminal_value),
                format_currency(self.pv_terminal_value),
                self.recommendation,
            ],
        }


class DCFEngine:
    """
    Two-stage Discounted Cash Flow (DCF) valuation engine.

    Implements the standard FCFF-based DCF with:
      - Explicit forecast of revenues, margins, capex, NWC
      - Gordon Growth Model terminal value
      - Intrinsic value per share calculation
      - Sensitivity analysis matrix (WACC × Terminal Growth)
      - Margin of Safety calculation

    Can be used in two modes:
      1. Auto mode: derives assumptions from historical data
      2. Manual mode: accepts user-supplied assumptions (DCF Lab)
    """

    MOS_THRESHOLDS = {
        "STRONG BUY": 0.30,   # >30% upside
        "BUY": 0.10,           # 10–30% upside
        "HOLD": -0.10,         # ±10%
        "SELL": -0.30,         # 10–30% downside
        "STRONG SELL": -1.0,   # >30% downside
    }

    def compute(
        self,
        data: FinancialData,
        fcff_result: Optional[FCFFResult] = None,
        wacc_result: Optional[WACCBreakdown] = None,
        # Manual overrides (for DCF Lab)
        revenue_growth_rates: Optional[list[float]] = None,
        ebit_margin: Optional[float] = None,
        wacc_override: Optional[float] = None,
        terminal_growth_rate: Optional[float] = None,
        forecast_years: int = 10,
        tax_rate_override: Optional[float] = None,
        capex_pct_revenue: Optional[float] = None,
        da_pct_revenue: Optional[float] = None,
        nwc_pct_revenue: Optional[float] = None,
        build_sensitivity: bool = True,
    ) -> DCFResult:
        """
        Perform complete DCF valuation.

        Args:
            data: Company financial data.
            fcff_result: Pre-computed FCFF history (computed if None).
            wacc_result: Pre-computed WACC (computed if None).
            revenue_growth_rates: List of per-year revenue growth rates.
            ebit_margin: Target EBIT margin for forecast period.
            wacc_override: Override WACC (e.g., from slider).
            terminal_growth_rate: Terminal/perpetuity growth rate.
            forecast_years: Number of explicit forecast years.
            tax_rate_override: Override effective tax rate.
            capex_pct_revenue: CAPEX as % of revenue.
            da_pct_revenue: D&A as % of revenue.
            nwc_pct_revenue: NWC as % of revenue.
            build_sensitivity: Whether to compute WACC × TGR sensitivity matrix.

        Returns:
            DCFResult with full valuation output.
        """
        ticker = data.ticker
        logger.info(f"[DCFEngine] Valuing {ticker} ({forecast_years}Y explicit)")

        # ── Step 1: Compute sub-engines if not provided ────────────────────
        if fcff_result is None:
            fcff_result = FCFFEngine().compute(data)
        if wacc_result is None:
            wacc_result = WACCEngine().compute(data)

        # ── Step 2: Resolve all assumptions ───────────────────────────────
        wacc = wacc_override or wacc_result.wacc
        tgr = terminal_growth_rate or 0.025
        tax_rate = tax_rate_override or wacc_result.tax_rate
        base_revenue = self._get_base_revenue(data)

        # Derive assumptions from history if not overridden
        hist_growth, hist_margin, hist_da, hist_capex, hist_nwc = self._extract_historical_ratios(
            data, fcff_result
        )

        growth_rates = revenue_growth_rates or self._build_growth_schedule(hist_growth, forecast_years)
        margin = ebit_margin or hist_margin
        da_pct = da_pct_revenue or hist_da
        capex_pct = capex_pct_revenue or hist_capex
        nwc_pct = nwc_pct_revenue or hist_nwc

        # ── Step 3: Build year-by-year projections ─────────────────────────
        projections: list[DCFYearProjection] = []
        prev_revenue = base_revenue

        for i in range(forecast_years):
            yr_num = i + 1
            g = growth_rates[i] if i < len(growth_rates) else growth_rates[-1]

            revenue = prev_revenue * (1 + g)
            ebit = revenue * margin
            nopat = ebit * (1 - tax_rate)
            da = revenue * da_pct
            capex = revenue * capex_pct
            nwc = revenue * nwc_pct
            prev_nwc = prev_revenue * nwc_pct
            delta_nwc = nwc - prev_nwc
            fcff_yr = nopat + da - delta_nwc - capex

            discount_factor = 1 / (1 + wacc) ** yr_num
            pv_fcff = fcff_yr * discount_factor

            proj = DCFYearProjection(
                year=yr_num,
                projected_revenue=revenue,
                revenue_growth_rate=g,
                ebit_margin=margin,
                ebit=ebit,
                tax_rate=tax_rate,
                nopat=nopat,
                da_rate=da_pct,
                da=da,
                capex_rate=capex_pct,
                capex=capex,
                nwc_rate=nwc_pct,
                delta_nwc=delta_nwc,
                fcff=fcff_yr,
                discount_factor=discount_factor,
                pv_fcff=pv_fcff,
            )
            projections.append(proj)
            prev_revenue = revenue

        # ── Step 4: Terminal Value ─────────────────────────────────────────
        terminal_fcff = projections[-1].fcff * (1 + tgr)
        terminal_value = terminal_fcff / (wacc - tgr) if wacc > tgr else 0.0
        pv_terminal = terminal_value / (1 + wacc) ** forecast_years

        # ── Step 5: Enterprise Value → Equity Value → Per Share ──────────
        pv_fcff_sum = sum(p.pv_fcff for p in projections)
        enterprise_value = pv_fcff_sum + pv_terminal

        total_debt = self._get_total_debt(data)
        cash = self._get_cash(data)
        equity_value = max(0.0, enterprise_value - total_debt + cash)

        shares = data.company_info.shares_outstanding or 1.0
        intrinsic_per_share = equity_value / shares

        # ── Step 6: Margin of Safety & Recommendation ────────────────────
        current_price = data.company_info.current_price or 0.0
        upside = (intrinsic_per_share - current_price) / current_price if current_price > 0 else 0
        mos = (intrinsic_per_share - current_price) / intrinsic_per_share if intrinsic_per_share > 0 else 0

        recommendation, color, confidence = self._make_recommendation(upside, wacc_result)

        # ── Step 7: Sensitivity Matrix ────────────────────────────────────
        sensitivity = None
        if build_sensitivity:
            sensitivity = self._build_sensitivity_matrix(
                projections=projections,
                terminal_fcff=terminal_fcff,
                pv_fcff_sum=pv_fcff_sum,
                total_debt=total_debt,
                cash=cash,
                shares=shares,
                forecast_years=forecast_years,
                base_wacc=wacc,
                base_tgr=tgr,
            )

        result = DCFResult(
            ticker=ticker,
            company_name=data.company_info.name,
            current_price=current_price,
            projections=projections,
            wacc=wacc,
            terminal_growth_rate=tgr,
            forecast_years=forecast_years,
            base_revenue=base_revenue,
            revenue_growth_rates=growth_rates,
            ebit_margin_forecast=margin,
            tax_rate=tax_rate,
            pv_fcff_sum=pv_fcff_sum,
            terminal_value=terminal_value,
            pv_terminal_value=pv_terminal,
            enterprise_value=enterprise_value,
            total_debt=total_debt,
            cash=cash,
            equity_value=equity_value,
            shares_outstanding=shares,
            intrinsic_value_per_share=intrinsic_per_share,
            margin_of_safety=mos,
            upside_potential=upside,
            recommendation=recommendation,
            recommendation_color=color,
            confidence_score=confidence,
            sensitivity_matrix=sensitivity,
        )

        logger.success(
            f"[DCFEngine] {ticker}: Intrinsic = {format_currency(intrinsic_per_share)} | "
            f"Current = {format_currency(current_price)} | "
            f"Upside = {format_percentage(upside)} | {recommendation}"
        )
        return result

    # ── Assumption Derivation ─────────────────────────────────────────────────

    def _extract_historical_ratios(
        self, data: FinancialData, fcff_result: FCFFResult
    ) -> tuple[float, float, float, float, float]:
        """Extract historical averages for growth, margin, D&A, CAPEX, NWC."""
        income_df = data.income_df()
        cashflow_df = data.cashflow_df()

        # Revenue growth (median of last 5 years)
        if "revenue" in income_df.columns and len(income_df) > 1:
            revs = income_df["revenue"].dropna()
            growth = revs.pct_change(-1).dropna()
            hist_growth = float(growth.median()) if not growth.empty else 0.05
        else:
            hist_growth = 0.05

        # EBIT margin (median)
        if "ebit_margin" in income_df.columns:
            margins = income_df["ebit_margin"].dropna()
            hist_margin = float(margins.median()) if not margins.empty else 0.15
        else:
            hist_margin = 0.15

        hist_margin = max(0.01, min(hist_margin, 0.60))  # Clamp

        # D&A as % of revenue
        if "depreciation_amortization" in cashflow_df.columns and "revenue" in income_df.columns:
            merged = pd.merge(
                cashflow_df[["fiscal_year", "depreciation_amortization"]],
                income_df[["fiscal_year", "revenue"]],
                on="fiscal_year",
            )
            merged["da_pct"] = merged["depreciation_amortization"].abs() / merged["revenue"]
            hist_da = float(merged["da_pct"].dropna().median()) if not merged.empty else 0.04
        else:
            hist_da = 0.04

        # CAPEX as % of revenue
        if "capital_expenditures" in cashflow_df.columns and "revenue" in income_df.columns:
            merged = pd.merge(
                cashflow_df[["fiscal_year", "capital_expenditures"]],
                income_df[["fiscal_year", "revenue"]],
                on="fiscal_year",
            )
            merged["capex_pct"] = merged["capital_expenditures"].abs() / merged["revenue"]
            hist_capex = float(merged["capex_pct"].dropna().median()) if not merged.empty else 0.05
        else:
            hist_capex = 0.05

        # NWC as % of revenue (simple approximation)
        hist_nwc = 0.10

        logger.debug(
            f"Historical assumptions — growth: {hist_growth:.1%}, margin: {hist_margin:.1%}, "
            f"D&A: {hist_da:.1%}, CAPEX: {hist_capex:.1%}"
        )
        return hist_growth, hist_margin, hist_da, hist_capex, hist_nwc

    def _build_growth_schedule(self, hist_growth: float, years: int) -> list[float]:
        """
        Build a declining growth schedule from historical to steady-state.

        Starts at historical average, linearly declines to 3% by terminal year.
        """
        start = min(max(hist_growth, 0.02), 0.40)   # Clamp: 2% – 40%
        end = 0.03                                    # 3% steady-state
        return list(np.linspace(start, end, years))

    def _get_base_revenue(self, data: FinancialData) -> float:
        latest = data.latest_income
        return latest.revenue or 0.0 if latest else 0.0

    def _get_total_debt(self, data: FinancialData) -> float:
        balance = data.latest_balance
        if not balance:
            return 0.0
        return balance.total_debt or (balance.long_term_debt or 0) + (balance.short_term_debt or 0)

    def _get_cash(self, data: FinancialData) -> float:
        balance = data.latest_balance
        return (balance.cash_and_equivalents or 0) + (balance.short_term_investments or 0) \
            if balance else 0.0

    def _make_recommendation(
        self, upside: float, wacc_result: WACCBreakdown
    ) -> tuple[str, str, float]:
        """Generate recommendation, color, and confidence score."""
        color_map = {
            "STRONG BUY": "#00C851",
            "BUY": "#00C851",
            "HOLD": "#FFD700",
            "SELL": "#FF4444",
            "STRONG SELL": "#CC0000",
        }

        if upside >= 0.30:
            rec, conf = "STRONG BUY", 0.90
        elif upside >= 0.10:
            rec, conf = "BUY", 0.75
        elif upside >= -0.10:
            rec, conf = "HOLD", 0.60
        elif upside >= -0.30:
            rec, conf = "SELL", 0.75
        else:
            rec, conf = "STRONG SELL", 0.88

        # Discount confidence if beta is very high or WACC was capped
        if wacc_result.beta > 2.0:
            conf -= 0.10

        return rec, color_map[rec], max(0.1, min(conf, 0.95))

    def _build_sensitivity_matrix(
        self,
        projections: list[DCFYearProjection],
        terminal_fcff: float,
        pv_fcff_sum: float,
        total_debt: float,
        cash: float,
        shares: float,
        forecast_years: int,
        base_wacc: float,
        base_tgr: float,
    ) -> pd.DataFrame:
        """
        Build WACC × Terminal Growth Rate sensitivity matrix.

        Returns a DataFrame where rows = WACC scenarios, columns = TGR scenarios.
        """
        wacc_range = np.arange(base_wacc - 0.03, base_wacc + 0.035, 0.01)
        tgr_range = np.arange(base_tgr - 0.015, base_tgr + 0.02, 0.005)

        rows = {}
        for w in wacc_range:
            w = round(w, 4)
            row = {}
            # Recompute PV of FCFF under new WACC
            pv_sum = sum(p.fcff / (1 + w) ** p.year for p in projections)
            for tgr in tgr_range:
                tgr = round(tgr, 4)
                if w <= tgr:
                    row[f"{tgr:.1%}"] = np.nan
                    continue
                tv = terminal_fcff / (w - tgr)
                pv_tv = tv / (1 + w) ** forecast_years
                ev = pv_sum + pv_tv
                eq = max(0, ev - total_debt + cash)
                intrinsic = eq / shares if shares > 0 else 0
                row[f"{tgr:.1%}"] = round(intrinsic, 2)
            rows[f"{w:.1%}"] = row

        return pd.DataFrame(rows).T


__all__ = ["DCFEngine", "DCFResult", "DCFYearProjection"]
