"""
FCFF (Free Cash Flow to Firm) Engine.

Calculates FCFF from financial statements following the standard formula:
    FCFF = EBIT × (1 - Tax Rate) + D&A - ΔNWC - CAPEX
         = NOPAT + D&A - ΔNWC - CAPEX

Also computes historical FCFF trends and growth rates for forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.schemas.financial_schemas import (
    AnnualBalanceSheet,
    AnnualCashFlowStatement,
    AnnualIncomeStatement,
    FinancialData,
)
from src.utils.helpers import format_currency, format_percentage, safe_divide


@dataclass
class FCFFBreakdown:
    """Detailed FCFF calculation for a single year."""

    fiscal_year: int
    revenue: Optional[float]
    ebit: Optional[float]
    tax_rate: Optional[float]
    nopat: Optional[float]          # EBIT × (1 - Tax Rate)
    depreciation: Optional[float]
    capex: Optional[float]
    delta_nwc: Optional[float]      # Change in Net Working Capital
    fcff: Optional[float]           # NOPAT + D&A - ΔNWC - CAPEX

    # Derived ratios
    ebit_margin: Optional[float] = None
    fcff_margin: Optional[float] = None
    fcff_growth_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "Year": self.fiscal_year,
            "Revenue": self.revenue,
            "EBIT": self.ebit,
            "Tax Rate": self.tax_rate,
            "NOPAT": self.nopat,
            "D&A": self.depreciation,
            "CAPEX": self.capex,
            "ΔNWC": self.delta_nwc,
            "FCFF": self.fcff,
            "EBIT Margin": self.ebit_margin,
            "FCFF Margin": self.fcff_margin,
            "FCFF Growth": self.fcff_growth_rate,
        }

    def formatted(self) -> dict:
        """Human-readable formatted version."""
        return {
            "Year": str(self.fiscal_year),
            "Revenue": format_currency(self.revenue or 0),
            "EBIT": format_currency(self.ebit or 0),
            "Tax Rate": format_percentage(self.tax_rate or 0),
            "NOPAT": format_currency(self.nopat or 0),
            "D&A": format_currency(self.depreciation or 0),
            "CAPEX": format_currency(self.capex or 0),
            "ΔNWC": format_currency(self.delta_nwc or 0),
            "FCFF": format_currency(self.fcff or 0),
            "EBIT Margin": format_percentage(self.ebit_margin or 0),
            "FCFF Margin": format_percentage(self.fcff_margin or 0),
        }


@dataclass
class FCFFResult:
    """Complete output of the FCFF engine."""

    ticker: str
    breakdowns: list[FCFFBreakdown] = field(default_factory=list)
    average_tax_rate: float = 0.21
    average_fcff_growth: float = 0.05
    latest_fcff: Optional[float] = None
    median_fcff_margin: Optional[float] = None
    formula: str = "FCFF = NOPAT + D&A − ΔNWC − CAPEX"
    notes: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return historical FCFF as a DataFrame."""
        return pd.DataFrame([b.to_dict() for b in self.breakdowns])

    def to_formatted_dataframe(self) -> pd.DataFrame:
        """Return formatted DataFrame for display."""
        return pd.DataFrame([b.formatted() for b in self.breakdowns])


class FCFFEngine:
    """
    Computes historical Free Cash Flow to Firm (FCFF).

    FCFF represents cash available to ALL capital providers (equity + debt),
    making it ideal for enterprise valuation via DCF.

    Formula:
        FCFF = EBIT × (1 − Tax Rate) + Depreciation & Amortization
                − Changes in Net Working Capital − Capital Expenditures

    Where:
        NOPAT = EBIT × (1 − Tax Rate)  [Net Operating Profit After Tax]
    """

    # Tax rate bounds
    MIN_TAX_RATE: float = 0.05
    MAX_TAX_RATE: float = 0.40
    DEFAULT_TAX_RATE: float = 0.21  # US corporate tax rate

    def compute(self, data: FinancialData) -> FCFFResult:
        """
        Compute historical FCFF from a FinancialData object.

        Args:
            data: Complete financial data for a company.

        Returns:
            FCFFResult with year-by-year breakdown and summary statistics.
        """
        ticker = data.ticker
        logger.info(f"[FCFFEngine] Computing FCFF for {ticker}")

        # Align statements by fiscal year
        income_map = {s.fiscal_year: s for s in data.income_statements}
        balance_map = {s.fiscal_year: s for s in data.balance_sheets}
        cashflow_map = {s.fiscal_year: s for s in data.cash_flow_statements}

        years = sorted(set(income_map) & set(balance_map) & set(cashflow_map), reverse=True)
        if not years:
            logger.warning(f"No aligned financial data found for {ticker}")
            return FCFFResult(ticker=ticker, notes=["Insufficient data for FCFF calculation"])

        # Compute average tax rate (will be used when individual year rate is missing)
        avg_tax = self._compute_avg_tax_rate(
            [income_map[y] for y in years if y in income_map]
        )

        breakdowns: list[FCFFBreakdown] = []
        for i, year in enumerate(years):
            income = income_map.get(year)
            balance = balance_map.get(year)
            cashflow = cashflow_map.get(year)

            if not income or not balance or not cashflow:
                continue

            breakdown = self._compute_single_year(
                year=year,
                income=income,
                balance=balance,
                cashflow=cashflow,
                prev_balance=balance_map.get(years[i + 1]) if i + 1 < len(years) else None,
                avg_tax_rate=avg_tax,
            )
            breakdowns.append(breakdown)

        # Compute growth rates (after we have all FCFFs)
        for i, bd in enumerate(breakdowns[:-1]):
            prev = breakdowns[i + 1]
            if bd.fcff and prev.fcff and prev.fcff != 0:
                bd.fcff_growth_rate = (bd.fcff - prev.fcff) / abs(prev.fcff)

        # Summary statistics
        fcff_values = [b.fcff for b in breakdowns if b.fcff is not None]
        fcff_margins = [b.fcff_margin for b in breakdowns if b.fcff_margin is not None]
        growth_rates = [b.fcff_growth_rate for b in breakdowns if b.fcff_growth_rate is not None]

        # Exclude extreme outliers for average growth
        growth_clean = [g for g in growth_rates if abs(g) < 5.0]  # Exclude >500% growth

        result = FCFFResult(
            ticker=ticker,
            breakdowns=breakdowns,
            average_tax_rate=avg_tax,
            average_fcff_growth=float(np.median(growth_clean)) if growth_clean else 0.05,
            latest_fcff=breakdowns[0].fcff if breakdowns else None,
            median_fcff_margin=float(np.median(fcff_margins)) if fcff_margins else None,
            notes=self._generate_notes(breakdowns, avg_tax),
        )

        logger.success(
            f"[FCFFEngine] {ticker}: Latest FCFF = {format_currency(result.latest_fcff or 0)}, "
            f"Avg Tax = {format_percentage(avg_tax)}"
        )
        return result

    def _compute_single_year(
        self,
        year: int,
        income: AnnualIncomeStatement,
        balance: AnnualBalanceSheet,
        cashflow: AnnualCashFlowStatement,
        prev_balance: Optional[AnnualBalanceSheet],
        avg_tax_rate: float,
    ) -> FCFFBreakdown:
        """Compute FCFF for a single fiscal year."""

        # 1. EBIT
        ebit = income.ebit or self._estimate_ebit(income)

        # 2. Effective Tax Rate
        tax_rate = self._get_tax_rate(income, avg_tax_rate)

        # 3. NOPAT = EBIT × (1 − Tax Rate)
        nopat = ebit * (1 - tax_rate) if ebit is not None else None

        # 4. D&A — prefer cash flow statement over income statement
        da = (
            cashflow.depreciation_amortization
            or income.depreciation_amortization
            or 0.0
        )

        # 5. CAPEX — use absolute value (capex is typically negative in yfinance)
        capex_raw = cashflow.capital_expenditures or 0.0
        capex = abs(capex_raw)

        # 6. ΔNWC = NWC_current - NWC_previous
        nwc_curr = balance.net_working_capital
        nwc_prev = prev_balance.net_working_capital if prev_balance else None
        delta_nwc = (nwc_curr - nwc_prev) if (nwc_curr is not None and nwc_prev is not None) else 0.0

        # 7. FCFF = NOPAT + D&A − ΔNWC − CAPEX
        if nopat is not None:
            fcff = nopat + da - delta_nwc - capex
        else:
            fcff = None

        return FCFFBreakdown(
            fiscal_year=year,
            revenue=income.revenue,
            ebit=ebit,
            tax_rate=tax_rate,
            nopat=nopat,
            depreciation=da,
            capex=capex,
            delta_nwc=delta_nwc,
            fcff=fcff,
            ebit_margin=safe_divide(ebit or 0, income.revenue or 0),
            fcff_margin=safe_divide(fcff or 0, income.revenue or 1) if fcff else None,
        )

    def _estimate_ebit(self, income: AnnualIncomeStatement) -> Optional[float]:
        """Estimate EBIT from available data."""
        if income.ebitda and income.depreciation_amortization:
            return income.ebitda - income.depreciation_amortization
        if income.pretax_income and income.interest_expense:
            return income.pretax_income + (income.interest_expense or 0)
        return None

    def _get_tax_rate(self, income: AnnualIncomeStatement, fallback: float) -> float:
        """Get effective tax rate, falling back to average or default."""
        if income.effective_tax_rate and self.MIN_TAX_RATE <= income.effective_tax_rate <= self.MAX_TAX_RATE:
            return income.effective_tax_rate
        if income.income_tax and income.pretax_income and income.pretax_income > 0:
            computed = income.income_tax / income.pretax_income
            if self.MIN_TAX_RATE <= computed <= self.MAX_TAX_RATE:
                return computed
        return fallback

    def _compute_avg_tax_rate(self, income_list: list[AnnualIncomeStatement]) -> float:
        """Compute average effective tax rate across years."""
        rates = []
        for income in income_list:
            if income.income_tax and income.pretax_income and income.pretax_income > 0:
                rate = income.income_tax / income.pretax_income
                if self.MIN_TAX_RATE <= rate <= self.MAX_TAX_RATE:
                    rates.append(rate)
        return float(np.median(rates)) if rates else self.DEFAULT_TAX_RATE

    def _generate_notes(self, breakdowns: list[FCFFBreakdown], tax_rate: float) -> list[str]:
        """Generate analyst-style notes on data quality."""
        notes = []
        missing_ebit = sum(1 for b in breakdowns if b.ebit is None)
        if missing_ebit > 0:
            notes.append(f"EBIT estimated for {missing_ebit} year(s) due to missing data")

        neg_fcff = sum(1 for b in breakdowns if b.fcff and b.fcff < 0)
        if neg_fcff > 0:
            notes.append(
                f"Negative FCFF in {neg_fcff} year(s) — company investing heavily (may normalize)"
            )

        if tax_rate == self.DEFAULT_TAX_RATE:
            notes.append(f"Using statutory tax rate of {format_percentage(tax_rate)} as fallback")

        return notes


__all__ = ["FCFFEngine", "FCFFResult", "FCFFBreakdown"]
