"""
WACC (Weighted Average Cost of Capital) Engine.

Computes WACC using CAPM for cost of equity and market-adjusted cost of debt.

    WACC = (E/V) × Ke + (D/V) × Kd × (1 − Tax Rate)

Where:
    Ke  = Cost of Equity  = Rf + β × (Rm − Rf)      [CAPM]
    Kd  = Cost of Debt    = Interest Expense / Total Debt
    E   = Market Value of Equity
    D   = Total Debt
    V   = E + D
    Rf  = Risk-Free Rate  (10Y US Treasury)
    Rm  = Market Return
    β   = Beta
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from configs.settings import settings
from src.data.schemas.financial_schemas import FinancialData
from src.utils.helpers import format_percentage, safe_divide


@dataclass
class WACCBreakdown:
    """Detailed WACC decomposition."""

    # Inputs
    beta: float
    risk_free_rate: float
    market_risk_premium: float
    tax_rate: float
    market_equity: float  # Market cap
    total_debt: float

    # CAPM
    cost_of_equity: float  # Ke = Rf + β × (Rm - Rf)
    cost_of_debt_pretax: float
    cost_of_debt_after_tax: float  # Kd × (1 - Tax)

    # Weights
    equity_weight: float  # E / V
    debt_weight: float  # D / V
    total_capital: float  # E + D

    # WACC
    wacc: float

    # Explanation strings
    capm_formula: str = ""
    wacc_formula: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "Component": [
                "Risk-Free Rate (Rf)",
                "Beta (β)",
                "Market Risk Premium (Rm − Rf)",
                "Cost of Equity (Ke = Rf + β × MRP)",
                "Cost of Debt (Pre-Tax Kd)",
                "Tax Rate",
                "Cost of Debt (After-Tax)",
                "Equity Weight (E/V)",
                "Debt Weight (D/V)",
                "WACC",
            ],
            "Value": [
                format_percentage(self.risk_free_rate),
                f"{self.beta:.2f}x",
                format_percentage(self.market_risk_premium),
                format_percentage(self.cost_of_equity),
                format_percentage(self.cost_of_debt_pretax),
                format_percentage(self.tax_rate),
                format_percentage(self.cost_of_debt_after_tax),
                format_percentage(self.equity_weight),
                format_percentage(self.debt_weight),
                format_percentage(self.wacc),
            ],
        }


class WACCEngine:
    """
    Computes the Weighted Average Cost of Capital.

    Assumptions & defaults are sourced from settings but can be overridden
    at call time — useful for the interactive DCF Lab sliders.

    Key references:
      - Damodaran (2024): Country Risk Premiums and Market Risk Premiums
      - CAPM: Sharpe (1964), Lintner (1965)
    """

    # WACC guardrails
    MIN_WACC: float = 0.04  # 4%
    MAX_WACC: float = 0.30  # 30%
    DEFAULT_BETA: float = 1.0

    def compute(
        self,
        data: FinancialData,
        risk_free_rate: Optional[float] = None,
        market_risk_premium: Optional[float] = None,
        tax_rate_override: Optional[float] = None,
        beta_override: Optional[float] = None,
    ) -> WACCBreakdown:
        """
        Compute WACC for a company.

        Args:
            data: FinancialData for the company.
            risk_free_rate: Override the 10Y treasury yield (decimal).
            market_risk_premium: Override the equity risk premium (decimal).
            tax_rate_override: Override the effective tax rate.
            beta_override: Override the beta.

        Returns:
            WACCBreakdown with all intermediate calculations.
        """
        ticker = data.ticker
        logger.info(f"[WACCEngine] Computing WACC for {ticker}")

        rf = risk_free_rate or settings.dcf.risk_free_rate
        mrp = market_risk_premium or settings.dcf.market_risk_premium

        # ── Step 1: Beta ─────────────────────────────────────────────────────
        beta = beta_override or data.company_info.beta or self.DEFAULT_BETA
        beta = max(0.1, min(beta, 3.0))  # Clamp to reasonable range
        notes: list[str] = []
        if data.company_info.beta is None:
            notes.append("Beta not available; using market beta of 1.0")

        # ── Step 2: Cost of Equity (CAPM) ────────────────────────────────────
        cost_of_equity = rf + beta * mrp
        capm_formula = f"Ke = {rf:.2%} + {beta:.2f} × {mrp:.2%} = {cost_of_equity:.2%}"

        # ── Step 3: Tax Rate ─────────────────────────────────────────────────
        tax_rate = tax_rate_override or self._estimate_tax_rate(data)

        # ── Step 4: Cost of Debt ─────────────────────────────────────────────
        kd_pretax = self._estimate_cost_of_debt(data)
        kd_after_tax = kd_pretax * (1 - tax_rate)

        # ── Step 5: Capital Structure ─────────────────────────────────────────
        market_equity = data.company_info.market_cap or 0.0
        total_debt = self._get_total_debt(data)
        total_capital = market_equity + total_debt

        if total_capital <= 0:
            raise ValueError(f"Cannot compute WACC: invalid capital structure for {ticker}")

        equity_weight = safe_divide(market_equity, total_capital)
        debt_weight = safe_divide(total_debt, total_capital)

        # ── Step 6: WACC ─────────────────────────────────────────────────────
        wacc = equity_weight * cost_of_equity + debt_weight * kd_after_tax
        wacc = max(self.MIN_WACC, min(wacc, self.MAX_WACC))

        wacc_formula = (
            f"WACC = {equity_weight:.1%} × {cost_of_equity:.2%} + "
            f"{debt_weight:.1%} × {kd_pretax:.2%} × (1 − {tax_rate:.1%}) = {wacc:.2%}"
        )

        # Analyst notes
        if total_debt < 1e6:
            notes.append("Company appears essentially debt-free; WACC ≈ cost of equity")
        if cost_of_equity > 0.20:
            notes.append("High cost of equity (>20%) — company carries significant systematic risk")
        if wacc == self.MAX_WACC:
            notes.append("WACC capped at 30% — actual computed value exceeded maximum bound")

        result = WACCBreakdown(
            beta=beta,
            risk_free_rate=rf,
            market_risk_premium=mrp,
            tax_rate=tax_rate,
            market_equity=market_equity,
            total_debt=total_debt,
            cost_of_equity=cost_of_equity,
            cost_of_debt_pretax=kd_pretax,
            cost_of_debt_after_tax=kd_after_tax,
            equity_weight=equity_weight,
            debt_weight=debt_weight,
            total_capital=total_capital,
            wacc=wacc,
            capm_formula=capm_formula,
            wacc_formula=wacc_formula,
            notes=notes,
        )

        logger.success(
            f"[WACCEngine] {ticker}: WACC = {format_percentage(wacc)}, "
            f"Ke = {format_percentage(cost_of_equity)}, "
            f"Kd (after-tax) = {format_percentage(kd_after_tax)}"
        )
        return result

    def _estimate_cost_of_debt(self, data: FinancialData) -> float:
        """
        Estimate pre-tax cost of debt.

        Prefers: Interest Expense / Total Debt ratio.
        Fallback: 5% (typical IG corporate bond yield).
        """
        fallback = 0.05  # 5% default

        latest_income = data.latest_income
        latest_balance = data.latest_balance

        if latest_income and latest_balance:
            interest = abs(latest_income.interest_expense or 0)
            debt = latest_balance.total_debt or latest_balance.long_term_debt or 0

            if debt > 1e6 and interest > 0:
                kd = interest / debt
                if 0.01 <= kd <= 0.20:  # Sanity check
                    return kd

        logger.debug("Using fallback cost of debt 5.0%")
        return fallback

    def _estimate_tax_rate(self, data: FinancialData) -> float:
        """Estimate effective tax rate from most recent income statement."""
        DEFAULT = 0.21

        for income in data.income_statements[:3]:  # Use last 3 years
            if income.income_tax and income.pretax_income and income.pretax_income > 0:
                rate = income.income_tax / income.pretax_income
                if 0.05 <= rate <= 0.40:
                    return rate

        # Check effective_tax_rate field
        if data.latest_income and data.latest_income.effective_tax_rate:
            rate = data.latest_income.effective_tax_rate
            if 0.05 <= rate <= 0.40:
                return rate

        return DEFAULT

    def _get_total_debt(self, data: FinancialData) -> float:
        """Get total debt from most recent balance sheet."""
        balance = data.latest_balance
        if not balance:
            return 0.0
        debt = balance.total_debt or 0
        if debt == 0:
            debt = (balance.long_term_debt or 0) + (balance.short_term_debt or 0)
        return debt


__all__ = ["WACCEngine", "WACCBreakdown"]
