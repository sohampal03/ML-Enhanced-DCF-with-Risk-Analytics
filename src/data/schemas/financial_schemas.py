"""
Pydantic schemas for financial statement data.

These are the canonical data models used throughout the platform.
All data source adapters must normalize their output to these schemas.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class CompanyInfo(BaseModel):
    """Core company metadata."""

    ticker: str
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: str = "USD"
    exchange: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None

    # Market data
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None

    # Fetched at
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("ticker", mode="before")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class AnnualIncomeStatement(BaseModel):
    """Normalized annual income statement."""

    ticker: str
    fiscal_year: int
    fiscal_year_end: Optional[date] = None

    # Revenue
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None

    # Operating
    research_and_development: Optional[float] = None
    selling_general_administrative: Optional[float] = None
    operating_expenses: Optional[float] = None
    ebitda: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    ebit: Optional[float] = None

    # Below the line
    interest_expense: Optional[float] = None
    other_income: Optional[float] = None
    pretax_income: Optional[float] = None
    income_tax: Optional[float] = None
    net_income: Optional[float] = None

    # Derived
    gross_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    net_margin: Optional[float] = None
    effective_tax_rate: Optional[float] = None
    eps_diluted: Optional[float] = None
    shares_diluted: Optional[float] = None


class AnnualBalanceSheet(BaseModel):
    """Normalized annual balance sheet."""

    ticker: str
    fiscal_year: int
    fiscal_year_end: Optional[date] = None

    # Assets
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    total_current_assets: Optional[float] = None
    property_plant_equipment: Optional[float] = None
    goodwill: Optional[float] = None
    intangible_assets: Optional[float] = None
    total_assets: Optional[float] = None

    # Liabilities
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    total_liabilities: Optional[float] = None

    # Equity
    total_equity: Optional[float] = None
    retained_earnings: Optional[float] = None

    # Net Working Capital
    net_working_capital: Optional[float] = None


class AnnualCashFlowStatement(BaseModel):
    """Normalized annual cash flow statement."""

    ticker: str
    fiscal_year: int
    fiscal_year_end: Optional[date] = None

    # Operating
    net_income: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    change_in_working_capital: Optional[float] = None
    operating_cash_flow: Optional[float] = None

    # Investing
    capital_expenditures: Optional[float] = None
    acquisitions: Optional[float] = None
    investing_cash_flow: Optional[float] = None

    # Financing
    debt_repayment: Optional[float] = None
    dividends_paid: Optional[float] = None
    share_repurchases: Optional[float] = None
    financing_cash_flow: Optional[float] = None

    # Free Cash Flow
    free_cash_flow: Optional[float] = None


class FinancialData(BaseModel):
    """Aggregated financial data container for a single company."""

    company_info: CompanyInfo
    income_statements: list[AnnualIncomeStatement] = Field(default_factory=list)
    balance_sheets: list[AnnualBalanceSheet] = Field(default_factory=list)
    cash_flow_statements: list[AnnualCashFlowStatement] = Field(default_factory=list)
    price_history: Optional[dict] = None  # Serialized DataFrame

    @property
    def ticker(self) -> str:
        return self.company_info.ticker

    @property
    def latest_income(self) -> Optional[AnnualIncomeStatement]:
        return self.income_statements[0] if self.income_statements else None

    @property
    def latest_balance(self) -> Optional[AnnualBalanceSheet]:
        return self.balance_sheets[0] if self.balance_sheets else None

    @property
    def latest_cash_flow(self) -> Optional[AnnualCashFlowStatement]:
        return self.cash_flow_statements[0] if self.cash_flow_statements else None

    def income_df(self) -> pd.DataFrame:
        """Return income statements as a DataFrame sorted by fiscal year desc."""
        return (
            pd.DataFrame([s.model_dump() for s in self.income_statements])
            .sort_values("fiscal_year", ascending=False)
            .reset_index(drop=True)
        )

    def balance_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame([s.model_dump() for s in self.balance_sheets])
            .sort_values("fiscal_year", ascending=False)
            .reset_index(drop=True)
        )

    def cashflow_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame([s.model_dump() for s in self.cash_flow_statements])
            .sort_values("fiscal_year", ascending=False)
            .reset_index(drop=True)
        )


class ValuationResult(BaseModel):
    """Output from the DCF valuation engine."""

    ticker: str
    company_name: str
    current_price: float
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    margin_of_safety: float  # (intrinsic - current) / intrinsic
    wacc: float
    terminal_growth_rate: float
    forecast_years: int

    # Recommendation
    recommendation: str  # BUY / HOLD / SELL / STRONG BUY / STRONG SELL
    confidence_score: float  # 0–1

    # Waterfall components
    pv_fcff_sum: float
    terminal_value: float
    pv_terminal_value: float
    total_debt: float
    cash: float

    # FCFF breakdown
    fcff_history: list[dict] = Field(default_factory=list)
    fcff_forecast: list[dict] = Field(default_factory=list)

    computed_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    "CompanyInfo",
    "AnnualIncomeStatement",
    "AnnualBalanceSheet",
    "AnnualCashFlowStatement",
    "FinancialData",
    "ValuationResult",
]
