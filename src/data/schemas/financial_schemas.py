"""
Pydantic schemas for financial statement data.

These are the canonical data models used throughout the platform.
All data source adapters must normalize their output to these schemas.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class CompanyInfo(BaseModel):
    """Core company metadata."""

    ticker: str
    name: str
    description: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str = "USD"
    exchange: str | None = None
    website: str | None = None
    employees: int | None = None

    # Market data
    current_price: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None
    beta: float | None = None
    dividend_yield: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    debt_to_equity: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None

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
    fiscal_year_end: date | None = None

    # Revenue
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None

    # Operating
    research_and_development: float | None = None
    selling_general_administrative: float | None = None
    operating_expenses: float | None = None
    ebitda: float | None = None
    depreciation_amortization: float | None = None
    ebit: float | None = None

    # Below the line
    interest_expense: float | None = None
    other_income: float | None = None
    pretax_income: float | None = None
    income_tax: float | None = None
    net_income: float | None = None

    # Derived
    gross_margin: float | None = None
    ebit_margin: float | None = None
    net_margin: float | None = None
    effective_tax_rate: float | None = None
    eps_diluted: float | None = None
    shares_diluted: float | None = None


class AnnualBalanceSheet(BaseModel):
    """Normalized annual balance sheet."""

    ticker: str
    fiscal_year: int
    fiscal_year_end: date | None = None

    # Assets
    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    accounts_receivable: float | None = None
    inventory: float | None = None
    total_current_assets: float | None = None
    property_plant_equipment: float | None = None
    goodwill: float | None = None
    intangible_assets: float | None = None
    total_assets: float | None = None

    # Liabilities
    accounts_payable: float | None = None
    short_term_debt: float | None = None
    total_current_liabilities: float | None = None
    long_term_debt: float | None = None
    total_debt: float | None = None
    total_liabilities: float | None = None

    # Equity
    total_equity: float | None = None
    retained_earnings: float | None = None

    # Net Working Capital
    net_working_capital: float | None = None


class AnnualCashFlowStatement(BaseModel):
    """Normalized annual cash flow statement."""

    ticker: str
    fiscal_year: int
    fiscal_year_end: date | None = None

    # Operating
    net_income: float | None = None
    depreciation_amortization: float | None = None
    stock_based_compensation: float | None = None
    change_in_working_capital: float | None = None
    operating_cash_flow: float | None = None

    # Investing
    capital_expenditures: float | None = None
    acquisitions: float | None = None
    investing_cash_flow: float | None = None

    # Financing
    debt_repayment: float | None = None
    dividends_paid: float | None = None
    share_repurchases: float | None = None
    financing_cash_flow: float | None = None

    # Free Cash Flow
    free_cash_flow: float | None = None


class FinancialData(BaseModel):
    """Aggregated financial data container for a single company."""

    company_info: CompanyInfo
    income_statements: list[AnnualIncomeStatement] = Field(default_factory=list)
    balance_sheets: list[AnnualBalanceSheet] = Field(default_factory=list)
    cash_flow_statements: list[AnnualCashFlowStatement] = Field(default_factory=list)
    price_history: dict | None = None  # Serialized DataFrame

    @property
    def ticker(self) -> str:
        return self.company_info.ticker

    @property
    def latest_income(self) -> AnnualIncomeStatement | None:
        return self.income_statements[0] if self.income_statements else None

    @property
    def latest_balance(self) -> AnnualBalanceSheet | None:
        return self.balance_sheets[0] if self.balance_sheets else None

    @property
    def latest_cash_flow(self) -> AnnualCashFlowStatement | None:
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
