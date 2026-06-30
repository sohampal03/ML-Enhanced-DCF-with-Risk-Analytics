"""
Unit tests for the FCFF Engine.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.finance.engines.fcff_engine import FCFFEngine, FCFFResult
from src.data.schemas.financial_schemas import (
    FinancialData,
    CompanyInfo,
    AnnualIncomeStatement,
    AnnualBalanceSheet,
    AnnualCashFlowStatement,
)


def make_sample_data(ticker: str = "TEST") -> FinancialData:
    """Create minimal FinancialData for testing."""
    income = [
        AnnualIncomeStatement(
            ticker=ticker,
            fiscal_year=2023,
            revenue=100e9,
            ebit=20e9,
            pretax_income=18e9,
            income_tax=3.78e9,
            net_income=14.22e9,
            depreciation_amortization=5e9,
            ebit_margin=0.20,
        ),
        AnnualIncomeStatement(
            ticker=ticker,
            fiscal_year=2022,
            revenue=90e9,
            ebit=18e9,
            pretax_income=16e9,
            income_tax=3.36e9,
            net_income=12.64e9,
            depreciation_amortization=4.5e9,
            ebit_margin=0.20,
        ),
        AnnualIncomeStatement(
            ticker=ticker,
            fiscal_year=2021,
            revenue=80e9,
            ebit=15e9,
            pretax_income=13e9,
            income_tax=2.73e9,
            net_income=10.27e9,
            depreciation_amortization=4e9,
            ebit_margin=0.19,
        ),
    ]
    balance = [
        AnnualBalanceSheet(
            ticker=ticker,
            fiscal_year=year,
            total_debt=30e9,
            cash_and_equivalents=10e9,
            total_assets=200e9,
            total_equity=100e9,
            net_working_capital=20e9 + (year - 2021) * 2e9,
        )
        for year in [2023, 2022, 2021]
    ]
    cashflow = [
        AnnualCashFlowStatement(
            ticker=ticker,
            fiscal_year=2023,
            operating_cash_flow=22e9,
            capital_expenditures=-5e9,
            free_cash_flow=17e9,
            depreciation_amortization=5e9,
        ),
        AnnualCashFlowStatement(
            ticker=ticker,
            fiscal_year=2022,
            operating_cash_flow=19e9,
            capital_expenditures=-4.5e9,
            free_cash_flow=14.5e9,
            depreciation_amortization=4.5e9,
        ),
        AnnualCashFlowStatement(
            ticker=ticker,
            fiscal_year=2021,
            operating_cash_flow=16e9,
            capital_expenditures=-4e9,
            free_cash_flow=12e9,
            depreciation_amortization=4e9,
        ),
    ]
    return FinancialData(
        company_info=CompanyInfo(
            ticker=ticker,
            name="Test Co",
            current_price=150.0,
            market_cap=500e9,
            shares_outstanding=3.33e9,
        ),
        income_statements=income,
        balance_sheets=balance,
        cash_flow_statements=cashflow,
    )


class TestFCFFEngine:
    def test_compute_returns_result(self):
        data = make_sample_data()
        engine = FCFFEngine()
        result = engine.compute(data)
        assert isinstance(result, FCFFResult)
        assert result.ticker == "TEST"

    def test_fcff_positive_for_profitable_company(self):
        data = make_sample_data()
        result = FCFFEngine().compute(data)
        assert result.latest_fcff is not None
        # A company with 20% EBIT margin and moderate capex should have positive FCFF
        assert result.latest_fcff > 0

    def test_tax_rate_within_bounds(self):
        data = make_sample_data()
        result = FCFFEngine().compute(data)
        assert 0.05 <= result.average_tax_rate <= 0.40

    def test_breakdowns_match_years(self):
        data = make_sample_data()
        result = FCFFEngine().compute(data)
        assert len(result.breakdowns) == 3

    def test_handles_missing_data(self):
        """Test that engine handles incomplete data gracefully."""
        data = FinancialData(
            company_info=CompanyInfo(ticker="EMPTY", name="Empty Co"),
            income_statements=[],
            balance_sheets=[],
            cash_flow_statements=[],
        )
        result = FCFFEngine().compute(data)
        assert result.latest_fcff is None

    def test_formula_string_present(self):
        result = FCFFEngine().compute(make_sample_data())
        assert "FCFF" in result.formula

    def test_to_dataframe(self):
        result = FCFFEngine().compute(make_sample_data())
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "FCFF" in df.columns
        assert len(df) > 0
