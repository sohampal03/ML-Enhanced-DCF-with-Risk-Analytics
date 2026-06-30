"""
yfinance data adapter.

Implements the DataSourceProtocol using yfinance as the underlying library.
Handles US tickers (AAPL, MSFT) and international tickers (INFY.NS, RELIANCE.NS).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception  # fallback for older yfinance

try:
    from curl_cffi import requests as curl_requests

    _CURL_SESSION = curl_requests.Session(impersonate="chrome")
    _HAS_CURL = True
    logger.info("curl_cffi session active — Yahoo Finance rate limiting bypassed")
except ImportError:
    _CURL_SESSION = None
    _HAS_CURL = False
    logger.warning("curl_cffi not available — may hit Yahoo Finance rate limits")

from src.data.schemas.financial_schemas import (
    AnnualBalanceSheet,
    AnnualCashFlowStatement,
    AnnualIncomeStatement,
    CompanyInfo,
    FinancialData,
)
from src.utils.helpers import disk_cache, timer


class YFinanceAdapter:
    """
    Data adapter wrapping yfinance for financial statement retrieval.

    Normalizes yfinance output to the canonical FinancialData schema.
    Supports disk-caching to avoid repeated API calls during development.
    """

    def __init__(self, cache_ttl: int = 3600) -> None:
        self.cache_ttl = cache_ttl
        logger.info("YFinanceAdapter initialized")

    @timer
    def fetch(self, ticker: str, years: int = 10) -> FinancialData:
        """
        Fetch all financial data for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'INFY.NS').
            years: Number of historical years to retrieve.

        Returns:
            FinancialData containing company info, income, balance, cashflow.

        Raises:
            ValueError: If ticker data cannot be fetched.
        """
        ticker = ticker.upper().strip()
        logger.info(f"Fetching data for {ticker} via yfinance")

        stock = self._make_ticker(ticker)

        try:
            info = self._safe_fetch_info(stock)
        except YFRateLimitError:
            logger.warning("Rate limited by Yahoo Finance. Waiting 15s before retry...")
            time.sleep(15)
            stock = self._make_ticker(ticker)  # fresh session
            try:
                info = self._safe_fetch_info(stock)
            except Exception as e:
                raise ValueError(
                    f"Yahoo Finance is rate limiting requests. "
                    f"Please wait a moment and try again. ({e})"
                ) from e
        except Exception as e:
            raise ValueError(f"Could not fetch data for '{ticker}': {e}") from e

        company_info = self._parse_company_info(ticker, info)
        time.sleep(0.5)  # small delay between sequential API calls
        income_statements = self._parse_income_statements(ticker, stock)
        time.sleep(0.3)
        balance_sheets = self._parse_balance_sheets(ticker, stock)
        time.sleep(0.3)
        cash_flow_statements = self._parse_cash_flows(ticker, stock)
        price_history = self._fetch_price_history(ticker, years)

        logger.success(f"Successfully fetched {len(income_statements)} years of data for {ticker}")

        return FinancialData(
            company_info=company_info,
            income_statements=income_statements[:years],
            balance_sheets=balance_sheets[:years],
            cash_flow_statements=cash_flow_statements[:years],
            price_history=price_history.to_dict() if price_history is not None else None,
        )

    def _make_ticker(self, ticker: str) -> yf.Ticker:
        """Create a yf.Ticker, using curl_cffi session if available to bypass rate limits."""
        if _HAS_CURL and _CURL_SESSION is not None:
            return yf.Ticker(ticker, session=_CURL_SESSION)
        return yf.Ticker(ticker)

    def _safe_fetch_info(self, stock: yf.Ticker) -> dict:
        """Fetch ticker info with fast_info fallback. Does NOT retry — caller handles rate limits."""
        # Let YFRateLimitError propagate so the caller can handle it with proper delay
        info = stock.info or {}
        # Modern yfinance may not include 'symbol'; check for any meaningful data
        _has_data = bool(
            info.get("longName")
            or info.get("shortName")
            or info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("marketCap")
        )
        if not info or not _has_data:
            # Try fast_info as a lightweight fallback (separate API endpoint)
            try:
                fast = stock.fast_info
                last_price = getattr(fast, "last_price", None)
                market_cap = getattr(fast, "market_cap", None)
                shares = getattr(fast, "shares", None)
                currency = getattr(fast, "currency", "USD")
                if last_price or market_cap:
                    logger.info(f"Using fast_info fallback for {stock.ticker}")
                    return {
                        "symbol": stock.ticker,
                        "longName": stock.ticker,
                        "shortName": stock.ticker,
                        "currentPrice": last_price,
                        "marketCap": market_cap,
                        "sharesOutstanding": shares,
                        "currency": currency,
                    }
            except YFRateLimitError:
                raise  # let rate limit bubble up
            except Exception as fe:
                logger.debug(f"fast_info fallback failed: {fe}")
            raise ValueError(
                f"No valid data found for ticker '{stock.ticker}'. "
                "Please verify the symbol is correct (e.g. AAPL, MSFT, INFY.NS)."
            )
        return info

    def _parse_company_info(self, ticker: str, info: dict) -> CompanyInfo:
        """Parse yfinance info dict into CompanyInfo schema."""
        currency = info.get("currency", "USD")

        return CompanyInfo(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            description=info.get("longBusinessSummary"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            currency=currency,
            exchange=info.get("exchange"),
            website=info.get("website"),
            employees=info.get("fullTimeEmployees"),
            current_price=self._safe_float(
                info.get("currentPrice") or info.get("regularMarketPrice")
            ),
            market_cap=self._safe_float(info.get("marketCap")),
            shares_outstanding=self._safe_float(info.get("sharesOutstanding")),
            beta=self._safe_float(info.get("beta")),
            dividend_yield=self._safe_float(info.get("dividendYield")),
            pe_ratio=self._safe_float(info.get("trailingPE")),
            forward_pe=self._safe_float(info.get("forwardPE")),
            pb_ratio=self._safe_float(info.get("priceToBook")),
            ev_ebitda=self._safe_float(info.get("enterpriseToEbitda")),
            debt_to_equity=self._safe_float(info.get("debtToEquity")),
            return_on_equity=self._safe_float(info.get("returnOnEquity")),
            return_on_assets=self._safe_float(info.get("returnOnAssets")),
            profit_margin=self._safe_float(info.get("profitMargins")),
            revenue_growth=self._safe_float(info.get("revenueGrowth")),
        )

    def _parse_income_statements(
        self, ticker: str, stock: yf.Ticker
    ) -> list[AnnualIncomeStatement]:
        """Parse annual income statements."""
        try:
            df = stock.financials  # columns = date, rows = metrics
            if df is None or df.empty:
                logger.warning(f"No income statement data for {ticker}")
                return []
            df = df.T.sort_index(ascending=False)
        except Exception as e:
            logger.error(f"Error fetching income statement for {ticker}: {e}")
            return []

        results: list[AnnualIncomeStatement] = []
        for period_date, row in df.iterrows():
            try:
                revenue = self._get_value(row, ["Total Revenue", "Revenue"])
                cogs = self._get_value(row, ["Cost Of Revenue", "Cost of Revenue"])
                gross_profit = self._get_value(row, ["Gross Profit"])
                ebit = self._get_value(row, ["EBIT", "Operating Income"])
                ebitda = self._get_value(row, ["EBITDA", "Normalized EBITDA"])
                da = self._get_value(
                    row, ["Reconciled Depreciation", "Depreciation And Amortization"]
                )
                interest = self._get_value(row, ["Interest Expense", "Net Interest Income"])
                pretax = self._get_value(row, ["Pretax Income"])
                tax = self._get_value(row, ["Tax Provision", "Income Tax Expense"])
                net_income = self._get_value(row, ["Net Income", "Net Income Common Stockholders"])
                sga = self._get_value(row, ["Selling General Administrative", "SGA"])
                rd = self._get_value(row, ["Research And Development"])
                opex = self._get_value(row, ["Operating Expense", "Total Expenses"])
                eps = self._get_value(row, ["Diluted EPS", "Basic EPS"])
                shares = self._get_value(row, ["Diluted Average Shares", "Basic Average Shares"])

                year = (
                    period_date.year if hasattr(period_date, "year") else int(str(period_date)[:4])
                )

                stmt = AnnualIncomeStatement(
                    ticker=ticker,
                    fiscal_year=year,
                    fiscal_year_end=period_date.date() if hasattr(period_date, "date") else None,
                    revenue=revenue,
                    cost_of_revenue=cogs,
                    gross_profit=gross_profit if gross_profit else self._subtract(revenue, cogs),
                    research_and_development=rd,
                    selling_general_administrative=sga,
                    operating_expenses=opex,
                    ebitda=ebitda,
                    depreciation_amortization=da,
                    ebit=ebit,
                    interest_expense=interest,
                    pretax_income=pretax,
                    income_tax=tax,
                    net_income=net_income,
                    gross_margin=self._margin(gross_profit, revenue),
                    ebit_margin=self._margin(ebit, revenue),
                    net_margin=self._margin(net_income, revenue),
                    effective_tax_rate=self._margin(tax, pretax) if pretax else None,
                    eps_diluted=eps,
                    shares_diluted=shares,
                )
                results.append(stmt)
            except Exception as e:
                logger.debug(f"Skipping row {period_date}: {e}")
                continue

        return results

    def _parse_balance_sheets(self, ticker: str, stock: yf.Ticker) -> list[AnnualBalanceSheet]:
        """Parse annual balance sheets."""
        try:
            df = stock.balance_sheet
            if df is None or df.empty:
                logger.warning(f"No balance sheet data for {ticker}")
                return []
            df = df.T.sort_index(ascending=False)
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            return []

        results: list[AnnualBalanceSheet] = []
        for period_date, row in df.iterrows():
            try:
                cash = self._get_value(
                    row,
                    [
                        "Cash And Cash Equivalents",
                        "Cash Cash Equivalents And Short Term Investments",
                    ],
                )
                sti = self._get_value(row, ["Short Term Investments"])
                ar = self._get_value(row, ["Accounts Receivable", "Net Receivables"])
                inventory = self._get_value(row, ["Inventory"])
                current_assets = self._get_value(row, ["Current Assets", "Total Current Assets"])
                ppe = self._get_value(
                    row,
                    ["Net PPE", "Property Plant Equipment Net", "Net Property Plant And Equipment"],
                )
                goodwill = self._get_value(row, ["Goodwill"])
                intangibles = self._get_value(row, ["Intangible Assets", "Other Intangible Assets"])
                total_assets = self._get_value(row, ["Total Assets"])
                ap = self._get_value(row, ["Accounts Payable"])
                std = self._get_value(row, ["Current Debt", "Short Long Term Debt"])
                current_liab = self._get_value(
                    row, ["Current Liabilities", "Total Current Liabilities"]
                )
                ltd = self._get_value(
                    row, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
                )
                total_debt = self._get_value(row, ["Total Debt"])
                total_liab = self._get_value(
                    row, ["Total Liabilities Net Minority Interest", "Total Liab"]
                )
                equity = self._get_value(
                    row,
                    ["Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity"],
                )
                retained = self._get_value(row, ["Retained Earnings"])

                year = (
                    period_date.year if hasattr(period_date, "year") else int(str(period_date)[:4])
                )

                # Net working capital: current assets - current liabilities
                nwc = (
                    self._subtract(current_assets, current_liab)
                    if current_assets and current_liab
                    else None
                )

                bs = AnnualBalanceSheet(
                    ticker=ticker,
                    fiscal_year=year,
                    fiscal_year_end=period_date.date() if hasattr(period_date, "date") else None,
                    cash_and_equivalents=cash,
                    short_term_investments=sti,
                    accounts_receivable=ar,
                    inventory=inventory,
                    total_current_assets=current_assets,
                    property_plant_equipment=ppe,
                    goodwill=goodwill,
                    intangible_assets=intangibles,
                    total_assets=total_assets,
                    accounts_payable=ap,
                    short_term_debt=std,
                    total_current_liabilities=current_liab,
                    long_term_debt=ltd,
                    total_debt=total_debt if total_debt else self._add(std, ltd),
                    total_liabilities=total_liab,
                    total_equity=equity,
                    retained_earnings=retained,
                    net_working_capital=nwc,
                )
                results.append(bs)
            except Exception as e:
                logger.debug(f"Skipping balance sheet row {period_date}: {e}")
                continue

        return results

    def _parse_cash_flows(self, ticker: str, stock: yf.Ticker) -> list[AnnualCashFlowStatement]:
        """Parse annual cash flow statements."""
        try:
            df = stock.cashflow
            if df is None or df.empty:
                logger.warning(f"No cash flow data for {ticker}")
                return []
            df = df.T.sort_index(ascending=False)
        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            return []

        results: list[AnnualCashFlowStatement] = []
        for period_date, row in df.iterrows():
            try:
                net_income = self._get_value(
                    row, ["Net Income", "Net Income From Continuing Operations"]
                )
                da = self._get_value(
                    row,
                    [
                        "Depreciation And Amortization",
                        "Depreciation Amortization Depletion",
                        "Depreciation",
                    ],
                )
                sbc = self._get_value(row, ["Stock Based Compensation"])
                wc_change = self._get_value(
                    row, ["Change In Working Capital", "Changes In Account Receivables"]
                )
                ocf = self._get_value(row, ["Operating Cash Flow", "Cash From Operations"])
                capex = self._get_value(
                    row, ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]
                )
                acquisitions = self._get_value(row, ["Acquisitions Net"])
                icf = self._get_value(row, ["Investing Cash Flow", "Cash From Investing"])
                debt_repay = self._get_value(row, ["Repayment Of Debt", "Debt Repayment"])
                dividends = self._get_value(row, ["Payment Of Dividends", "Cash Dividends Paid"])
                repurchases = self._get_value(
                    row, ["Repurchase Of Capital Stock", "Common Stock Repurchased"]
                )
                fcf = self._get_value(row, ["Free Cash Flow"])
                fcf_calc = (
                    self._subtract(ocf, abs(capex) if capex else 0) if ocf and capex else None
                )

                year = (
                    period_date.year if hasattr(period_date, "year") else int(str(period_date)[:4])
                )

                cf = AnnualCashFlowStatement(
                    ticker=ticker,
                    fiscal_year=year,
                    fiscal_year_end=period_date.date() if hasattr(period_date, "date") else None,
                    net_income=net_income,
                    depreciation_amortization=da,
                    stock_based_compensation=sbc,
                    change_in_working_capital=wc_change,
                    operating_cash_flow=ocf,
                    capital_expenditures=capex,
                    acquisitions=acquisitions,
                    investing_cash_flow=icf,
                    debt_repayment=debt_repay,
                    dividends_paid=dividends,
                    share_repurchases=repurchases,
                    free_cash_flow=fcf if fcf else fcf_calc,
                )
                results.append(cf)
            except Exception as e:
                logger.debug(f"Skipping cash flow row {period_date}: {e}")
                continue

        return results

    @disk_cache(ttl_seconds=900)
    def _fetch_price_history(self, ticker: str, years: int = 10) -> pd.DataFrame | None:
        """Fetch historical OHLCV price data."""
        try:
            period = f"{min(years, 10)}y"
            session = _CURL_SESSION if _HAS_CURL else None
            df = yf.download(
                ticker, period=period, auto_adjust=True, progress=False, session=session
            )
            if df.empty:
                return None
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Price history fetch failed for {ticker}: {e}")
            return None

    # ── Helper methods ────────────────────────────────────────────────────────

    @staticmethod
    def _get_value(row: pd.Series, keys: list[str]) -> float | None:
        """Try multiple column name variants, return first found."""
        for key in keys:
            if key in row.index and not pd.isna(row[key]):
                val = row[key]
                return float(val) if val is not None else None
        return None

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            f = float(value)  # type: ignore
            return None if np.isnan(f) or np.isinf(f) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _margin(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _subtract(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    @staticmethod
    def _add(a: float | None, b: float | None) -> float | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)


__all__ = ["YFinanceAdapter"]
