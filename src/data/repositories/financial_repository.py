"""
Financial data repository with multi-level caching.

Provides a clean interface for retrieving financial data, abstracting
the underlying adapter (yfinance, FMP, etc.) and caching strategy.
"""

from __future__ import annotations

from loguru import logger

from configs.settings import settings
from src.data.adapters.yfinance_adapter import YFinanceAdapter
from src.data.schemas.financial_schemas import FinancialData


class FinancialRepository:
    """
    Repository pattern for financial data access.

    Provides a unified interface over multiple data adapters with
    automatic fallback and caching.

    Usage:
        repo = FinancialRepository()
        data = repo.get_financial_data("AAPL")
    """

    def __init__(self) -> None:
        self._adapter = self._build_adapter()
        self._in_memory_cache: dict[str, FinancialData] = {}
        logger.info(
            f"FinancialRepository initialized with adapter: {settings.data.primary_adapter}"
        )

    def _build_adapter(self) -> YFinanceAdapter:
        """Factory: build the configured data adapter."""
        adapter_type = settings.data.primary_adapter
        if adapter_type == "yfinance":
            return YFinanceAdapter(cache_ttl=settings.data.cache_ttl_seconds)
        # Future: elif adapter_type == "fmp": return FMPAdapter(...)
        logger.warning(f"Unknown adapter '{adapter_type}', falling back to yfinance")
        return YFinanceAdapter(cache_ttl=settings.data.cache_ttl_seconds)

    def get_financial_data(
        self,
        ticker: str,
        years: int | None = None,
        force_refresh: bool = False,
    ) -> FinancialData:
        """
        Retrieve complete financial data for a ticker.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
            years: Historical years. Defaults to settings value.
            force_refresh: Bypass all caches and re-fetch.

        Returns:
            FinancialData containing all financial statements.

        Raises:
            ValueError: If ticker is invalid or data is unavailable.
        """
        ticker = ticker.upper().strip()
        years = years or settings.data.historical_years

        # 1. In-memory cache (session-level)
        if not force_refresh and ticker in self._in_memory_cache:
            logger.debug(f"In-memory cache HIT for {ticker}")
            return self._in_memory_cache[ticker]

        # 2. Fetch from adapter (which handles disk cache internally)
        logger.info(f"Fetching {ticker} from adapter")
        data = self._adapter.fetch(ticker, years=years)

        # Validate we got useful data
        if not data.income_statements:
            raise ValueError(
                f"No financial statement data found for '{ticker}'. "
                "Please verify the ticker symbol is correct."
            )

        # Store in memory cache
        self._in_memory_cache[ticker] = data
        return data

    def invalidate_cache(self, ticker: str) -> None:
        """Remove a ticker from the in-memory cache."""
        self._in_memory_cache.pop(ticker.upper(), None)
        logger.info(f"Cache invalidated for {ticker}")

    def get_current_price(self, ticker: str) -> float | None:
        """Convenience method to get just the current price."""
        try:
            data = self.get_financial_data(ticker)
            return data.company_info.current_price
        except Exception as e:
            logger.error(f"Could not get current price for {ticker}: {e}")
            return None


__all__ = ["FinancialRepository"]
