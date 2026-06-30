"""
Utility helpers: caching, timing, retry, and formatting decorators.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

F = TypeVar("F", bound=Callable[..., Any])

# ── Disk-based cache ─────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(*args: Any, **kwargs: Any) -> str:
    raw = json.dumps({"args": str(args), "kwargs": str(kwargs)}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def disk_cache(ttl_seconds: int = 3600) -> Callable[[F], F]:
    """
    Simple disk-based cache decorator for expensive API calls.

    Args:
        ttl_seconds: Time-to-live in seconds. Default 1 hour.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _cache_key(func.__name__, *args, **kwargs)
            cache_file = _CACHE_DIR / f"{key}.pkl"

            if cache_file.exists():
                age = time.time() - cache_file.stat().st_mtime
                if age < ttl_seconds:
                    logger.debug(f"Cache HIT: {func.__name__} [{key[:8]}]")
                    return pd.read_pickle(cache_file)

            logger.debug(f"Cache MISS: {func.__name__} [{key[:8]}]")
            result = func(*args, **kwargs)

            if result is not None:
                try:
                    pd.to_pickle(result, cache_file)
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return result

        return wrapper  # type: ignore

    return decorator


# ── Performance timing ────────────────────────────────────────────────────────


def timer(func: F) -> F:
    """Log function execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱  {func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper  # type: ignore


# ── Retry with exponential backoff ────────────────────────────────────────────


def with_retry(
    max_attempts: int = 3, wait_min: float = 1.0, wait_max: float = 10.0
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff for API calls."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.fn.__name__} "  # type: ignore
            f"(attempt {retry_state.attempt_number}/{max_attempts})"
        ),
    )  # type: ignore


# ── Financial formatting ──────────────────────────────────────────────────────


def format_currency(value: float, currency: str = "USD", compact: bool = True) -> str:
    """Format a number as currency with appropriate scale."""
    if pd.isna(value) or value is None:
        return "N/A"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    symbol = "$" if currency == "USD" else "₹" if currency == "INR" else currency

    if compact:
        if abs_val >= 1e12:
            return f"{sign}{symbol}{abs_val / 1e12:.2f}T"
        elif abs_val >= 1e9:
            return f"{sign}{symbol}{abs_val / 1e9:.2f}B"
        elif abs_val >= 1e6:
            return f"{sign}{symbol}{abs_val / 1e6:.2f}M"
        elif abs_val >= 1e3:
            return f"{sign}{symbol}{abs_val / 1e3:.2f}K"
    return f"{sign}{symbol}{abs_val:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_multiple(value: float, suffix: str = "x", decimals: int = 1) -> str:
    """Format a valuation multiple."""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default on zero denominator."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a pandas series to remove extreme outliers."""
    low = series.quantile(lower)
    high = series.quantile(upper)
    return series.clip(lower=low, upper=high)


def pct_change_safe(series: pd.Series) -> pd.Series:
    """Safe percentage change handling zeros and NaN."""
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


__all__ = [
    "disk_cache",
    "timer",
    "with_retry",
    "format_currency",
    "format_percentage",
    "format_multiple",
    "safe_divide",
    "winsorize_series",
    "pct_change_safe",
]
