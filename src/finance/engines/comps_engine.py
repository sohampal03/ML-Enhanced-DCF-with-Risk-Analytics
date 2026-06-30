"""
Comparable Company Analysis (Comps) Engine.

Fetches sector peers, computes valuation multiples, and ranks
the target company relative to peers.

Multiples: P/E, EV/EBITDA, EV/Sales, P/B, PEG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.schemas.financial_schemas import FinancialData
from src.utils.helpers import format_multiple, safe_divide

# Sector → Peer tickers mapping (curated list)
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "ADBE", "CRM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA"],
    "Consumer Cyclical": ["AMZN", "TSLA", "NKE", "HD", "MCD", "SBUX", "TGT"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "DHR"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C"],
    "Industrials": ["GE", "BA", "CAT", "DE", "MMM", "HON", "UNP", "RTX"],
    "Energy": ["XOM", "CVX", "SLB", "COP", "EOG", "PXD", "OXY"],
    "Basic Materials": ["LIN", "APD", "ECL", "NEM", "FCX", "VMC"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
}


@dataclass
class CompanyMultiples:
    """Valuation multiples for a single company."""

    ticker: str
    name: str
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    ev_ebitda: Optional[float]
    ev_sales: Optional[float]
    pb_ratio: Optional[float]
    peg_ratio: Optional[float]
    market_cap: Optional[float]
    revenue_growth: Optional[float]
    profit_margin: Optional[float]
    roe: Optional[float]
    is_target: bool = False

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Company": self.name,
            "P/E": format_multiple(self.pe_ratio),
            "Fwd P/E": format_multiple(self.forward_pe),
            "EV/EBITDA": format_multiple(self.ev_ebitda),
            "EV/Sales": format_multiple(self.ev_sales),
            "P/B": format_multiple(self.pb_ratio),
            "PEG": format_multiple(self.peg_ratio, "x", 2),
            "Rev Growth": f"{self.revenue_growth * 100:.1f}%" if self.revenue_growth else "N/A",
            "Net Margin": f"{self.profit_margin * 100:.1f}%" if self.profit_margin else "N/A",
            "ROE": f"{self.roe * 100:.1f}%" if self.roe else "N/A",
            "Is Target": self.is_target,
        }


@dataclass
class CompsResult:
    """Comparable companies analysis output."""

    target_ticker: str
    sector: Optional[str]
    industry: Optional[str]
    company_multiples: list[CompanyMultiples] = field(default_factory=list)
    peer_median_pe: Optional[float] = None
    peer_median_ev_ebitda: Optional[float] = None
    peer_median_ev_sales: Optional[float] = None
    implied_values: dict = field(default_factory=dict)
    attractiveness_rank: Optional[int] = None
    attractiveness_score: Optional[float] = None  # 0–10
    notes: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return multiples as a styled DataFrame."""
        rows = [m.to_dict() for m in self.company_multiples]
        return pd.DataFrame(rows)

    @property
    def target_multiples(self) -> Optional[CompanyMultiples]:
        for m in self.company_multiples:
            if m.is_target:
                return m
        return None


class CompsEngine:
    """
    Comparable Company Analysis engine.

    Fetches sector peers from a curated universe, pulls their key multiples
    via yfinance, and benchmarks the target company against them.
    """

    MAX_PEERS = 8

    def compute(self, data: FinancialData) -> CompsResult:
        """
        Compute comparable analysis for a company.

        Args:
            data: FinancialData for the target company.

        Returns:
            CompsResult with peer multiples and implied valuations.
        """
        ticker = data.ticker
        sector = data.company_info.sector
        industry = data.company_info.industry
        logger.info(f"[CompsEngine] Running comps for {ticker} (Sector: {sector})")

        # Identify peers
        peer_tickers = self._get_peers(ticker, sector)
        logger.info(f"[CompsEngine] Identified {len(peer_tickers)} peers")

        # Build multiples for target + peers
        all_multiples: list[CompanyMultiples] = []

        # Target company first
        target_mult = self._build_multiples(ticker, data.company_info.__dict__, is_target=True)
        all_multiples.append(target_mult)

        # Peers
        for peer in peer_tickers[: self.MAX_PEERS]:
            try:
                peer_info = yf.Ticker(peer).info or {}
                mult = self._build_multiples(peer, peer_info, is_target=False)
                all_multiples.append(mult)
            except Exception as e:
                logger.debug(f"Failed to fetch peer {peer}: {e}")
                continue

        # Compute peer medians (exclude target)
        peers_only = [m for m in all_multiples if not m.is_target]
        median_pe = self._median_multiple([m.pe_ratio for m in peers_only])
        median_ev_ebitda = self._median_multiple([m.ev_ebitda for m in peers_only])
        median_ev_sales = self._median_multiple([m.ev_sales for m in peers_only])

        # Implied valuations from comps
        implied_values = self._compute_implied_values(
            data, median_pe, median_ev_ebitda, median_ev_sales
        )

        # Attractiveness score
        score, rank = self._compute_attractiveness(target_mult, peers_only)

        notes = self._generate_notes(target_mult, peers_only, median_pe)

        result = CompsResult(
            target_ticker=ticker,
            sector=sector,
            industry=industry,
            company_multiples=all_multiples,
            peer_median_pe=median_pe,
            peer_median_ev_ebitda=median_ev_ebitda,
            peer_median_ev_sales=median_ev_sales,
            implied_values=implied_values,
            attractiveness_rank=rank,
            attractiveness_score=score,
            notes=notes,
        )

        logger.success(
            f"[CompsEngine] {ticker}: Attractiveness Score = {score:.1f}/10 (Rank {rank})"
        )
        return result

    def _get_peers(self, target_ticker: str, sector: Optional[str]) -> list[str]:
        """Get sector peers, excluding the target company."""
        if sector and sector in SECTOR_PEERS:
            peers = [t for t in SECTOR_PEERS[sector] if t.upper() != target_ticker.upper()]
        else:
            # Default to large-cap tech as fallback
            peers = [t for t in SECTOR_PEERS["Technology"] if t.upper() != target_ticker.upper()]
        return peers

    def _build_multiples(self, ticker: str, info: dict, is_target: bool) -> CompanyMultiples:
        """Build CompanyMultiples from raw info dict."""

        def safe_get(key: str) -> Optional[float]:
            val = info.get(key)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return None
            return float(val)

        return CompanyMultiples(
            ticker=ticker.upper(),
            name=info.get("longName") or info.get("name") or ticker,
            pe_ratio=safe_get("trailingPE") or safe_get("pe_ratio"),
            forward_pe=safe_get("forwardPE") or safe_get("forward_pe"),
            ev_ebitda=safe_get("enterpriseToEbitda") or safe_get("ev_ebitda"),
            ev_sales=safe_get("enterpriseToRevenue"),
            pb_ratio=safe_get("priceToBook") or safe_get("pb_ratio"),
            peg_ratio=safe_get("pegRatio"),
            market_cap=safe_get("marketCap") or safe_get("market_cap"),
            revenue_growth=safe_get("revenueGrowth") or safe_get("revenue_growth"),
            profit_margin=safe_get("profitMargins") or safe_get("profit_margin"),
            roe=safe_get("returnOnEquity") or safe_get("return_on_equity"),
            is_target=is_target,
        )

    def _median_multiple(self, values: list[Optional[float]]) -> Optional[float]:
        """Compute median of a list, ignoring None and extremes."""
        clean = [v for v in values if v is not None and 0 < v < 200]
        return float(np.median(clean)) if clean else None

    def _compute_implied_values(
        self,
        data: FinancialData,
        median_pe: Optional[float],
        median_ev_ebitda: Optional[float],
        median_ev_sales: Optional[float],
    ) -> dict:
        """Compute implied share price from peer median multiples."""
        result = {}
        info = data.company_info
        income = data.latest_income
        shares = info.shares_outstanding or 1.0
        total_debt = (data.latest_balance.total_debt or 0) if data.latest_balance else 0
        cash = (data.latest_balance.cash_and_equivalents or 0) if data.latest_balance else 0

        if median_pe and income and income.net_income and shares:
            eps = income.net_income / shares
            result["P/E Implied Price"] = median_pe * eps

        if median_ev_ebitda and income and income.ebitda:
            implied_ev = median_ev_ebitda * income.ebitda
            implied_equity = max(0, implied_ev - total_debt + cash)
            result["EV/EBITDA Implied Price"] = implied_equity / shares

        if median_ev_sales and income and income.revenue:
            implied_ev = median_ev_sales * income.revenue
            implied_equity = max(0, implied_ev - total_debt + cash)
            result["EV/Sales Implied Price"] = implied_equity / shares

        return result

    def _compute_attractiveness(
        self, target: CompanyMultiples, peers: list[CompanyMultiples]
    ) -> tuple[float, int]:
        """Score the target company's attractiveness vs peers (0–10 scale)."""
        score = 5.0  # Neutral start

        if not peers:
            return score, 1

        peer_pe = self._median_multiple([m.pe_ratio for m in peers])
        peer_ev = self._median_multiple([m.ev_ebitda for m in peers])

        # Lower PE = better value (if positive)
        if peer_pe and target.pe_ratio and target.pe_ratio > 0:
            if target.pe_ratio < peer_pe * 0.8:
                score += 1.5
            elif target.pe_ratio > peer_pe * 1.2:
                score -= 1.5

        # Higher revenue growth = better
        peer_growth = self._median_multiple([m.revenue_growth for m in peers])
        if peer_growth is not None and target.revenue_growth is not None:
            if target.revenue_growth > peer_growth * 1.1:
                score += 1.5
            elif target.revenue_growth < peer_growth * 0.9:
                score -= 1.0

        # Higher ROE = better
        peer_roe = self._median_multiple([m.roe for m in peers])
        if peer_roe is not None and target.roe is not None:
            if target.roe > peer_roe:
                score += 1.0

        score = max(0.0, min(10.0, score))

        # Rank (1 = most attractive)
        all_scores = [5.0] * len(peers)  # Simplified — target vs neutral peers
        rank = 1  # Simplified ranking

        return score, rank

    def _generate_notes(
        self,
        target: CompanyMultiples,
        peers: list[CompanyMultiples],
        median_pe: Optional[float],
    ) -> list[str]:
        """Generate analyst commentary."""
        notes = []

        if target.pe_ratio and median_pe:
            if target.pe_ratio < median_pe:
                discount = (median_pe - target.pe_ratio) / median_pe
                notes.append(
                    f"Trading at {discount:.0%} discount to peer median P/E "
                    f"({target.pe_ratio:.1f}x vs {median_pe:.1f}x)"
                )
            else:
                premium = (target.pe_ratio - median_pe) / median_pe
                notes.append(
                    f"Trading at {premium:.0%} premium to peer median P/E "
                    f"({target.pe_ratio:.1f}x vs {median_pe:.1f}x)"
                )

        if len(peers) < 3:
            notes.append("Limited peer universe — comps should be interpreted with caution")

        return notes


__all__ = ["CompsEngine", "CompsResult", "CompanyMultiples"]
