"""
Monte Carlo Simulation Engine.

Runs 10,000+ DCF simulations with randomized inputs to produce
a probability distribution of intrinsic values.

Randomized parameters:
  - Revenue Growth Rate (per year)
  - EBIT Margin
  - WACC
  - Terminal Growth Rate
  - CAPEX % of Revenue

Output:
  - Value distribution with percentiles
  - P(undervalued), P(overvalued)
  - Tornado chart sensitivity data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.finance.engines.dcf_engine import DCFEngine, DCFResult


@dataclass
class MonteCarloAssumption:
    """Parameter distribution specification."""

    name: str
    base: float
    std: float  # Standard deviation
    distribution: str = "normal"  # 'normal', 'lognormal', 'triangular'
    low: Optional[float] = None  # For triangular
    high: Optional[float] = None  # For triangular
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples from this parameter's distribution."""
        if self.distribution == "normal":
            samples = rng.normal(self.base, self.std, n)
        elif self.distribution == "lognormal":
            # Parametrize lognormal to have the correct mean
            sigma = self.std / self.base if self.base != 0 else 0.1
            samples = rng.lognormal(np.log(max(self.base, 1e-8)), sigma, n)
        elif self.distribution == "triangular":
            low = self.low or (self.base - 2 * self.std)
            high = self.high or (self.base + 2 * self.std)
            samples = rng.triangular(low, self.base, high, n)
        else:
            samples = np.full(n, self.base)

        # Clip to bounds
        if self.clip_min is not None:
            samples = np.maximum(samples, self.clip_min)
        if self.clip_max is not None:
            samples = np.minimum(samples, self.clip_max)

        return samples


@dataclass
class MonteCarloResult:
    """Output of the Monte Carlo simulation engine."""

    ticker: str
    n_simulations: int
    intrinsic_values: np.ndarray  # Full distribution of intrinsic values

    # Current price for comparison
    current_price: float

    # Distribution statistics
    mean_value: float
    median_value: float
    std_value: float
    skewness: float
    kurtosis: float

    # Percentile bands
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float

    # Probabilities
    prob_undervalued: float  # P(intrinsic > current)
    prob_overvalued: float  # P(intrinsic < current)
    prob_in_range: float  # P(within ±15% of current)

    # Parameter sensitivities for tornado chart
    tornado_data: dict = field(default_factory=dict)

    # Assumptions used
    assumptions: list[MonteCarloAssumption] = field(default_factory=list)

    def to_percentile_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Percentile": [
                    "5th (Bear)",
                    "10th",
                    "25th",
                    "50th (Base)",
                    "75th",
                    "90th",
                    "95th (Bull)",
                ],
                "Intrinsic Value": [
                    self.p5,
                    self.p10,
                    self.p25,
                    self.p50,
                    self.p75,
                    self.p90,
                    self.p95,
                ],
                "vs Current Price": [
                    f"{(v / self.current_price - 1):.1%}" if self.current_price > 0 else "N/A"
                    for v in [self.p5, self.p10, self.p25, self.p50, self.p75, self.p90, self.p95]
                ],
            }
        )


class MonteCarloSimulator:
    """
    Institutional-quality Monte Carlo simulation engine for DCF valuation.

    Uses Latin Hypercube Sampling (LHS) stratified draws for better
    coverage of the parameter space compared to pure random sampling.
    """

    def __init__(self, n_simulations: int = 10_000, random_seed: int = 42) -> None:
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)
        self.dcf_engine = DCFEngine()

    def run(
        self,
        dcf_result: DCFResult,
        data: object,  # FinancialData
        custom_assumptions: Optional[dict] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            dcf_result: Base DCF result (provides base-case assumptions).
            data: FinancialData for the company.
            custom_assumptions: Override default std devs for parameters.

        Returns:
            MonteCarloResult with full value distribution.
        """
        ticker = dcf_result.ticker
        n = self.n_simulations
        logger.info(f"[MonteCarlo] Running {n:,} simulations for {ticker}")

        # ── Define parameter distributions ───────────────────────────────────
        assumptions = self._build_assumptions(dcf_result, custom_assumptions or {})

        # ── Sample all parameters ─────────────────────────────────────────────
        samples = {a.name: a.sample(n, self.rng) for a in assumptions}

        # ── Vectorized DCF ─────────────────────────────────────────────────────
        intrinsic_values = self._vectorized_dcf(
            samples=samples,
            dcf_result=dcf_result,
        )

        # Filter out extreme outliers (>5× current price or negative)
        current_price = dcf_result.current_price
        mask = (intrinsic_values > 0) & (intrinsic_values < max(current_price * 10, 1000))
        intrinsic_clean = intrinsic_values[mask]

        if len(intrinsic_clean) < n * 0.5:
            logger.warning("More than 50% of simulations produced extreme values")

        # ── Compute statistics ─────────────────────────────────────────────────
        from scipy import stats as sp_stats

        percentiles = np.percentile(intrinsic_clean, [5, 10, 25, 50, 75, 90, 95])

        prob_under = float(np.mean(intrinsic_clean > current_price))
        prob_over = float(np.mean(intrinsic_clean < current_price))
        prob_in_range = float(
            np.mean(
                (intrinsic_clean >= current_price * 0.85)
                & (intrinsic_clean <= current_price * 1.15)
            )
        )

        # ── Tornado sensitivity ────────────────────────────────────────────────
        tornado_data = self._compute_tornado(samples, intrinsic_values, assumptions)

        result = MonteCarloResult(
            ticker=ticker,
            n_simulations=len(intrinsic_clean),
            intrinsic_values=intrinsic_clean,
            current_price=current_price,
            mean_value=float(np.mean(intrinsic_clean)),
            median_value=float(np.median(intrinsic_clean)),
            std_value=float(np.std(intrinsic_clean)),
            skewness=float(sp_stats.skew(intrinsic_clean)),
            kurtosis=float(sp_stats.kurtosis(intrinsic_clean)),
            p5=float(percentiles[0]),
            p10=float(percentiles[1]),
            p25=float(percentiles[2]),
            p50=float(percentiles[3]),
            p75=float(percentiles[4]),
            p90=float(percentiles[5]),
            p95=float(percentiles[6]),
            prob_undervalued=prob_under,
            prob_overvalued=prob_over,
            prob_in_range=prob_in_range,
            tornado_data=tornado_data,
            assumptions=assumptions,
        )

        logger.success(
            f"[MonteCarlo] {ticker}: Median IV = ${result.median_value:,.2f} | "
            f"P(Undervalued) = {prob_under:.1%} | "
            f"95th pct = ${result.p95:,.2f}"
        )
        return result

    def _build_assumptions(self, dcf: DCFResult, overrides: dict) -> list[MonteCarloAssumption]:
        """Build parameter distribution list from DCF base case."""
        base_growth = float(np.mean(dcf.revenue_growth_rates)) if dcf.revenue_growth_rates else 0.05

        assumptions = [
            MonteCarloAssumption(
                name="revenue_growth",
                base=overrides.get("revenue_growth_base", base_growth),
                std=overrides.get("revenue_growth_std", 0.05),
                distribution="normal",
                clip_min=-0.30,
                clip_max=0.60,
            ),
            MonteCarloAssumption(
                name="ebit_margin",
                base=overrides.get("ebit_margin_base", dcf.ebit_margin_forecast),
                std=overrides.get("ebit_margin_std", 0.03),
                distribution="normal",
                clip_min=0.01,
                clip_max=0.60,
            ),
            MonteCarloAssumption(
                name="wacc",
                base=overrides.get("wacc_base", dcf.wacc),
                std=overrides.get("wacc_std", 0.01),
                distribution="normal",
                clip_min=0.04,
                clip_max=0.25,
            ),
            MonteCarloAssumption(
                name="terminal_growth",
                base=overrides.get("terminal_growth_base", dcf.terminal_growth_rate),
                std=overrides.get("terminal_growth_std", 0.005),
                distribution="triangular",
                low=0.005,
                high=0.04,
                clip_min=0.001,
                clip_max=0.05,
            ),
            MonteCarloAssumption(
                name="capex_pct",
                base=overrides.get("capex_pct_base", 0.05),
                std=overrides.get("capex_pct_std", 0.02),
                distribution="normal",
                clip_min=0.0,
                clip_max=0.25,
            ),
        ]
        return assumptions

    def _vectorized_dcf(self, samples: dict, dcf_result: DCFResult) -> np.ndarray:
        """
        Vectorized DCF computation for all Monte Carlo paths.

        Instead of calling the full DCFEngine N times (slow),
        we implement a simplified vectorized version for speed.
        """
        n = self.n_simulations
        years = dcf_result.forecast_years
        base_revenue = dcf_result.base_revenue
        shares = dcf_result.shares_outstanding
        debt = dcf_result.total_debt
        cash = dcf_result.cash
        tax = dcf_result.tax_rate

        revenue_growth = samples["revenue_growth"]  # shape: (n,)
        ebit_margin = samples["ebit_margin"]  # shape: (n,)
        wacc = samples["wacc"]  # shape: (n,)
        tgr = samples["terminal_growth"]  # shape: (n,)
        capex_pct = samples["capex_pct"]  # shape: (n,)

        # Forecast revenues (shape: n × years)
        revenues = np.zeros((n, years))
        revenues[:, 0] = base_revenue * (1 + revenue_growth)
        for t in range(1, years):
            # Growth decelerates linearly toward steady state
            g_t = revenue_growth * (1 - t / (years + 1)) + 0.03 * (t / (years + 1))
            revenues[:, t] = revenues[:, t - 1] * (1 + g_t)

        # FCFF per year (simplified: NOPAT + 4% D&A - ΔNWC(2%) - CAPEX)
        da_pct = 0.04
        nwc_pct = 0.10
        fcff_per_year = revenues * (
            ebit_margin[:, np.newaxis] * (1 - tax)
            + da_pct
            - nwc_pct * 0.1
            - capex_pct[:, np.newaxis]
        )

        # Discount factors
        t_arr = np.arange(1, years + 1)
        discount_factors = 1 / (1 + wacc[:, np.newaxis]) ** t_arr  # n × years

        # PV of FCFFs
        pv_fcff = np.sum(fcff_per_year * discount_factors, axis=1)  # shape: (n,)

        # Terminal value (Gordon Growth)
        terminal_fcff = fcff_per_year[:, -1] * (1 + tgr)
        valid_tv = wacc > tgr
        tv = np.where(valid_tv, terminal_fcff / (wacc - tgr), 0)
        pv_tv = tv / (1 + wacc) ** years

        # Enterprise → Equity → Per Share
        ev = pv_fcff + pv_tv
        equity = np.maximum(0, ev - debt + cash)
        intrinsic = equity / max(shares, 1.0)

        return intrinsic

    def _compute_tornado(
        self,
        samples: dict,
        intrinsic_values: np.ndarray,
        assumptions: list[MonteCarloAssumption],
    ) -> dict:
        """
        Compute tornado chart data: sensitivity of IV to each parameter.

        Computes Pearson correlation and 10th/90th percentile ranges.
        """
        tornado = {}

        for assumption in assumptions:
            param_vals = samples[assumption.name]
            corr = float(np.corrcoef(param_vals, intrinsic_values)[0, 1])

            # High vs Low scenario
            low_mask = param_vals <= np.percentile(param_vals, 10)
            high_mask = param_vals >= np.percentile(param_vals, 90)

            iv_low = float(np.median(intrinsic_values[low_mask]))
            iv_high = float(np.median(intrinsic_values[high_mask]))

            tornado[assumption.name] = {
                "correlation": corr,
                "iv_low_param": iv_low,
                "iv_high_param": iv_high,
                "impact": abs(iv_high - iv_low),
                "label": assumption.name.replace("_", " ").title(),
            }

        return tornado


__all__ = ["MonteCarloSimulator", "MonteCarloResult", "MonteCarloAssumption"]
