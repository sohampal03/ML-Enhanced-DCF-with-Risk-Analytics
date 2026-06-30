"""
Valuation Orchestrator — Central Service.

Coordinates all engines (FCFF, WACC, DCF, Comps, ML, Monte Carlo, XAI)
into a single, cohesive analysis pipeline.

Usage:
    orchestrator = ValuationOrchestrator()
    report = orchestrator.analyze("AAPL")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.data.repositories.financial_repository import FinancialRepository
from src.data.schemas.financial_schemas import FinancialData
from src.finance.engines.comps_engine import CompsEngine, CompsResult
from src.finance.engines.dcf_engine import DCFEngine, DCFResult
from src.finance.engines.fcff_engine import FCFFEngine, FCFFResult
from src.finance.engines.wacc_engine import WACCBreakdown, WACCEngine
from src.forecasting.models.revenue_forecaster import (
    ForecastResult,
    MarginForecastingPipeline,
    RevenueForecastingPipeline,
)
from src.forecasting.models.valuation_classifier import (
    ClassificationResult,
    ValuationClassifier,
)
from src.simulation.monte_carlo import MonteCarloResult, MonteCarloSimulator


@dataclass
class FullValuationReport:
    """
    Complete output of the ValuationOrchestrator.

    Contains all sub-engine results for the dashboard to consume.
    """

    ticker: str
    company_name: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Raw data
    financial_data: FinancialData | None = None

    # Financial engines
    fcff_result: FCFFResult | None = None
    wacc_result: WACCBreakdown | None = None
    dcf_result: DCFResult | None = None
    comps_result: CompsResult | None = None

    # ML
    revenue_forecast: ForecastResult | None = None
    margin_forecasts: dict[str, ForecastResult] | None = None
    classification: ClassificationResult | None = None

    # Risk
    monte_carlo: MonteCarloResult | None = None

    # Errors
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True if core valuation engines succeeded."""
        return self.dcf_result is not None and self.fcff_result is not None

    @property
    def current_price(self) -> float | None:
        return self.financial_data.company_info.current_price if self.financial_data else None

    @property
    def intrinsic_value(self) -> float | None:
        return self.dcf_result.intrinsic_value_per_share if self.dcf_result else None

    @property
    def recommendation(self) -> str:
        return self.dcf_result.recommendation if self.dcf_result else "N/A"


class ValuationOrchestrator:
    """
    Central orchestrator for the AI Business Valuation Platform.

    Runs all valuation engines in sequence with graceful error handling —
    a failure in one engine (e.g., ML) does not prevent others from running.
    """

    def __init__(self) -> None:
        self.repository = FinancialRepository()
        self.fcff_engine = FCFFEngine()
        self.wacc_engine = WACCEngine()
        self.dcf_engine = DCFEngine()
        self.comps_engine = CompsEngine()
        self.revenue_forecaster = RevenueForecastingPipeline()
        self.margin_forecaster = MarginForecastingPipeline()
        self.classifier = ValuationClassifier()
        self.mc_simulator = MonteCarloSimulator()
        logger.info("ValuationOrchestrator initialized")

    def analyze(
        self,
        ticker: str,
        force_refresh: bool = False,
        run_ml: bool = True,
        run_monte_carlo: bool = True,
        run_comps: bool = True,
        # DCF overrides
        wacc_override: float | None = None,
        terminal_growth_rate: float | None = None,
        forecast_years: int = 10,
        revenue_growth_override: float | None = None,
        ebit_margin_override: float | None = None,
    ) -> FullValuationReport:
        """
        Run complete valuation analysis for a ticker.

        Args:
            ticker: Stock ticker symbol.
            force_refresh: Bypass all caches.
            run_ml: Whether to run ML forecasting.
            run_monte_carlo: Whether to run Monte Carlo.
            run_comps: Whether to run comparable analysis.
            wacc_override: Override WACC (decimal).
            terminal_growth_rate: Override TGR.
            forecast_years: DCF explicit forecast horizon.
            revenue_growth_override: Override revenue growth assumption.
            ebit_margin_override: Override EBIT margin assumption.

        Returns:
            FullValuationReport with all results.
        """
        ticker = ticker.upper().strip()
        logger.info(f"═══ Starting full analysis for {ticker} ═══")

        report = FullValuationReport(ticker=ticker, company_name=ticker)

        # ── Step 1: Fetch Data ────────────────────────────────────────────────
        try:
            data = self.repository.get_financial_data(ticker, force_refresh=force_refresh)
            report.financial_data = data
            report.company_name = data.company_info.name
            logger.info(f"✓ Data fetched: {len(data.income_statements)} years")
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            report.errors["data"] = str(e)
            return report

        # ── Step 2: FCFF Engine ───────────────────────────────────────────────
        try:
            report.fcff_result = self.fcff_engine.compute(data)
            logger.info("✓ FCFF computed")
        except Exception as e:
            logger.error(f"FCFF failed: {e}")
            report.errors["fcff"] = str(e)

        # ── Step 3: WACC Engine ───────────────────────────────────────────────
        try:
            report.wacc_result = self.wacc_engine.compute(data)
            logger.info(f"✓ WACC = {report.wacc_result.wacc:.2%}")
        except Exception as e:
            logger.error(f"WACC failed: {e}")
            report.errors["wacc"] = str(e)

        # ── Step 4: DCF Engine ────────────────────────────────────────────────
        try:
            # Build growth schedule if override provided
            growth_rates = None
            if revenue_growth_override is not None:
                # Declining from override to 3%
                import numpy as np

                growth_rates = list(np.linspace(revenue_growth_override, 0.03, forecast_years))

            report.dcf_result = self.dcf_engine.compute(
                data=data,
                fcff_result=report.fcff_result,
                wacc_result=report.wacc_result,
                wacc_override=wacc_override,
                terminal_growth_rate=terminal_growth_rate,
                forecast_years=forecast_years,
                revenue_growth_rates=growth_rates,
                ebit_margin=ebit_margin_override,
                build_sensitivity=True,
            )
            logger.info(
                f"✓ DCF: IV = ${report.dcf_result.intrinsic_value_per_share:,.2f} "
                f"({report.dcf_result.recommendation})"
            )
        except Exception as e:
            logger.error(f"DCF failed: {e}")
            report.errors["dcf"] = str(e)

        # ── Step 5: Comparable Analysis ───────────────────────────────────────
        if run_comps:
            try:
                report.comps_result = self.comps_engine.compute(data)
                logger.info(f"✓ Comps: {len(report.comps_result.company_multiples)} companies")
            except Exception as e:
                logger.warning(f"Comps failed (non-critical): {e}")
                report.errors["comps"] = str(e)

        # ── Step 6: ML Forecasting ────────────────────────────────────────────
        if run_ml:
            try:
                report.revenue_forecast = self.revenue_forecaster.forecast(data, target="revenue")
                logger.info(f"✓ Revenue forecast: best={report.revenue_forecast.best_model_name}")
            except Exception as e:
                logger.warning(f"Revenue forecast failed: {e}")
                report.errors["revenue_forecast"] = str(e)

            try:
                report.margin_forecasts = self.margin_forecaster.forecast_all_margins(data)
                logger.info(f"✓ Margin forecasts: {list(report.margin_forecasts.keys())}")
            except Exception as e:
                logger.warning(f"Margin forecast failed: {e}")
                report.errors["margin_forecast"] = str(e)

        # ── Step 7: Valuation Classification ─────────────────────────────────
        if report.dcf_result:
            try:
                info = data.company_info
                mc = report.comps_result
                report.classification = self.classifier.classify_current(
                    margin_of_safety=report.dcf_result.margin_of_safety,
                    pe_ratio=info.pe_ratio,
                    peer_median_pe=mc.peer_median_pe if mc else None,
                    revenue_growth=info.revenue_growth,
                    fcf_yield=(
                        report.dcf_result.pv_fcff_sum / max(info.market_cap or 1, 1)
                        if info.market_cap
                        else None
                    ),
                    wacc=report.dcf_result.wacc,
                    ev_ebitda=info.ev_ebitda,
                    net_margin=info.profit_margin,
                    roe=info.return_on_equity,
                    debt_to_equity=info.debt_to_equity,
                    ticker=ticker,
                )
                logger.info(f"✓ Classification: {report.classification.predicted_label}")
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                report.errors["classification"] = str(e)

        # ── Step 8: Monte Carlo Simulation ────────────────────────────────────
        if run_monte_carlo and report.dcf_result:
            try:
                report.monte_carlo = self.mc_simulator.run(
                    dcf_result=report.dcf_result,
                    data=data,
                )
                logger.info(
                    f"✓ Monte Carlo: P(undervalued)={report.monte_carlo.prob_undervalued:.1%}"
                )
            except Exception as e:
                logger.warning(f"Monte Carlo failed: {e}")
                report.errors["monte_carlo"] = str(e)

        logger.success(f"═══ Analysis complete for {ticker} ═══")
        return report


__all__ = ["ValuationOrchestrator", "FullValuationReport"]
