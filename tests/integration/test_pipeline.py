"""
Integration test: full valuation pipeline end-to-end.
Uses synthetic data to avoid network calls.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.valuation.orchestrator import ValuationOrchestrator, FullValuationReport
from tests.unit.test_fcff_engine import make_sample_data


class MockRepository:
    """Repository that returns synthetic data without network calls."""

    def get_financial_data(self, ticker, **kwargs):
        return make_sample_data(ticker)


class TestFullPipeline:
    """Integration tests for the complete valuation pipeline."""

    def _make_orchestrator_with_mock(self) -> ValuationOrchestrator:
        orch = ValuationOrchestrator()
        orch.repository = MockRepository()
        return orch

    def test_full_pipeline_completes(self):
        orch = self._make_orchestrator_with_mock()
        report = orch.analyze("TEST", run_ml=False, run_monte_carlo=False, run_comps=False)
        assert report.is_complete
        assert report.dcf_result is not None
        assert report.fcff_result is not None
        assert report.wacc_result is not None

    def test_pipeline_with_ml(self):
        orch = self._make_orchestrator_with_mock()
        report = orch.analyze("TEST", run_ml=True, run_monte_carlo=False, run_comps=False)
        # ML may fail with synthetic data but pipeline should not crash
        assert isinstance(report, FullValuationReport)

    def test_pipeline_with_monte_carlo(self):
        orch = self._make_orchestrator_with_mock()
        report = orch.analyze("TEST", run_ml=False, run_monte_carlo=True, run_comps=False)
        if report.monte_carlo:
            mc = report.monte_carlo
            assert mc.n_simulations > 100
            assert 0 <= mc.prob_undervalued <= 1
            assert 0 <= mc.prob_overvalued <= 1
            assert abs(mc.prob_undervalued + mc.prob_in_range + mc.prob_overvalued - 1.0) < 0.1

    def test_wacc_override_propagates(self):
        orch = self._make_orchestrator_with_mock()
        r1 = orch.analyze(
            "TEST", wacc_override=0.06, run_ml=False, run_monte_carlo=False, run_comps=False
        )
        r2 = orch.analyze(
            "TEST", wacc_override=0.15, run_ml=False, run_monte_carlo=False, run_comps=False
        )
        if r1.dcf_result and r2.dcf_result:
            assert r1.dcf_result.intrinsic_value_per_share > r2.dcf_result.intrinsic_value_per_share

    def test_errors_do_not_crash_pipeline(self):
        """Pipeline should handle individual module errors gracefully."""
        orch = self._make_orchestrator_with_mock()
        # Even with all modules running, should not raise
        report = orch.analyze("TEST", run_ml=True, run_monte_carlo=True, run_comps=True)
        assert isinstance(report, FullValuationReport)
        # Core should succeed
        assert report.dcf_result is not None

    def test_recommendation_consistency(self):
        orch = self._make_orchestrator_with_mock()
        report = orch.analyze("TEST", run_ml=False, run_monte_carlo=False, run_comps=False)
        if report.dcf_result:
            dcf = report.dcf_result
            if dcf.upside_potential > 0.30:
                assert dcf.recommendation in ["STRONG BUY", "BUY"]
            elif dcf.upside_potential < -0.30:
                assert dcf.recommendation in ["STRONG SELL", "SELL"]
