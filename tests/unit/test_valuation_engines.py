"""Unit tests for WACC and DCF engines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.finance.engines.dcf_engine import DCFEngine, DCFResult
from src.finance.engines.wacc_engine import WACCBreakdown, WACCEngine
from tests.unit.test_fcff_engine import make_sample_data


class TestWACCEngine:
    def test_compute_returns_breakdown(self) -> None:
        data = make_sample_data()
        result = WACCEngine().compute(data)
        assert isinstance(result, WACCBreakdown)

    def test_wacc_within_reasonable_bounds(self) -> None:
        data = make_sample_data()
        result = WACCEngine().compute(data)
        assert 0.04 <= result.wacc <= 0.30

    def test_weights_sum_to_one(self) -> None:
        data = make_sample_data()
        result = WACCEngine().compute(data)
        assert abs(result.equity_weight + result.debt_weight - 1.0) < 0.001

    def test_capm_formula_string(self) -> None:
        data = make_sample_data()
        result = WACCEngine().compute(data)
        assert "Ke" in result.capm_formula
        assert "=" in result.capm_formula

    def test_override_beta(self) -> None:
        data = make_sample_data()
        r1 = WACCEngine().compute(data, beta_override=0.5)
        r2 = WACCEngine().compute(data, beta_override=2.0)
        assert r1.cost_of_equity < r2.cost_of_equity

    def test_override_risk_free_rate(self) -> None:
        data = make_sample_data()
        r1 = WACCEngine().compute(data, risk_free_rate=0.02)
        r2 = WACCEngine().compute(data, risk_free_rate=0.06)
        assert r1.cost_of_equity < r2.cost_of_equity


class TestDCFEngine:
    def test_compute_returns_result(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data)
        assert isinstance(result, DCFResult)

    def test_intrinsic_value_positive(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data)
        assert result.intrinsic_value_per_share > 0

    def test_projections_length(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data, forecast_years=7)
        assert len(result.projections) == 7

    def test_terminal_value_dominance(self) -> None:
        """Terminal value should be significant portion of enterprise value."""
        data = make_sample_data()
        result = DCFEngine().compute(data)
        tv_pct = result.pv_terminal_value / result.enterprise_value
        assert 0.3 < tv_pct < 0.95  # Typically 60-80%

    def test_recommendation_valid(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data)
        assert result.recommendation in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

    def test_sensitivity_matrix_built(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data, build_sensitivity=True)
        assert result.sensitivity_matrix is not None
        assert result.sensitivity_matrix.shape[0] > 0

    def test_wacc_override(self) -> None:
        data = make_sample_data()
        r_low = DCFEngine().compute(data, wacc_override=0.06)
        r_high = DCFEngine().compute(data, wacc_override=0.14)
        assert r_low.intrinsic_value_per_share > r_high.intrinsic_value_per_share

    def test_equity_value_non_negative(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data)
        assert result.equity_value >= 0

    def test_enterprise_value_components(self) -> None:
        data = make_sample_data()
        result = DCFEngine().compute(data)
        expected_ev = result.pv_fcff_sum + result.pv_terminal_value
        assert abs(result.enterprise_value - expected_ev) < 1.0  # Within $1
