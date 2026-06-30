"""
SHAP Explainability Engine.

Computes SHAP values for ML model predictions and generates
natural-language explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available — explainability features disabled")


@dataclass
class SHAPResult:
    """Output of the SHAP explainability engine."""

    ticker: str
    model_name: str
    feature_names: list[str]
    shap_values: Optional[np.ndarray]  # shape: (n_samples, n_features)
    base_value: Optional[float]  # Model's expected value
    feature_importances: dict[str, float]  # Global mean |SHAP|
    top_positive_drivers: list[dict]  # Features increasing value
    top_negative_drivers: list[dict]  # Features decreasing value
    natural_language_explanation: str
    notes: list[str] = field(default_factory=list)

    def to_importance_df(self) -> pd.DataFrame:
        """Return global feature importances as a DataFrame."""
        return pd.DataFrame(
            [
                {"Feature": k, "Mean |SHAP|": v}
                for k, v in sorted(self.feature_importances.items(), key=lambda x: -x[1])
            ]
        )


class SHAPExplainer:
    """
    SHAP-based model explanation engine.

    Supports:
      - TreeExplainer (XGBoost, LightGBM, CatBoost, Random Forest)
      - LinearExplainer (Linear Regression)
      - KernelExplainer (fallback for any model)

    Generates:
      - Global feature importance (mean |SHAP| values)
      - Local explanations (force plots for single predictions)
      - Natural language summaries
    """

    def explain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_explain: Optional[pd.DataFrame] = None,
        model_name: str = "ML Model",
        ticker: str = "TICKER",
        target: str = "revenue",
    ) -> SHAPResult:
        """
        Compute SHAP values and generate explanations.

        Args:
            model: Trained scikit-learn / XGBoost / LightGBM model.
            X_train: Training feature matrix (used for background).
            X_explain: Data to explain (uses last row of X_train if None).
            model_name: Human-readable model name.
            ticker: Ticker symbol for labeling.
            target: Target variable name.

        Returns:
            SHAPResult with values and natural language explanation.
        """
        if not HAS_SHAP:
            return self._dummy_result(ticker, model_name, X_train)

        logger.info(f"[SHAPExplainer] Computing SHAP for {ticker} ({model_name})")

        feature_names = X_train.columns.tolist()
        X_explain = X_explain if X_explain is not None else X_train.iloc[[-1]]

        try:
            explainer, shap_values = self._compute_shap(model, X_train, X_explain)
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return self._dummy_result(ticker, model_name, X_train)

        # Global importances: mean |SHAP| across all explanation rows
        if shap_values.ndim == 2:
            global_importance = np.abs(shap_values).mean(axis=0)
        else:
            global_importance = np.abs(shap_values)

        feature_importances = dict(zip(feature_names, global_importance.tolist()))

        # Top drivers for the LAST row (most recent prediction)
        last_shap = shap_values[-1] if shap_values.ndim == 2 else shap_values
        base_value = (
            float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
        )

        drivers = sorted(
            zip(feature_names, last_shap.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_positive = [
            {
                "feature": name,
                "shap_value": val,
                "pct_contribution": abs(val) / (sum(abs(v) for _, v in drivers) + 1e-10),
            }
            for name, val in drivers
            if val > 0
        ][:5]

        top_negative = [
            {
                "feature": name,
                "shap_value": val,
                "pct_contribution": abs(val) / (sum(abs(v) for _, v in drivers) + 1e-10),
            }
            for name, val in drivers
            if val < 0
        ][:5]

        nl_explanation = self._generate_narrative(
            target, top_positive, top_negative, feature_importances
        )

        result = SHAPResult(
            ticker=ticker,
            model_name=model_name,
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            feature_importances=feature_importances,
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            natural_language_explanation=nl_explanation,
        )

        logger.success(f"[SHAPExplainer] {ticker}: Explanation generated for {model_name}")
        return result

    def _compute_shap(
        self, model: Any, X_train: pd.DataFrame, X_explain: pd.DataFrame
    ) -> tuple[Any, np.ndarray]:
        """Select the most appropriate SHAP explainer."""
        model_class = type(model).__name__

        if "XGB" in model_class or "LGBM" in model_class or "CatBoost" in model_class:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain.fillna(0).values)

        elif "RandomForest" in model_class or "GradientBoosting" in model_class:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain.fillna(0).values)

        elif "Ridge" in model_class or "Linear" in model_class:
            background = shap.maskers.Independent(X_train.fillna(0).values, max_samples=50)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_explain.fillna(0).values)

        else:
            # Kernel explainer as fallback (slower)
            background = shap.sample(X_train.fillna(0).values, 50)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_explain.fillna(0).values)

        return explainer, np.array(shap_values)

    def _generate_narrative(
        self,
        target: str,
        positives: list[dict],
        negatives: list[dict],
        importances: dict,
    ) -> str:
        """Generate a natural-language explanation from SHAP values."""
        target_label = target.replace("_", " ").title()
        lines = [f"**Why did the model predict this {target_label}?**\n"]

        lines.append("**Key Drivers (Increasing prediction):**")
        for driver in positives[:3]:
            pct = driver["pct_contribution"] * 100
            name = driver["feature"].replace("_", " ").title()
            lines.append(f"  • {name}: contributed +{pct:.1f}% to the forecast")

        if negatives:
            lines.append("\n**Key Headwinds (Decreasing prediction):**")
            for driver in negatives[:3]:
                pct = driver["pct_contribution"] * 100
                name = driver["feature"].replace("_", " ").title()
                lines.append(f"  • {name}: pulled forecast down by {pct:.1f}%")

        if importances:
            top_feature = max(importances, key=importances.get)  # type: ignore
            lines.append(
                f"\n**Most influential overall feature:** {top_feature.replace('_', ' ').title()}"
            )

        return "\n".join(lines)

    def _dummy_result(self, ticker: str, model_name: str, X_train: pd.DataFrame) -> SHAPResult:
        """Return empty result when SHAP is unavailable."""
        return SHAPResult(
            ticker=ticker,
            model_name=model_name,
            feature_names=X_train.columns.tolist(),
            shap_values=None,
            base_value=None,
            feature_importances={},
            top_positive_drivers=[],
            top_negative_drivers=[],
            natural_language_explanation="SHAP library not installed. Install with: pip install shap",
            notes=["SHAP not available"],
        )


__all__ = ["SHAPExplainer", "SHAPResult"]
