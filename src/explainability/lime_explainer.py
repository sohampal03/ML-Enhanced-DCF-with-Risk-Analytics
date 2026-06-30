"""
LIME (Local Interpretable Model-agnostic Explanations) wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

try:
    from lime import lime_tabular

    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.warning("LIME not available")


@dataclass
class LIMEResult:
    """Output of the LIME explainability engine."""

    ticker: str
    model_name: str
    feature_names: list[str]
    lime_importances: dict
    explanation_text: str
    notes: list[str] = field(default_factory=list)


class LIMEExplainer:
    """LIME-based local model explanation."""

    def explain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        instance: pd.Series,
        model_name: str = "ML Model",
        ticker: str = "TICKER",
    ) -> LIMEResult:
        if not HAS_LIME:
            return LIMEResult(
                ticker=ticker,
                model_name=model_name,
                feature_names=X_train.columns.tolist(),
                lime_importances={},
                explanation_text="LIME not installed. Run: pip install lime",
                notes=["LIME not available"],
            )

        try:
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train.fillna(0).values,
                feature_names=X_train.columns.tolist(),
                mode="regression",
                random_state=42,
            )
            exp = explainer.explain_instance(
                instance.fillna(0).values,
                model.predict,
                num_features=10,
            )
            importances = dict(exp.as_list())
            text = self._narrative(importances, ticker)

            return LIMEResult(
                ticker=ticker,
                model_name=model_name,
                feature_names=X_train.columns.tolist(),
                lime_importances=importances,
                explanation_text=text,
            )
        except Exception as e:
            logger.error(f"LIME failed: {e}")
            return LIMEResult(
                ticker=ticker,
                model_name=model_name,
                feature_names=X_train.columns.tolist(),
                lime_importances={},
                explanation_text=f"LIME explanation failed: {e}",
                notes=[str(e)],
            )

    def _narrative(self, importances: dict, ticker: str) -> str:
        lines = [f"**LIME Explanation for {ticker}:**"]
        for feat, val in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            direction = "increased" if val > 0 else "decreased"
            lines.append(f"  • {feat}: {direction} prediction by {abs(val):.4f}")
        return "\n".join(lines)


__all__ = ["LIMEExplainer", "LIMEResult"]
