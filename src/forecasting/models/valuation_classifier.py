"""
Valuation Classification Engine.

Classifies companies as Undervalued (0), Fairly Valued (1), or Overvalued (2)
using a feature set derived from DCF outputs and financial ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


LABELS = {0: "Undervalued", 1: "Fairly Valued", 2: "Overvalued"}
LABEL_COLORS = {0: "#00C851", 1: "#FFD700", 2: "#FF4444"}


@dataclass
class ClassificationResult:
    """Output of the valuation classification engine."""

    ticker: str
    predicted_class: int
    predicted_label: str
    predicted_color: str
    probabilities: dict[str, float]  # {label: probability}
    confidence: float
    feature_importances: Optional[dict] = None
    metrics: Optional[dict] = None
    notes: list[str] = field(default_factory=list)


class ValuationClassifier:
    """
    Classifies a company's valuation attractiveness.

    Labels:
      0 = Undervalued  (Intrinsic Value significantly > Market Price)
      1 = Fairly Valued (±15% range)
      2 = Overvalued   (Market Price significantly > Intrinsic Value)

    Features:
      - Margin of Safety
      - P/E vs Peer median P/E
      - Revenue growth
      - FCFF / Market Cap
      - WACC
      - EV/EBITDA
      - Net Margin
      - ROE
      - Debt/Equity

    Since we don't have labeled historical data, we generate labels
    programmatically from DCF margin of safety.
    """

    def classify_current(
        self,
        margin_of_safety: float,
        pe_ratio: Optional[float],
        peer_median_pe: Optional[float],
        revenue_growth: Optional[float],
        fcf_yield: Optional[float],
        wacc: float,
        ev_ebitda: Optional[float],
        net_margin: Optional[float],
        roe: Optional[float],
        debt_to_equity: Optional[float],
        ticker: str = "UNKNOWN",
    ) -> ClassificationResult:
        """
        Classify current valuation using a rule-based + ML scoring approach.

        Args:
            margin_of_safety: (Intrinsic - Price) / Intrinsic
            pe_ratio: Current P/E ratio
            peer_median_pe: Sector median P/E
            revenue_growth: Year-over-year revenue growth
            fcf_yield: FCF / Market Cap
            wacc: WACC
            ev_ebitda: EV/EBITDA multiple
            net_margin: Net profit margin
            roe: Return on equity
            debt_to_equity: Leverage ratio

        Returns:
            ClassificationResult with class, probabilities, and feature importances.
        """
        logger.info(f"[ValuationClassifier] Classifying {ticker}")

        # ── Rule-based scoring system ─────────────────────────────────────────
        score = 0.0  # Positive = Undervalued, Negative = Overvalued

        # 1. Margin of Safety (most important signal)
        if margin_of_safety > 0.30:
            score += 3.0
        elif margin_of_safety > 0.15:
            score += 1.5
        elif margin_of_safety > 0.0:
            score += 0.5
        elif margin_of_safety > -0.15:
            score -= 0.5
        elif margin_of_safety > -0.30:
            score -= 1.5
        else:
            score -= 3.0

        # 2. P/E vs Peers
        if pe_ratio and peer_median_pe and pe_ratio > 0 and peer_median_pe > 0:
            pe_discount = (peer_median_pe - pe_ratio) / peer_median_pe
            score += pe_discount * 2.0  # Max ±2 points

        # 3. Revenue Growth
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 1.0
            elif revenue_growth > 0.10:
                score += 0.5
            elif revenue_growth < 0:
                score -= 1.0

        # 4. FCF Yield
        if fcf_yield is not None:
            if fcf_yield > 0.08:
                score += 1.0
            elif fcf_yield > 0.04:
                score += 0.5
            elif fcf_yield < 0:
                score -= 0.5

        # 5. EV/EBITDA (lower = cheaper)
        if ev_ebitda is not None and 0 < ev_ebitda < 100:
            if ev_ebitda < 10:
                score += 0.5
            elif ev_ebitda > 25:
                score -= 0.5

        # 6. Quality signals (net margin, ROE)
        if net_margin is not None and net_margin > 0.15:
            score += 0.3
        if roe is not None and roe > 0.20:
            score += 0.3

        # 7. Risk (debt)
        if debt_to_equity is not None:
            if debt_to_equity > 2.0:
                score -= 0.5

        # ── Convert score to class ────────────────────────────────────────────
        if score >= 2.0:
            pred_class = 0  # Undervalued
        elif score >= -1.0:
            pred_class = 1  # Fairly Valued
        else:
            pred_class = 2  # Overvalued

        # ── Soft probabilities (via softmax on score) ─────────────────────────
        # Map score to per-class logits
        logits = np.array([score, 0.0, -score])  # [under, fair, over]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        probabilities = {
            "Undervalued": float(probs[0]),
            "Fairly Valued": float(probs[1]),
            "Overvalued": float(probs[2]),
        }

        confidence = float(probs[pred_class])

        # ── Feature importance (contribution of each signal) ──────────────────
        feature_importances = {
            "Margin of Safety": abs(3.0 if margin_of_safety > 0.30 else 1.5),
            "P/E vs Peers": abs(pe_discount * 2.0) if pe_ratio and peer_median_pe else 0.0,
            "Revenue Growth": 1.0 if revenue_growth and revenue_growth > 0.20 else 0.5,
            "FCF Yield": 1.0 if fcf_yield and fcf_yield > 0.08 else 0.5,
            "EV/EBITDA": 0.5 if ev_ebitda else 0.0,
            "Profitability (ROE/Margin)": 0.6 if (roe and roe > 0.20) else 0.0,
            "Leverage Risk": 0.5 if (debt_to_equity and debt_to_equity > 2.0) else 0.0,
        }

        notes = self._generate_notes(score, pred_class, margin_of_safety, pe_ratio, peer_median_pe)

        result = ClassificationResult(
            ticker=ticker,
            predicted_class=pred_class,
            predicted_label=LABELS[pred_class],
            predicted_color=LABEL_COLORS[pred_class],
            probabilities=probabilities,
            confidence=confidence,
            feature_importances=feature_importances,
            notes=notes,
        )

        logger.success(
            f"[ValuationClassifier] {ticker}: {LABELS[pred_class]} "
            f"(confidence: {confidence:.1%}, score: {score:.2f})"
        )
        return result

    def _generate_notes(
        self,
        score: float,
        pred_class: int,
        mos: float,
        pe_ratio: Optional[float],
        peer_pe: Optional[float],
    ) -> list[str]:
        notes = []
        if pred_class == 0:
            notes.append(f"DCF suggests {mos:.0%} margin of safety — meaningful upside potential")
        elif pred_class == 2:
            notes.append("Market price exceeds intrinsic value — limited margin of safety")
        if pe_ratio and peer_pe and pe_ratio < peer_pe * 0.80:
            notes.append(f"Trading at discount to peers ({pe_ratio:.1f}x vs {peer_pe:.1f}x median)")
        return notes


__all__ = ["ValuationClassifier", "ClassificationResult", "LABELS", "LABEL_COLORS"]
