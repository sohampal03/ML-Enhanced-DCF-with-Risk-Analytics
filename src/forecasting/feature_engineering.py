"""
Feature Engineering Pipeline.

Transforms raw financial statements into ML-ready feature matrices
with engineered ratios, growth rates, lags, and rolling statistics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.data.schemas.financial_schemas import FinancialData
from src.utils.helpers import pct_change_safe, winsorize_series


class FeatureEngineer:
    """
    Creates ML feature matrices from financial statement data.

    Features include:
      - Growth rates (1Y, 3Y, 5Y CAGR)
      - Profitability margins and ratios
      - Leverage metrics
      - Momentum and trend features
      - Lagged features (t-1, t-2)
      - Rolling statistics (3Y mean, std)
    """

    def __init__(self, scale: bool = True) -> None:
        self.scale = scale
        self._scaler: Optional[StandardScaler] = None

    def build_features(self, data: FinancialData) -> pd.DataFrame:
        """
        Build a feature matrix from FinancialData.

        Args:
            data: Financial statements.

        Returns:
            DataFrame indexed by fiscal year (ascending), feature columns.
        """
        logger.info(f"Building features for {data.ticker}")

        income_df = data.income_df().sort_values("fiscal_year")
        balance_df = data.balance_df().sort_values("fiscal_year")
        cashflow_df = data.cashflow_df().sort_values("fiscal_year")

        df = income_df[
            [
                "fiscal_year",
                "revenue",
                "gross_profit",
                "ebit",
                "net_income",
                "ebitda",
                "depreciation_amortization",
                "gross_margin",
                "ebit_margin",
                "net_margin",
            ]
        ].copy()

        # Merge balance and cashflow data
        df = df.merge(
            balance_df[
                [
                    "fiscal_year",
                    "total_assets",
                    "total_debt",
                    "total_equity",
                    "cash_and_equivalents",
                    "net_working_capital",
                    "total_current_assets",
                    "total_current_liabilities",
                ]
            ],
            on="fiscal_year",
            how="left",
        )
        df = df.merge(
            cashflow_df[
                [
                    "fiscal_year",
                    "operating_cash_flow",
                    "capital_expenditures",
                    "free_cash_flow",
                    "depreciation_amortization",
                ]
            ],
            on="fiscal_year",
            how="left",
            suffixes=("", "_cf"),
        )

        # Use cash flow D&A if income D&A is missing
        df["da"] = df["depreciation_amortization"].fillna(df["depreciation_amortization_cf"])

        # ── Growth Features ───────────────────────────────────────────────────
        df["revenue_growth_1y"] = pct_change_safe(df["revenue"])
        df["revenue_growth_3y"] = df["revenue"].pct_change(3)
        df["ebit_growth_1y"] = pct_change_safe(df["ebit"])
        df["net_income_growth_1y"] = pct_change_safe(df["net_income"])
        df["fcf_growth_1y"] = pct_change_safe(df["free_cash_flow"])

        # ── Profitability ─────────────────────────────────────────────────────
        df["gross_margin"] = df["gross_margin"].fillna(
            df["gross_profit"] / df["revenue"].replace(0, np.nan)
        )
        df["ebit_margin"] = df["ebit_margin"].fillna(df["ebit"] / df["revenue"].replace(0, np.nan))
        df["net_margin"] = df["net_margin"].fillna(
            df["net_income"] / df["revenue"].replace(0, np.nan)
        )
        df["fcf_margin"] = df["free_cash_flow"] / df["revenue"].replace(0, np.nan)
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        df["roe"] = df["net_income"] / df["total_equity"].replace(0, np.nan)
        df["asset_turnover"] = df["revenue"] / df["total_assets"].replace(0, np.nan)

        # ── Leverage & Liquidity ──────────────────────────────────────────────
        df["debt_to_equity"] = df["total_debt"] / df["total_equity"].replace(0, np.nan)
        df["debt_to_assets"] = df["total_debt"] / df["total_assets"].replace(0, np.nan)
        df["current_ratio"] = df["total_current_assets"] / df["total_current_liabilities"].replace(
            0, np.nan
        )
        df["net_debt_to_ebitda"] = (df["total_debt"] - df["cash_and_equivalents"]) / df[
            "ebitda"
        ].replace(0, np.nan)
        df["interest_coverage"] = df["ebit"] / (df["total_debt"] * 0.05).replace(0, np.nan)

        # ── Capex & Investment ────────────────────────────────────────────────
        df["capex_pct_revenue"] = df["capital_expenditures"].abs() / df["revenue"].replace(
            0, np.nan
        )
        df["da_pct_revenue"] = df["da"] / df["revenue"].replace(0, np.nan)
        df["capex_pct_assets"] = df["capital_expenditures"].abs() / df["total_assets"].replace(
            0, np.nan
        )

        # ── Cash Quality ──────────────────────────────────────────────────────
        df["ocf_to_net_income"] = df["operating_cash_flow"] / df["net_income"].replace(0, np.nan)
        df["fcf_to_ebitda"] = df["free_cash_flow"] / df["ebitda"].replace(0, np.nan)

        # ── Rolling Statistics (3Y) ───────────────────────────────────────────
        for col in ["revenue_growth_1y", "ebit_margin", "net_margin"]:
            df[f"{col}_3y_mean"] = df[col].rolling(3, min_periods=1).mean()
            df[f"{col}_3y_std"] = df[col].rolling(3, min_periods=1).std()

        # ── Lagged Features ───────────────────────────────────────────────────
        for col in ["revenue", "ebit_margin", "revenue_growth_1y"]:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

        # ── Time feature ──────────────────────────────────────────────────────
        df["year_idx"] = range(len(df))

        # Drop raw financials (keep only engineered features)
        raw_cols = [
            "gross_profit",
            "net_income",
            "depreciation_amortization",
            "depreciation_amortization_cf",
            "da",
            "total_current_assets",
            "total_current_liabilities",
        ]
        df = df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")

        # Winsorize to remove outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != "fiscal_year":
                df[col] = winsorize_series(df[col].dropna()).reindex(df.index)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.set_index("fiscal_year")

        logger.success(f"Built {df.shape[1]} features for {data.ticker} ({len(df)} years)")
        return df

    def prepare_revenue_forecast_data(
        self, data: FinancialData, target_col: str = "revenue"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X, y for revenue / margin forecasting.

        Args:
            data: Financial data.
            target_col: Target variable ('revenue', 'ebit_margin', etc.).

        Returns:
            Tuple of (X, y) with NaN rows dropped.
        """
        features = self.build_features(data)

        if target_col not in features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features")

        y = features[target_col]
        X = features.drop(columns=[target_col])

        # Drop NaN rows
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 3:
            raise ValueError(f"Insufficient clean data ({len(X_clean)} rows) for {data.ticker}")

        logger.info(f"Training data shape: X={X_clean.shape}, y={y_clean.shape}")
        return X_clean, y_clean

    def get_feature_names(self, data: FinancialData) -> list[str]:
        """Return list of all feature names."""
        return self.build_features(data).columns.tolist()


__all__ = ["FeatureEngineer"]
