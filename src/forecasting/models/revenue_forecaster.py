"""
Revenue & Margin Forecasting Engine.

Implements a model leaderboard approach with TimeSeriesSplit cross-validation.
Models: LinearRegression, RandomForest, XGBoost, LightGBM, CatBoost.
Auto-selects the best model based on MAPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from configs.settings import settings
from src.data.schemas.financial_schemas import FinancialData
from src.forecasting.feature_engineering import FeatureEngineer

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    logger.warning("CatBoost not available")


@dataclass
class ModelScore:
    """Performance metrics for a single model."""

    name: str
    rmse: float
    mae: float
    mape: float
    r2: float
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "Model": self.name,
            "RMSE": f"{self.rmse:,.2f}",
            "MAE": f"{self.mae:,.2f}",
            "MAPE": f"{self.mape:.2f}%",
            "R²": f"{self.r2:.4f}",
            "Rank": self.rank,
        }


@dataclass
class ForecastResult:
    """Output of the revenue/margin forecasting pipeline."""

    ticker: str
    target: str                       # 'revenue', 'ebit_margin', etc.
    best_model_name: str
    forecast_values: list[float]      # Next N years
    forecast_years: list[int]
    confidence_lower: list[float]
    confidence_upper: list[float]
    historical_values: list[float]
    historical_years: list[int]
    model_leaderboard: list[ModelScore] = field(default_factory=list)
    feature_importances: Optional[dict] = None
    notes: list[str] = field(default_factory=list)

    def to_forecast_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Year": self.forecast_years,
            "Forecast": self.forecast_values,
            "Lower CI": self.confidence_lower,
            "Upper CI": self.confidence_upper,
        })

    def to_leaderboard_df(self) -> pd.DataFrame:
        return pd.DataFrame([m.to_dict() for m in self.model_leaderboard])


class RevenueForecastingPipeline:
    """
    Multi-model revenue forecasting pipeline with automatic model selection.

    Uses TimeSeriesSplit cross-validation and selects the best model
    by MAPE. Generates confidence intervals via bootstrap.
    """

    CV_FOLDS = 3  # Limited by small financial datasets
    FORECAST_HORIZON = 5  # Years ahead

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()
        self._models = self._build_model_registry()

    def forecast(
        self,
        data: FinancialData,
        target: str = "revenue",
        horizon: int = 5,
    ) -> ForecastResult:
        """
        Run full forecasting pipeline.

        Args:
            data: Financial data.
            target: Target variable ('revenue', 'ebit_margin', 'gross_margin').
            horizon: Forecast horizon in years.

        Returns:
            ForecastResult with best model forecast and leaderboard.
        """
        ticker = data.ticker
        logger.info(f"[RevenueForecastingPipeline] Forecasting {target} for {ticker}")

        try:
            X, y = self.feature_engineer.prepare_revenue_forecast_data(data, target_col=target)
        except ValueError as e:
            logger.warning(f"Cannot forecast: {e}")
            return self._fallback_forecast(data, target, horizon)

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X.fillna(0)), index=X.index, columns=X.columns)

        # ── Cross-Validation Leaderboard ──────────────────────────────────────
        scores: list[ModelScore] = []
        tscv = TimeSeriesSplit(n_splits=min(self.CV_FOLDS, len(X) - 2))

        for name, model in self._models.items():
            try:
                score = self._cross_validate(name, model, X_scaled, y, tscv)
                scores.append(score)
            except Exception as e:
                logger.debug(f"Model {name} failed CV: {e}")

        if not scores:
            return self._fallback_forecast(data, target, horizon)

        # Rank by MAPE (lower = better)
        scores.sort(key=lambda s: s.mape)
        for i, s in enumerate(scores):
            s.rank = i + 1

        best_name = scores[0].name
        best_model = self._models[best_name]
        logger.info(f"Best model: {best_name} (MAPE: {scores[0].mape:.2f}%)")

        # ── Fit Best Model on Full Data ────────────────────────────────────────
        X_full = X_scaled.fillna(0).values
        y_arr = y.values
        best_model.fit(X_full, y_arr)

        # ── Feature Importance ─────────────────────────────────────────────────
        feature_importances = self._get_feature_importance(best_model, X.columns.tolist())

        # ── Future Feature Extrapolation ───────────────────────────────────────
        last_year = int(y.index[-1])
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))

        # Extrapolate features using recent trends
        X_future = self._extrapolate_features(X_scaled, horizon)
        point_forecasts = best_model.predict(X_future)

        # ── Bootstrap Confidence Intervals ─────────────────────────────────────
        lower_ci, upper_ci = self._bootstrap_ci(best_model, X_full, y_arr, X_future)

        # Handle negative revenue forecasts
        if target == "revenue":
            point_forecasts = np.maximum(point_forecasts, 0)
            lower_ci = np.maximum(lower_ci, 0)
            upper_ci = np.maximum(upper_ci, 0)

        result = ForecastResult(
            ticker=ticker,
            target=target,
            best_model_name=best_name,
            forecast_values=point_forecasts.tolist(),
            forecast_years=forecast_years,
            confidence_lower=lower_ci.tolist(),
            confidence_upper=upper_ci.tolist(),
            historical_values=y.tolist(),
            historical_years=list(y.index),
            model_leaderboard=scores,
            feature_importances=feature_importances,
        )

        logger.success(
            f"[RevenueForecastingPipeline] {ticker} {target} forecast complete. "
            f"Best: {best_name}, MAPE: {scores[0].mape:.2f}%"
        )
        return result

    def _cross_validate(
        self,
        name: str,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        tscv: TimeSeriesSplit,
    ) -> ModelScore:
        """Run time-series cross-validation for a model."""
        all_preds, all_actual = [], []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

            model.fit(X_train, y_train)  # type: ignore
            preds = model.predict(X_val)  # type: ignore

            all_preds.extend(preds)
            all_actual.extend(y_val)

        actual = np.array(all_actual)
        preds = np.array(all_preds)

        rmse = float(np.sqrt(mean_squared_error(actual, preds)))
        mae = float(mean_absolute_error(actual, preds))
        r2 = float(r2_score(actual, preds))
        mape = float(
            np.mean(np.abs((actual - preds) / np.where(actual == 0, 1, actual))) * 100
        )

        return ModelScore(name=name, rmse=rmse, mae=mae, mape=mape, r2=r2)

    def _extrapolate_features(self, X_scaled: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        Simple feature extrapolation: extend recent trend via linear regression.
        Used to create 'future' feature vectors for out-of-sample prediction.
        """
        X_arr = X_scaled.fillna(0).values
        n = len(X_arr)
        future_rows = []

        for step in range(1, horizon + 1):
            if n >= 2:
                # Linear trend extrapolation per feature
                t = np.arange(n)
                future_row = []
                for col_idx in range(X_arr.shape[1]):
                    col = X_arr[:, col_idx]
                    coeffs = np.polyfit(t, col, 1)
                    future_val = np.polyval(coeffs, n + step - 1)
                    future_row.append(future_val)
                future_rows.append(future_row)
            else:
                future_rows.append(X_arr[-1].tolist())

        return np.array(future_rows)

    def _bootstrap_ci(
        self,
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_future: np.ndarray,
        n_bootstrap: int = 100,
        ci_level: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap residuals to estimate confidence intervals."""
        residuals = y_train - model.predict(X_train)  # type: ignore
        bootstrap_preds = []

        for _ in range(n_bootstrap):
            noise = np.random.choice(residuals, size=len(X_future), replace=True)
            pred = model.predict(X_future) + noise  # type: ignore
            bootstrap_preds.append(pred)

        bootstrap_preds = np.array(bootstrap_preds)
        alpha = (1 - ci_level) / 2
        lower = np.percentile(bootstrap_preds, alpha * 100, axis=0)
        upper = np.percentile(bootstrap_preds, (1 - alpha) * 100, axis=0)
        return lower, upper

    def _get_feature_importance(
        self, model: object, feature_names: list[str]
    ) -> Optional[dict]:
        """Extract feature importance if supported by model."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_  # type: ignore
                return dict(
                    sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:15]
                )
        except Exception:
            pass
        return None

    def _build_model_registry(self) -> dict:
        """Build the dictionary of models to evaluate."""
        models: dict = {
            "Linear Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=settings.ml.random_seed,
                n_jobs=settings.ml.n_jobs,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=settings.ml.random_seed,
            ),
        }

        if HAS_XGB and settings.ml.enable_xgboost:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=settings.ml.random_seed,
                verbosity=0,
            )

        if HAS_LGB and settings.ml.enable_lightgbm:
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05,
                random_state=settings.ml.random_seed,
                verbose=-1,
            )

        if HAS_CAT and settings.ml.enable_catboost:
            models["CatBoost"] = CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.05,
                random_seed=settings.ml.random_seed,
                verbose=0,
            )

        return models

    def _fallback_forecast(
        self, data: FinancialData, target: str, horizon: int
    ) -> ForecastResult:
        """Return a simple trend-based forecast when ML fails."""
        logger.warning(f"Using fallback linear trend for {data.ticker} {target}")

        income_df = data.income_df().sort_values("fiscal_year")
        if target in income_df.columns:
            vals = income_df[target].dropna()
            years = income_df.loc[vals.index, "fiscal_year"].tolist()
        else:
            vals = pd.Series([0.0])
            years = [2024]

        last_year = max(years) if years else 2024
        last_val = float(vals.iloc[-1]) if not vals.empty else 0

        # Simple 5% growth fallback
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        forecasts = [last_val * (1.05 ** i) for i in range(1, horizon + 1)]
        lower = [f * 0.85 for f in forecasts]
        upper = [f * 1.15 for f in forecasts]

        return ForecastResult(
            ticker=data.ticker,
            target=target,
            best_model_name="Linear Trend (Fallback)",
            forecast_values=forecasts,
            forecast_years=forecast_years,
            confidence_lower=lower,
            confidence_upper=upper,
            historical_values=vals.tolist(),
            historical_years=years,
            notes=["ML forecasting unavailable — using simple trend extrapolation"],
        )


class MarginForecastingPipeline(RevenueForecastingPipeline):
    """
    Margin forecasting: EBIT Margin, Gross Margin, Net Margin.

    Inherits RevenueForecastingPipeline with margin-specific logic.
    """

    def forecast_all_margins(self, data: FinancialData, horizon: int = 5) -> dict[str, ForecastResult]:
        """Forecast all three key margins."""
        results = {}
        for margin in ["ebit_margin", "gross_margin", "net_margin"]:
            try:
                results[margin] = self.forecast(data, target=margin, horizon=horizon)
            except Exception as e:
                logger.error(f"Margin forecast failed for {margin}: {e}")
        return results


__all__ = ["RevenueForecastingPipeline", "MarginForecastingPipeline", "ForecastResult", "ModelScore"]
