"""
Application configuration using Pydantic BaseSettings.

All settings can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root
ROOT_DIR = Path(__file__).parent.parent


class DataSourceSettings(BaseSettings):
    """Data source adapter configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    primary_adapter: Literal["yfinance", "fmp", "alpha_vantage", "polygon"] = "yfinance"
    fmp_api_key: str = Field(default="", description="Financial Modeling Prep API key")
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")
    polygon_api_key: str = Field(default="", description="Polygon.io API key")
    finnhub_api_key: str = Field(default="", description="Finnhub API key")
    fred_api_key: str = Field(default="", description="FRED API key")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    historical_years: int = Field(default=10, description="Years of historical data to fetch")


class MLSettings(BaseSettings):
    """Machine learning pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore")

    cv_folds: int = Field(default=5, description="TimeSeriesSplit folds")
    optuna_trials: int = Field(default=50, description="Optuna HPO trials")
    random_seed: int = Field(default=42)
    model_save_dir: Path = Field(default=ROOT_DIR / "models" / "saved")
    enable_lstm: bool = Field(default=False, description="Enable LSTM (requires GPU)")
    enable_catboost: bool = Field(default=True)
    enable_lightgbm: bool = Field(default=True)
    enable_xgboost: bool = Field(default=True)
    n_jobs: int = Field(default=-1, description="Parallel jobs (-1 = all cores)")


class SimulationSettings(BaseSettings):
    """Monte Carlo simulation configuration."""

    model_config = SettingsConfigDict(env_prefix="SIM_", extra="ignore")

    n_simulations: int = Field(default=10_000, description="Monte Carlo paths")
    confidence_levels: list[float] = Field(default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    random_seed: int = Field(default=42)


class DCFSettings(BaseSettings):
    """DCF engine defaults."""

    model_config = SettingsConfigDict(env_prefix="DCF_", extra="ignore")

    default_forecast_years: int = Field(default=10)
    default_terminal_growth_rate: float = Field(default=0.025)  # 2.5%
    default_margin_of_safety: float = Field(default=0.30)  # 30%
    risk_free_rate: float = Field(default=0.0445)  # 10Y US Treasury
    market_risk_premium: float = Field(default=0.055)  # Damodaran estimate


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    url: str = Field(
        default="postgresql://postgres:password@localhost:5432/valuation_db",
        description="SQLAlchemy database URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    pool_size: int = Field(default=5)
    max_overflow: int = Field(default=10)


class APISettings(BaseSettings):
    """FastAPI settings."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(default=["http://localhost:8501"])
    api_prefix: str = Field(default="/api/v1")


class DashboardSettings(BaseSettings):
    """Streamlit dashboard settings."""

    model_config = SettingsConfigDict(env_prefix="DASHBOARD_", extra="ignore")

    title: str = Field(default="AlphaForge — Intelligent Business Valuation")
    default_theme: Literal["dark", "light"] = Field(default="dark")
    page_width: Literal["centered", "wide"] = Field(default="wide")


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="AlphaForge")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_file: Path = Field(default=ROOT_DIR / "logs" / "app.log")

    # Sub-settings (instantiated as nested objects)
    data: DataSourceSettings = Field(default_factory=DataSourceSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    dcf: DCFSettings = Field(default_factory=DCFSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    @field_validator("log_file", mode="before")
    @classmethod
    def create_log_dir(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance
settings = Settings()

__all__ = ["settings", "Settings", "ROOT_DIR"]
