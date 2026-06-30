# AlphaForge — AI-Powered Intelligent Business Valuation Platform

<div align="center">

![AlphaForge Banner](docs/banner.png)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI/CD](https://github.com/yourusername/ai-business-valuation/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/ai-business-valuation/actions)

**Machine Learning Enhanced DCF Valuation with Probabilistic Risk Analysis and Explainable AI**

*Institutional-grade financial analysis for any publicly traded company*

</div>

---

## 🎯 What is AlphaForge?

AlphaForge is a production-grade financial machine learning platform that combines traditional investment banking valuation methodologies with cutting-edge machine learning to provide institutional-quality company valuations.

**Enter any ticker → Get a complete investment report in seconds.**

### Core Capabilities

| Module | Description |
|---|---|
| **DCF Engine** | Two-stage FCFF-based DCF with Gordon Growth terminal value |
| **WACC Engine** | CAPM cost of equity + market-based cost of debt |
| **FCFF Engine** | Historical free cash flow to firm computation |
| **ML Forecasting** | Revenue & margin forecasting with 6 models + leaderboard |
| **Monte Carlo** | 10,000+ vectorized simulations with tornado sensitivity |
| **Explainable AI** | SHAP values with natural language explanations |
| **Comps Engine** | Sector peer comparison with key multiples |
| **Learning Center** | Interactive finance education (9 concepts) |
| **Report Generator** | HTML/CSV institutional reports |

---

## 🖥️ Screenshots

The dashboard features a Bloomberg Terminal-inspired dark mode with:
- Live DCF sliders with instant recalculation
- Monte Carlo histograms, violin plots, and fan charts
- SHAP feature importance attribution
- Interactive sensitivity heatmaps
- Executive summary with traffic-light indicators

---

## 🚀 Quick Start

### Local Development (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-business-valuation.git
cd ai-business-valuation/ai_business_valuation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
cp .env.example .env

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at **http://localhost:8501**

---

## 📁 Project Structure

```
ai_business_valuation/
│
├── dashboard/                  # Streamlit Bloomberg-style UI
│   ├── app.py                  # Main entry point
│   ├── pages/
│   │   ├── 01_Executive_Summary.py
│   │   ├── 02_Company_Explorer.py
│   │   ├── 03_Financial_Statements.py
│   │   ├── 04_Forecasting_Center.py
│   │   ├── 05_DCF_Lab.py       ← MOST IMPORTANT
│   │   ├── 06_Risk_Analytics.py
│   │   ├── 07_Explainability_Hub.py
│   │   ├── 08_Learning_Center.py
│   │   └── 09_Report_Generator.py
│   ├── components/             # Reusable UI components
│   └── styles/main.css         # Bloomberg-inspired design system
│
├── src/
│   ├── data/                   # Data layer
│   │   ├── adapters/           # yfinance, FMP adapters
│   │   ├── repositories/       # Repository pattern + caching
│   │   └── schemas/            # Pydantic financial schemas
│   ├── finance/engines/        # Core financial engines
│   │   ├── fcff_engine.py      # FCFF computation
│   │   ├── wacc_engine.py      # WACC + CAPM
│   │   ├── dcf_engine.py       # Two-stage DCF
│   │   └── comps_engine.py     # Comparable analysis
│   ├── forecasting/            # ML pipeline
│   │   ├── feature_engineering.py
│   │   └── models/             # Revenue, Margin, Classifier
│   ├── simulation/             # Monte Carlo engine
│   ├── explainability/         # SHAP + LIME + narrative
│   ├── valuation/orchestrator.py  # Central coordinator
│   ├── api/main.py             # FastAPI backend
│   └── utils/                  # Logging, helpers, caching
│
├── configs/settings.py         # Pydantic BaseSettings
├── tests/                      # pytest unit + integration tests
└── .github/workflows/ci.yml    # GitHub Actions CI/CD
```

---

## 🔧 Configuration

All settings can be overridden via environment variables or `.env` file:

```env
# Data Sources
DATA_PRIMARY_ADAPTER=yfinance    # yfinance | fmp | alpha_vantage
DATA_FMP_API_KEY=your_key        # Optional: Financial Modeling Prep
DATA_CACHE_TTL_SECONDS=3600

# ML Settings
ML_CV_FOLDS=5
ML_RANDOM_SEED=42
ML_ENABLE_CATBOOST=true
ML_ENABLE_LIGHTGBM=true
ML_ENABLE_XGBOOST=true

# Simulation
SIM_N_SIMULATIONS=10000

# DCF Defaults
DCF_DEFAULT_FORECAST_YEARS=10
DCF_DEFAULT_TERMINAL_GROWTH_RATE=0.025
DCF_RISK_FREE_RATE=0.0445         # 10Y US Treasury
DCF_MARKET_RISK_PREMIUM=0.055     # Damodaran

# Database
DB_REDIS_URL=redis://localhost:6379/0
DB_URL=postgresql://postgres:password@localhost:5432/valuation_db
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Coverage report
open htmlcov/index.html
```

---

## 📊 Supported Tickers

| Market | Examples |
|---|---|
| **US Equities** | AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA |
| **US Tech** | META, ADBE, CRM, ORCL, INTC |
| **Finance** | JPM, GS, MS, BAC, BLK |
| **NSE India** | INFY.NS, RELIANCE.NS, TCS.NS, HDFCBANK.NS |
| **Other** | Most yfinance-supported international tickers |

---

## 🏗️ Architecture

```
User → Streamlit Dashboard → ValuationOrchestrator → {
  FinancialRepository (yfinance/FMP)
  FCFFEngine → WACCEngine → DCFEngine
  RevenueForecastingPipeline (6 ML models)
  MonteCarloSimulator (10K paths)
  CompsEngine (Sector peers)
  ValuationClassifier (Undervalued/Fair/Overvalued)
  SHAPExplainer (Feature attribution)
} → FullValuationReport → Dashboard Pages
```

---

## 🚀 Deployment

### Render (Free Tier)
1. Connect GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run dashboard/app.py --server.port=$PORT`
4. Set environment variables in Render dashboard

---

## 📚 Financial Methodology

### DCF Model
Two-stage FCFF-based DCF:
- **Stage 1**: Explicit forecast (5–10 years) with ML-enhanced revenue/margin forecasts
- **Stage 2**: Gordon Growth terminal value
- **WACC**: CAPM cost of equity + interest-based cost of debt

### Monte Carlo
10,000 vectorized simulations with:
- Revenue growth: Normal distribution
- EBIT margin: Normal distribution  
- WACC: Normal distribution
- Terminal growth: Triangular distribution
- CAPEX: Normal distribution

### ML Models Compared
- Linear Regression / Ridge
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

Selected by lowest MAPE on TimeSeriesSplit cross-validation.

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**. Valuations generated by AlphaForge are based on publicly available data and mathematical models. This is **not financial advice**. Always conduct your own due diligence before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using Python, Streamlit, and Machine Learning
</div>
