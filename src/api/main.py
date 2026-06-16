"""FastAPI backend application."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.settings import settings
from src.utils.logging import configure_logging
from src.valuation.orchestrator import ValuationOrchestrator

configure_logging(settings.log_level)

# Global orchestrator (singleton)
_orchestrator: Optional[ValuationOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    _orchestrator = ValuationOrchestrator()
    yield
    _orchestrator = None


app = FastAPI(
    title="AlphaForge API",
    description="AI-Powered Business Valuation Platform API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValuationRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_years: int = Field(default=10, ge=5, le=15)
    wacc_override: Optional[float] = Field(default=None, ge=0.04, le=0.30)
    terminal_growth_rate: Optional[float] = Field(default=None, ge=0.001, le=0.05)
    run_ml: bool = Field(default=True)
    run_monte_carlo: bool = Field(default=True)


class ValuationResponse(BaseModel):
    ticker: str
    company_name: str
    current_price: Optional[float]
    intrinsic_value: Optional[float]
    margin_of_safety: Optional[float]
    recommendation: Optional[str]
    wacc: Optional[float]
    enterprise_value: Optional[float]
    prob_undervalued: Optional[float]
    errors: dict


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post(f"{settings.api.api_prefix}/valuate", response_model=ValuationResponse)
async def valuate(request: ValuationRequest):
    """Run complete valuation analysis for a ticker."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        report = _orchestrator.analyze(
            ticker=request.ticker,
            run_ml=request.run_ml,
            run_monte_carlo=request.run_monte_carlo,
            forecast_years=request.forecast_years,
            wacc_override=request.wacc_override,
            terminal_growth_rate=request.terminal_growth_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Valuation failed: {str(e)}")

    dcf = report.dcf_result
    mc = report.monte_carlo

    return ValuationResponse(
        ticker=report.ticker,
        company_name=report.company_name,
        current_price=dcf.current_price if dcf else None,
        intrinsic_value=dcf.intrinsic_value_per_share if dcf else None,
        margin_of_safety=dcf.margin_of_safety if dcf else None,
        recommendation=dcf.recommendation if dcf else None,
        wacc=dcf.wacc if dcf else None,
        enterprise_value=dcf.enterprise_value if dcf else None,
        prob_undervalued=mc.prob_undervalued if mc else None,
        errors=report.errors,
    )


@app.get(f"{settings.api.api_prefix}/company/{{ticker}}")
async def get_company_info(ticker: str):
    """Get company information and key metrics."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        data = _orchestrator.repository.get_financial_data(ticker)
        return data.company_info.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.api.host, port=settings.api.port, reload=True)
