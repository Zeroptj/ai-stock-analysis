"""Industry classifier — maps ticker/sector to the appropriate valuation model.

Uses GICS sector classification to route to the right DCF variant.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import structlog
import yfinance as yf

from stock_analysis.data.cache import cached

logger = structlog.get_logger(__name__)


class ValuationModel(str, Enum):
    DCF_FCFF = "dcf_fcff"          # Tech, Healthcare, Consumer, Industrials
    DDM = "ddm"                     # Banks, Insurance with stable dividends
    RESIDUAL_INCOME = "residual_income"  # Financials (banks, diversified)
    AFFO_REIT = "affo_reit"         # REITs
    DCF_CYCLICAL = "dcf_cyclical"   # Energy, Materials, Commodities


# GICS Sector → default model mapping
SECTOR_MODEL_MAP: dict[str, ValuationModel] = {
    "Information Technology": ValuationModel.DCF_FCFF,
    "Technology": ValuationModel.DCF_FCFF,
    "Communication Services": ValuationModel.DCF_FCFF,
    "Health Care": ValuationModel.DCF_FCFF,
    "Consumer Discretionary": ValuationModel.DCF_FCFF,
    "Consumer Staples": ValuationModel.DCF_FCFF,
    "Industrials": ValuationModel.DCF_FCFF,
    "Financials": ValuationModel.RESIDUAL_INCOME,
    "Real Estate": ValuationModel.AFFO_REIT,
    "Energy": ValuationModel.DCF_CYCLICAL,
    "Materials": ValuationModel.DCF_CYCLICAL,
    "Utilities": ValuationModel.DDM,
}

# Sub-industry overrides (more specific)
SUBINDUSTRY_OVERRIDES: dict[str, ValuationModel] = {
    "Diversified Banks": ValuationModel.RESIDUAL_INCOME,
    "Regional Banks": ValuationModel.RESIDUAL_INCOME,
    "Investment Banking & Brokerage": ValuationModel.RESIDUAL_INCOME,
    "Life & Health Insurance": ValuationModel.DDM,
    "Property & Casualty Insurance": ValuationModel.DDM,
    "Multi-Utilities": ValuationModel.DDM,
    "Electric Utilities": ValuationModel.DDM,
    "Equity REITs": ValuationModel.AFFO_REIT,
    "Mortgage REITs": ValuationModel.AFFO_REIT,
    "Specialty REITs": ValuationModel.AFFO_REIT,
    "Oil & Gas Exploration & Production": ValuationModel.DCF_CYCLICAL,
    "Oil & Gas Integrated": ValuationModel.DCF_CYCLICAL,
    "Commodity Chemicals": ValuationModel.DCF_CYCLICAL,
    "Gold": ValuationModel.DCF_CYCLICAL,
    "Steel": ValuationModel.DCF_CYCLICAL,
}

# Ticker-level overrides (for edge cases)
TICKER_OVERRIDES: dict[str, ValuationModel] = {
    "BRK-B": ValuationModel.RESIDUAL_INCOME,  # Berkshire is unique
    "JPM": ValuationModel.RESIDUAL_INCOME,
    "BAC": ValuationModel.RESIDUAL_INCOME,
    "WFC": ValuationModel.RESIDUAL_INCOME,
    "GS": ValuationModel.RESIDUAL_INCOME,
    "T": ValuationModel.DDM,         # AT&T — dividend focus
    "VZ": ValuationModel.DDM,        # Verizon — dividend focus
    "O": ValuationModel.AFFO_REIT,   # Realty Income
    "SPG": ValuationModel.AFFO_REIT, # Simon Property
}


def classify_ticker(ticker: str, sector: str = "", sub_industry: str = "") -> ValuationModel:
    """Determine the best valuation model for a ticker."""
    ticker = ticker.upper()

    # 1. Check ticker overrides
    if ticker in TICKER_OVERRIDES:
        model = TICKER_OVERRIDES[ticker]
        logger.info("model_classified", ticker=ticker, model=model.value, source="ticker_override")
        return model

    # 2. Check sub-industry overrides
    if sub_industry:
        for key, model in SUBINDUSTRY_OVERRIDES.items():
            if key.lower() in sub_industry.lower():
                logger.info("model_classified", ticker=ticker, model=model.value, source="sub_industry")
                return model

    # 3. Check sector mapping
    if sector in SECTOR_MODEL_MAP:
        model = SECTOR_MODEL_MAP[sector]
        logger.info("model_classified", ticker=ticker, model=model.value, source="sector")
        return model

    # 4. Fallback: fetch from yfinance (cached 12h)
    if not sector:
        try:
            info = cached(
                key=f"yf_info:{ticker}",
                ttl_hours=12,
                fetcher=lambda: dict(yf.Ticker(ticker).info or {}),
            ) or {}
            sector = info.get("sector", "")
            sub_industry = info.get("industry", "")
            if sector or sub_industry:
                return classify_ticker(ticker, sector, sub_industry)
        except Exception as e:
            logger.warning("classification_fallback", ticker=ticker, error=str(e))

    # Default
    logger.info("model_classified", ticker=ticker, model="dcf_fcff", source="default")
    return ValuationModel.DCF_FCFF


def get_model_description(model: ValuationModel) -> str:
    """Get human-readable description of the valuation model."""
    descriptions = {
        ValuationModel.DCF_FCFF: "Discounted Cash Flow (Free Cash Flow to Firm) — 2-stage model",
        ValuationModel.DDM: "Dividend Discount Model — Gordon Growth + 2-stage",
        ValuationModel.RESIDUAL_INCOME: "Residual Income Model — for banks & financials",
        ValuationModel.AFFO_REIT: "AFFO-based model — for Real Estate Investment Trusts",
        ValuationModel.DCF_CYCLICAL: "Normalized DCF — for cyclical industries (Energy, Materials)",
    }
    return descriptions.get(model, "Unknown model")
