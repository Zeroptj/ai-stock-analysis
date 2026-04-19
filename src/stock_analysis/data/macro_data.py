"""Macro data client — FRED API for risk-free rate, inflation, etc."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from stock_analysis.config import settings

logger = structlog.get_logger(__name__)

# FRED series IDs
SERIES = {
    "risk_free_rate_10y": "DGS10",       # 10-Year Treasury Constant Maturity Rate
    "risk_free_rate_2y": "DGS2",          # 2-Year Treasury
    "risk_free_rate_3m": "DTB3",          # 3-Month Treasury Bill
    "inflation_cpi": "CPIAUCSL",          # Consumer Price Index
    "inflation_expectations": "T10YIE",   # 10-Year Breakeven Inflation Rate
    "fed_funds_rate": "FEDFUNDS",         # Federal Funds Rate
    "gdp_growth": "A191RL1Q225SBEA",      # Real GDP Growth Rate
    "unemployment": "UNRATE",             # Unemployment Rate
    "sp500": "SP500",                     # S&P 500 Index
    "vix": "VIXCLS",                      # CBOE Volatility Index
    "corporate_spread_baa": "BAAFFM",     # Moody's Baa Corporate Bond Spread
}


def fetch_fred_series(series_id: str, observation_count: int = 12) -> list[dict[str, Any]]:
    """Fetch a FRED data series — returns recent observations."""
    api_key = settings.api_keys.fred_api_key
    if not api_key:
        logger.warning("fred_api_key_not_set", series_id=series_id)
        return []

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)

        if data is None or data.empty:
            return []

        recent = data.dropna().tail(observation_count)
        return [
            {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
            for idx, val in recent.items()
        ]
    except Exception as e:
        logger.error("fred_fetch_error", series_id=series_id, error=str(e))
        return []


def get_risk_free_rate() -> float:
    """Get current risk-free rate (10-Year Treasury yield)."""
    if settings.dcf.risk_free_rate_source == "manual":
        return settings.dcf.risk_free_rate_manual

    data = fetch_fred_series("DGS10", observation_count=1)
    if data:
        rate = data[-1]["value"] / 100  # Convert from percentage
        logger.info("risk_free_rate", rate=rate, source="FRED DGS10")
        return rate

    logger.warning("using_manual_risk_free_rate")
    return settings.dcf.risk_free_rate_manual


def get_market_risk_premium() -> float:
    """Get market risk premium (default from config, can be enhanced with FRED data)."""
    return settings.dcf.market_risk_premium


def fetch_macro_data() -> dict[str, Any]:
    """Main entry point — fetch all relevant macro data."""
    logger.info("fetching_macro_data")

    result: dict[str, Any] = {
        "risk_free_rate": get_risk_free_rate(),
        "market_risk_premium": get_market_risk_premium(),
    }

    # Fetch key macro indicators
    for name, series_id in SERIES.items():
        data = fetch_fred_series(series_id, observation_count=6)
        if data:
            result[name] = {
                "latest": data[-1]["value"],
                "history": data,
            }

    result["source"] = "FRED"
    result["fetched_at"] = datetime.utcnow().isoformat()

    return result
