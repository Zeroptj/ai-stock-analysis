"""Dividend Discount Model (DDM) — for Banks, Utilities, stable dividend payers.

Two-stage DDM: explicit high-growth phase + terminal value via Gordon Growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from stock_analysis.calc_engine.ratios import safe_div
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class DDMAssumptions:
    """Assumptions for DDM valuation."""

    current_dividend_per_share: float = 0
    dividend_growth_rates: list[float] = field(default_factory=list)  # Stage 1 growth
    terminal_dividend_growth: float = 0.03

    # Cost of equity (CAPM)
    risk_free_rate: float = 0.043
    beta: float = 0.8
    market_risk_premium: float = 0.055

    # Payout ratio (for sanity check)
    payout_ratio: float = 0.50
    eps: float = 0

    projection_years: int = 5
    shares_outstanding: float = 0


@dataclass
class DDMResult:
    """Result of DDM valuation."""

    fair_value_per_share: float
    cost_of_equity: float
    projected_dividends: list[float]
    pv_dividends: list[float]
    terminal_value: float
    pv_terminal: float
    assumptions: DDMAssumptions
    current_price: float = 0
    upside_pct: float = 0


def run_ddm(assumptions: DDMAssumptions, current_price: float = 0) -> DDMResult:
    """Run Dividend Discount Model."""
    logger.info("running_ddm")

    # Cost of equity
    cost_of_equity = (
        assumptions.risk_free_rate
        + assumptions.beta * assumptions.market_risk_premium
    )

    # Project dividends
    dividends = []
    current_div = assumptions.current_dividend_per_share
    for i in range(assumptions.projection_years):
        if i < len(assumptions.dividend_growth_rates):
            g = assumptions.dividend_growth_rates[i]
        else:
            g = assumptions.terminal_dividend_growth
        current_div *= (1 + g)
        dividends.append(current_div)

    # PV of dividends
    pv_divs = [d / (1 + cost_of_equity) ** (i + 1) for i, d in enumerate(dividends)]

    # Terminal value (Gordon Growth)
    if cost_of_equity > assumptions.terminal_dividend_growth:
        terminal_div = dividends[-1] * (1 + assumptions.terminal_dividend_growth)
        terminal_value = terminal_div / (cost_of_equity - assumptions.terminal_dividend_growth)
    else:
        terminal_value = dividends[-1] * 20  # Fallback cap

    pv_terminal = terminal_value / (1 + cost_of_equity) ** assumptions.projection_years

    # Fair value
    fair_value = sum(pv_divs) + pv_terminal
    upside = safe_div(fair_value - current_price, current_price) if current_price > 0 else None

    result = DDMResult(
        fair_value_per_share=round(fair_value, 2),
        cost_of_equity=round(cost_of_equity, 4),
        projected_dividends=[round(d, 4) for d in dividends],
        pv_dividends=[round(pv, 4) for pv in pv_divs],
        terminal_value=round(terminal_value, 2),
        pv_terminal=round(pv_terminal, 2),
        assumptions=assumptions,
        current_price=current_price,
        upside_pct=round(upside, 4) if upside else 0,
    )

    logger.info("ddm_result", fair_value=result.fair_value_per_share, upside=result.upside_pct)
    return result


def build_ddm_assumptions(
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
) -> DDMAssumptions:
    """Build DDM assumptions from SEC + market data.

    Prefers SEC dividends-per-share and net-income history when available.
    """
    info = market_data.get("info", {})
    financials = sec_data.get("financials", {})

    # Dividend per share — SEC first, else yfinance yield × price
    dps_series = financials.get("dividends_per_share", [])
    current_div = dps_series[0]["value"] if dps_series else 0
    if not current_div:
        div_yield = info.get("dividend_yield", 0) or 0
        price = info.get("current_price", 0)
        current_div = div_yield * price if div_yield and price else 0

    # Dividend growth from SEC history (CAGR over available years)
    div_growth = 0.05
    if len(dps_series) >= 2:
        try:
            first = dps_series[-1]["value"]
            last = dps_series[0]["value"]
            n = len(dps_series) - 1
            if first and last and first > 0 and n > 0:
                cagr = (last / first) ** (1 / n) - 1
                if -0.2 < cagr < 0.3:
                    div_growth = cagr
        except Exception:
            pass
    else:
        div_growth = min(info.get("earnings_growth", 0.05) or 0.05, 0.10)

    payout = info.get("payout_ratio", 0.5) or 0.5
    eps = info.get("eps", 0) or 0

    beta = info.get("beta", 0.8) or 0.8
    rfr = macro_data.get("risk_free_rate", settings.dcf.risk_free_rate_manual)

    return DDMAssumptions(
        current_dividend_per_share=current_div,
        dividend_growth_rates=[div_growth] * 3 + [div_growth * 0.7, div_growth * 0.5],
        terminal_dividend_growth=min(div_growth * 0.4, 0.03),
        risk_free_rate=rfr,
        beta=beta,
        market_risk_premium=settings.dcf.market_risk_premium,
        payout_ratio=payout,
        eps=eps,
        shares_outstanding=info.get("shares_outstanding", 0) or 0,
    )
