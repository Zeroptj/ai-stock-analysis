"""Residual Income Model — for Banks and Financials.

RI = Net Income - (Equity × Cost of Equity)
Value = Book Value + PV of future Residual Income
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from stock_analysis.calc_engine.ratios import safe_div
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class RIAssumptions:
    """Assumptions for Residual Income model."""

    book_value_per_share: float = 0
    eps: float = 0
    roe: float = 0.12  # Return on equity
    roe_fade_to: float = 0.10  # Long-term ROE
    cost_of_equity: float = 0.10

    # CAPM inputs
    risk_free_rate: float = 0.043
    beta: float = 1.0
    market_risk_premium: float = 0.055

    projection_years: int = 5
    terminal_growth: float = 0.02
    shares_outstanding: float = 0
    payout_ratio: float = 0.30  # Retention = 1 - payout


@dataclass
class RIResult:
    """Result of Residual Income valuation."""

    fair_value_per_share: float
    book_value_component: float
    pv_residual_income: float
    terminal_component: float
    projected_bvps: list[float]
    projected_eps: list[float]
    projected_ri: list[float]
    assumptions: RIAssumptions
    current_price: float = 0
    upside_pct: float = 0


def run_residual_income(
    assumptions: RIAssumptions, current_price: float = 0
) -> RIResult:
    """Run Residual Income valuation model."""
    logger.info("running_residual_income")

    coe = assumptions.risk_free_rate + assumptions.beta * assumptions.market_risk_premium

    bvps_list = []
    eps_list = []
    ri_list = []
    pv_ri_list = []

    bvps = assumptions.book_value_per_share
    retention = 1 - assumptions.payout_ratio

    for year in range(assumptions.projection_years):
        # Fade ROE toward long-term
        fade = year / max(assumptions.projection_years - 1, 1)
        roe = assumptions.roe + fade * (assumptions.roe_fade_to - assumptions.roe)

        # EPS = BVPS × ROE
        eps = bvps * roe

        # Residual Income = EPS - (BVPS × Cost of Equity)
        ri = eps - (bvps * coe)

        # PV of RI
        pv_ri = ri / (1 + coe) ** (year + 1)

        bvps_list.append(round(bvps, 2))
        eps_list.append(round(eps, 2))
        ri_list.append(round(ri, 2))
        pv_ri_list.append(round(pv_ri, 2))

        # Update BVPS: retained earnings grow book value
        bvps = bvps + eps * retention

    # Terminal RI (continuing value)
    terminal_ri = ri_list[-1] if ri_list else 0
    if coe > assumptions.terminal_growth:
        terminal_value = terminal_ri * (1 + assumptions.terminal_growth) / (coe - assumptions.terminal_growth)
    else:
        terminal_value = terminal_ri * 15  # Fallback

    pv_terminal = terminal_value / (1 + coe) ** assumptions.projection_years

    # Fair value = current BVPS + PV of RI + PV of terminal
    fair_value = assumptions.book_value_per_share + sum(pv_ri_list) + pv_terminal
    upside = safe_div(fair_value - current_price, current_price) if current_price > 0 else None

    result = RIResult(
        fair_value_per_share=round(fair_value, 2),
        book_value_component=round(assumptions.book_value_per_share, 2),
        pv_residual_income=round(sum(pv_ri_list), 2),
        terminal_component=round(pv_terminal, 2),
        projected_bvps=bvps_list,
        projected_eps=eps_list,
        projected_ri=ri_list,
        assumptions=assumptions,
        current_price=current_price,
        upside_pct=round(upside, 4) if upside else 0,
    )

    logger.info("ri_result", fair_value=result.fair_value_per_share, upside=result.upside_pct)
    return result


def build_ri_assumptions(
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
) -> RIAssumptions:
    """Build RI assumptions from SEC + market data.

    Prefers SEC-reported values (stockholders' equity, net income) for BVPS / ROE
    and falls back to yfinance info when SEC facts are missing.
    """
    info = market_data.get("info", {})
    financials = sec_data.get("financials", {})
    shares = info.get("shares_outstanding", 0) or 0

    equity_series = financials.get("stockholders_equity", [])
    net_income_series = financials.get("net_income", [])

    equity_latest = equity_series[0]["value"] if equity_series else 0
    bvps = safe_div(equity_latest, shares) if shares and equity_latest else (
        safe_div(info.get("market_cap", 0),
                 (shares or 1) * (info.get("price_to_book", 1) or 1)) or 0
    )

    # ROE from SEC: average of last N years
    if equity_series and net_income_series:
        pairs = zip(net_income_series, equity_series)
        roe_history = [
            safe_div(ni["value"], eq["value"])
            for ni, eq in pairs if ni.get("value") and eq.get("value")
        ]
        roe_history = [r for r in roe_history if r is not None and 0 < r < 1.0]
        roe = sum(roe_history) / len(roe_history) if roe_history else (info.get("roe", 0.12) or 0.12)
    else:
        roe = info.get("roe", 0.12) or 0.12

    eps = info.get("eps", 0) or safe_div(
        net_income_series[0]["value"] if net_income_series else 0, shares or 1
    ) or 0
    beta = info.get("beta", 1.0) or 1.0
    rfr = macro_data.get("risk_free_rate", settings.dcf.risk_free_rate_manual)
    payout = info.get("payout_ratio", 0.30) or 0.30

    return RIAssumptions(
        book_value_per_share=bvps,
        eps=eps,
        roe=roe,
        roe_fade_to=max(roe * 0.8, 0.08),
        risk_free_rate=rfr,
        beta=beta,
        market_risk_premium=settings.dcf.market_risk_premium,
        payout_ratio=min(payout, 0.90),
        shares_outstanding=shares,
    )
