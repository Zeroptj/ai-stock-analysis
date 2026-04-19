"""AFFO-based valuation for REITs.

REITs use Adjusted Funds From Operations (AFFO) instead of traditional earnings.
Value = AFFO per share / Cap Rate (or P/AFFO multiple approach)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from stock_analysis.calc_engine.ratios import safe_div
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class AFFOAssumptions:
    """Assumptions for REIT AFFO model."""

    ffo_per_share: float = 0  # Funds From Operations per share
    affo_per_share: float = 0  # Adjusted FFO per share
    affo_growth_rates: list[float] = field(default_factory=list)
    terminal_affo_growth: float = 0.02

    # Dividend
    dividend_per_share: float = 0
    affo_payout_ratio: float = 0.75  # Typical REIT payout

    # Discount rate
    risk_free_rate: float = 0.043
    beta: float = 0.7  # REITs typically lower beta
    market_risk_premium: float = 0.055
    cap_rate_spread: float = 0.02  # Premium over risk-free

    # NAV approach
    nav_per_share: float = 0  # Net Asset Value
    nav_premium_discount: float = 0  # Current premium/discount to NAV

    projection_years: int = 5
    shares_outstanding: float = 0


@dataclass
class AFFOResult:
    """Result of REIT AFFO valuation."""

    fair_value_per_share: float
    p_affo_implied: float
    projected_affo: list[float]
    projected_dividends: list[float]
    pv_dividends: list[float]
    terminal_value: float
    nav_fair_value: float | None
    assumptions: AFFOAssumptions
    current_price: float = 0
    upside_pct: float = 0


def run_affo_valuation(
    assumptions: AFFOAssumptions, current_price: float = 0
) -> AFFOResult:
    """Run REIT AFFO-based valuation."""
    logger.info("running_affo_valuation")

    cost_of_equity = (
        assumptions.risk_free_rate
        + assumptions.beta * assumptions.market_risk_premium
    )

    # Project AFFO
    affo_list = []
    current_affo = assumptions.affo_per_share or assumptions.ffo_per_share
    for i in range(assumptions.projection_years):
        if i < len(assumptions.affo_growth_rates):
            g = assumptions.affo_growth_rates[i]
        else:
            g = assumptions.terminal_affo_growth
        current_affo *= (1 + g)
        affo_list.append(current_affo)

    # Project dividends (AFFO × payout ratio)
    div_list = [affo * assumptions.affo_payout_ratio for affo in affo_list]

    # PV of dividends
    pv_divs = [d / (1 + cost_of_equity) ** (i + 1) for i, d in enumerate(div_list)]

    # Terminal value via P/AFFO multiple
    # Implied P/AFFO from cap rate
    implied_cap_rate = assumptions.risk_free_rate + assumptions.cap_rate_spread
    implied_p_affo = 1 / implied_cap_rate if implied_cap_rate > 0 else 15

    terminal_value = affo_list[-1] * implied_p_affo
    pv_terminal = terminal_value / (1 + cost_of_equity) ** assumptions.projection_years

    # Fair value from DDM approach
    fair_value = sum(pv_divs) + pv_terminal

    # NAV-based fair value (if available)
    nav_fv = None
    if assumptions.nav_per_share > 0:
        nav_fv = assumptions.nav_per_share * (1 + assumptions.terminal_affo_growth)

    upside = safe_div(fair_value - current_price, current_price) if current_price > 0 else None

    result = AFFOResult(
        fair_value_per_share=round(fair_value, 2),
        p_affo_implied=round(implied_p_affo, 1),
        projected_affo=[round(a, 2) for a in affo_list],
        projected_dividends=[round(d, 2) for d in div_list],
        pv_dividends=[round(pv, 2) for pv in pv_divs],
        terminal_value=round(terminal_value, 2),
        nav_fair_value=round(nav_fv, 2) if nav_fv else None,
        assumptions=assumptions,
        current_price=current_price,
        upside_pct=round(upside, 4) if upside else 0,
    )

    logger.info("affo_result", fair_value=result.fair_value_per_share, upside=result.upside_pct)
    return result


def build_affo_assumptions(
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
) -> AFFOAssumptions:
    """Build AFFO assumptions.

    REIT-specific: FFO ≈ Net Income + D&A - Gains on sales. We approximate
    FFO/share using SEC net income + depreciation; AFFO ≈ FFO × 0.9.
    """
    info = market_data.get("info", {})
    financials = sec_data.get("financials", {})
    shares = info.get("shares_outstanding", 0) or 0

    ni_latest = (financials.get("net_income") or [{}])[0].get("value") or 0
    dep_latest = (financials.get("depreciation") or [{}])[0].get("value") or 0
    ffo_total = ni_latest + dep_latest  # FFO proxy (skip gain adjustments)
    ffo_per_share = safe_div(ffo_total, shares) if shares else 0

    if not ffo_per_share:
        eps = info.get("eps", 0) or 0
        ffo_per_share = eps * 1.5 if eps > 0 else 0

    # Dividend per share from SEC
    dps_series = financials.get("dividends_per_share", [])
    div_per_share = dps_series[0]["value"] if dps_series else 0
    if not div_per_share:
        div_yield = info.get("dividend_yield", 0) or 0
        price = info.get("current_price", 0)
        div_per_share = div_yield * price if div_yield and price else 0

    # Revenue growth from SEC history
    rev_series = financials.get("revenue", [])
    growth = info.get("revenue_growth", 0.03) or 0.03
    if len(rev_series) >= 2:
        try:
            g = safe_div(
                rev_series[0]["value"] - rev_series[1]["value"],
                abs(rev_series[1]["value"]),
            )
            if g is not None and -0.2 < g < 0.3:
                growth = g
        except Exception:
            pass

    beta = info.get("beta", 0.7) or 0.7
    rfr = macro_data.get("risk_free_rate", settings.dcf.risk_free_rate_manual)

    affo_ps = ffo_per_share * 0.9
    payout = safe_div(div_per_share, affo_ps) if affo_ps else 0.75
    payout = min(max(payout or 0.75, 0.5), 0.95)

    return AFFOAssumptions(
        ffo_per_share=ffo_per_share,
        affo_per_share=affo_ps,
        affo_growth_rates=[growth] * 3 + [growth * 0.7, growth * 0.5],
        terminal_affo_growth=min(max(growth * 0.3, 0.015), 0.025),
        dividend_per_share=div_per_share,
        affo_payout_ratio=payout,
        risk_free_rate=rfr,
        beta=beta,
        market_risk_premium=settings.dcf.market_risk_premium,
        shares_outstanding=shares,
    )
