"""DCF Model (Free Cash Flow to Firm) — 2-stage model for Tech/General.

All calculations are pure Python — no LLM involved.
Produces fair value estimate with bull/base/bear scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from stock_analysis.calc_engine.ratios import safe_div
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class DCFAssumptions:
    """Assumptions for the DCF model."""

    # Revenue projection
    revenue_base: float = 0  # Latest annual revenue
    revenue_growth_rates: list[float] = field(default_factory=list)  # Year-by-year growth

    # Margins
    operating_margin: float = 0.20
    tax_rate: float = 0.21

    # Capital structure
    capex_to_revenue: float = 0.05
    depreciation_to_revenue: float = 0.04
    nwc_change_to_revenue: float = 0.01  # Net working capital change as % of revenue

    # WACC components
    risk_free_rate: float = 0.043
    beta: float = 1.0
    market_risk_premium: float = 0.055
    cost_of_debt: float = 0.05
    debt_ratio: float = 0.20  # D / (D+E)

    # Terminal value
    terminal_growth_rate: float = 0.025
    terminal_method: str = "gordon"  # "gordon" or "exit_multiple"
    exit_ev_ebitda_multiple: float = 15.0

    # Share info
    shares_outstanding: float = 0
    net_debt: float = 0  # Total debt - cash

    # Projection
    projection_years: int = 5


@dataclass
class DCFResult:
    """Result of DCF valuation."""

    fair_value_per_share: float
    enterprise_value: float
    equity_value: float
    wacc: float
    terminal_value: float
    pv_fcff: list[float]
    projected_revenue: list[float]
    projected_fcff: list[float]
    assumptions: DCFAssumptions
    current_price: float = 0
    upside_pct: float = 0


def calculate_wacc(assumptions: DCFAssumptions) -> float:
    """Calculate Weighted Average Cost of Capital."""
    # Cost of equity (CAPM)
    cost_of_equity = (
        assumptions.risk_free_rate
        + assumptions.beta * assumptions.market_risk_premium
    )

    # After-tax cost of debt
    after_tax_cost_of_debt = assumptions.cost_of_debt * (1 - assumptions.tax_rate)

    # WACC
    equity_ratio = 1 - assumptions.debt_ratio
    wacc = (
        equity_ratio * cost_of_equity
        + assumptions.debt_ratio * after_tax_cost_of_debt
    )

    logger.info(
        "wacc_calculated",
        cost_of_equity=round(cost_of_equity, 4),
        after_tax_cost_of_debt=round(after_tax_cost_of_debt, 4),
        wacc=round(wacc, 4),
    )
    return wacc


def project_revenue(assumptions: DCFAssumptions) -> list[float]:
    """Project revenue for each year."""
    revenues = []
    current = assumptions.revenue_base
    for i in range(assumptions.projection_years):
        if i < len(assumptions.revenue_growth_rates):
            growth = assumptions.revenue_growth_rates[i]
        else:
            # Fade growth toward terminal rate
            last_growth = assumptions.revenue_growth_rates[-1] if assumptions.revenue_growth_rates else 0.05
            fade_factor = (i - len(assumptions.revenue_growth_rates) + 1) / (
                assumptions.projection_years - len(assumptions.revenue_growth_rates) + 1
            )
            growth = last_growth + fade_factor * (assumptions.terminal_growth_rate - last_growth)

        current = current * (1 + growth)
        revenues.append(current)

    return revenues


def project_fcff(revenues: list[float], assumptions: DCFAssumptions) -> list[float]:
    """Project Free Cash Flow to Firm for each year.

    FCFF = EBIT × (1 - tax) + D&A - CapEx - ΔWC
    """
    fcffs = []
    for rev in revenues:
        ebit = rev * assumptions.operating_margin
        nopat = ebit * (1 - assumptions.tax_rate)
        depreciation = rev * assumptions.depreciation_to_revenue
        capex = rev * assumptions.capex_to_revenue
        nwc_change = rev * assumptions.nwc_change_to_revenue

        fcff = nopat + depreciation - capex - nwc_change
        fcffs.append(fcff)

    return fcffs


def calculate_terminal_value(
    last_fcff: float, wacc: float, assumptions: DCFAssumptions
) -> float:
    """Calculate terminal value using Gordon Growth or Exit Multiple."""
    if assumptions.terminal_method == "exit_multiple":
        # Terminal EV = EBITDA × Multiple
        # Approximate EBITDA from FCFF
        last_revenue = assumptions.revenue_base
        for g in assumptions.revenue_growth_rates:
            last_revenue *= (1 + g)
        ebitda = last_revenue * (assumptions.operating_margin + assumptions.depreciation_to_revenue)
        return ebitda * assumptions.exit_ev_ebitda_multiple

    # Gordon Growth Model: TV = FCFF × (1 + g) / (WACC - g)
    if wacc <= assumptions.terminal_growth_rate:
        logger.warning("wacc_below_terminal_growth", wacc=wacc, g=assumptions.terminal_growth_rate)
        return last_fcff * 20  # Fallback cap

    return last_fcff * (1 + assumptions.terminal_growth_rate) / (wacc - assumptions.terminal_growth_rate)


def run_dcf(assumptions: DCFAssumptions, current_price: float = 0) -> DCFResult:
    """Run the full DCF model and return fair value per share."""
    logger.info("running_dcf", method="FCFF 2-stage")

    wacc = calculate_wacc(assumptions)
    revenues = project_revenue(assumptions)
    fcffs = project_fcff(revenues, assumptions)

    # Discount FCFFs to present value
    pv_fcffs = [
        fcff / (1 + wacc) ** (i + 1) for i, fcff in enumerate(fcffs)
    ]
    sum_pv_fcff = sum(pv_fcffs)

    # Terminal value
    terminal_value = calculate_terminal_value(fcffs[-1], wacc, assumptions)
    pv_terminal = terminal_value / (1 + wacc) ** assumptions.projection_years

    # Enterprise value
    enterprise_value = sum_pv_fcff + pv_terminal

    # Equity value
    equity_value = enterprise_value - assumptions.net_debt

    # Fair value per share
    fair_value = equity_value / assumptions.shares_outstanding if assumptions.shares_outstanding > 0 else 0

    upside = safe_div(fair_value - current_price, current_price) if current_price > 0 else None

    result = DCFResult(
        fair_value_per_share=round(fair_value, 2),
        enterprise_value=round(enterprise_value, 2),
        equity_value=round(equity_value, 2),
        wacc=round(wacc, 4),
        terminal_value=round(terminal_value, 2),
        pv_fcff=[round(pv, 2) for pv in pv_fcffs],
        projected_revenue=[round(r, 2) for r in revenues],
        projected_fcff=[round(f, 2) for f in fcffs],
        assumptions=assumptions,
        current_price=current_price,
        upside_pct=round(upside, 4) if upside else 0,
    )

    logger.info(
        "dcf_result",
        fair_value=result.fair_value_per_share,
        ev=result.enterprise_value,
        upside_pct=result.upside_pct,
    )
    return result


def build_assumptions_from_data(
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
) -> DCFAssumptions:
    """Build DCF assumptions from fetched data — automatic calibration."""
    info = market_data.get("info", {})
    financials = sec_data.get("financials", {})

    # Revenue base
    revenue_data = financials.get("revenue", [])
    revenue_base = revenue_data[0]["value"] if revenue_data else 0

    # Revenue growth rates (from historical, fading forward)
    growth_rates = []
    if len(revenue_data) >= 2:
        for i in range(min(3, len(revenue_data) - 1)):
            g = safe_div(
                revenue_data[i]["value"] - revenue_data[i + 1]["value"],
                abs(revenue_data[i + 1]["value"]),
            )
            if g is not None:
                growth_rates.append(g)
        growth_rates.reverse()  # Chronological order

    if not growth_rates:
        growth_rates = [0.05, 0.05, 0.04, 0.03, 0.025]

    # Operating margin
    op_income_data = financials.get("operating_income", [])
    if op_income_data and revenue_data:
        op_margin = safe_div(op_income_data[0]["value"], revenue_data[0]["value"])
    else:
        op_margin = info.get("operating_margin") or 0.20

    # Beta
    beta = info.get("beta", 1.0) or 1.0

    # Debt info
    total_debt = financials.get("total_debt") or []
    debt_val = (total_debt[0].get("value") if total_debt else 0) or 0
    cash_data = financials.get("cash") or []
    cash_val = (cash_data[0].get("value") if cash_data else 0) or 0
    equity_data = financials.get("stockholders_equity") or []
    equity_val = (equity_data[0].get("value") if equity_data else 0) or 1

    debt_ratio = safe_div(debt_val, debt_val + equity_val) if (debt_val + equity_val) > 0 else 0.2

    # Risk-free rate from macro data
    rfr = macro_data.get("risk_free_rate", settings.dcf.risk_free_rate_manual)

    # Shares outstanding
    shares = info.get("shares_outstanding", 0) or 0

    return DCFAssumptions(
        revenue_base=revenue_base,
        revenue_growth_rates=growth_rates,
        operating_margin=op_margin or 0.20,
        tax_rate=settings.dcf.tax_rate,
        risk_free_rate=rfr,
        beta=beta,
        market_risk_premium=settings.dcf.market_risk_premium,
        debt_ratio=debt_ratio or 0.2,
        terminal_growth_rate=settings.dcf.terminal_growth_rate,
        shares_outstanding=shares,
        net_debt=debt_val - cash_val,
        projection_years=settings.dcf.projection_years,
    )
