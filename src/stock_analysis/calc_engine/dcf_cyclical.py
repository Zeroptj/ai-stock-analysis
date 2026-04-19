"""Normalized DCF for cyclical industries — Energy, Materials, Mining.

Uses mid-cycle normalized earnings instead of current-year financials
to avoid over/undervaluation at cycle peaks/troughs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from stock_analysis.calc_engine.dcf_fcff import DCFAssumptions, calculate_wacc, run_dcf
from stock_analysis.calc_engine.ratios import safe_div
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class CyclicalAssumptions:
    """Assumptions for cyclical DCF — includes normalization."""

    # Historical data for normalization
    historical_revenue: list[float] = field(default_factory=list)  # 5-10 years
    historical_margins: list[float] = field(default_factory=list)
    historical_capex_ratio: list[float] = field(default_factory=list)

    # Normalized (mid-cycle) values
    normalized_revenue: float = 0
    normalized_margin: float = 0
    normalized_capex_ratio: float = 0

    # Cycle position
    cycle_position: str = "mid"  # peak, mid, trough

    # DCF base assumptions (reuse)
    dcf_base: DCFAssumptions = field(default_factory=DCFAssumptions)


@dataclass
class CyclicalResult:
    """Result of cyclical DCF."""

    fair_value_per_share: float
    normalized_revenue: float
    normalized_margin: float
    cycle_position: str
    dcf_fair_value: float  # From standard DCF with normalized inputs
    current_price: float = 0
    upside_pct: float = 0
    cycle_adjustment: str = ""


def normalize_revenue(historical: list[float]) -> float:
    """Calculate mid-cycle normalized revenue using trimmed mean."""
    if not historical:
        return 0
    if len(historical) <= 2:
        return float(np.mean(historical))

    # Trimmed mean: remove highest and lowest
    sorted_rev = sorted(historical)
    trimmed = sorted_rev[1:-1]
    return float(np.mean(trimmed))


def normalize_margin(historical: list[float]) -> float:
    """Calculate mid-cycle normalized margin."""
    if not historical:
        return 0.10
    return float(np.median(historical))


def detect_cycle_position(
    historical_revenue: list[float], current_revenue: float
) -> str:
    """Detect where in the cycle the company currently is."""
    if not historical_revenue:
        return "mid"

    avg = np.mean(historical_revenue)
    std = np.std(historical_revenue)

    if std == 0:
        return "mid"

    z_score = (current_revenue - avg) / std

    if z_score > 0.75:
        return "peak"
    elif z_score < -0.75:
        return "trough"
    return "mid"


def run_cyclical_dcf(
    assumptions: CyclicalAssumptions, current_price: float = 0
) -> CyclicalResult:
    """Run normalized DCF for cyclical company."""
    logger.info("running_cyclical_dcf")

    # Normalize
    norm_rev = assumptions.normalized_revenue or normalize_revenue(
        assumptions.historical_revenue
    )
    norm_margin = assumptions.normalized_margin or normalize_margin(
        assumptions.historical_margins
    )

    # Detect cycle
    current_rev = assumptions.dcf_base.revenue_base
    cycle = detect_cycle_position(assumptions.historical_revenue, current_rev)

    # Adjust revenue base to normalized
    adjusted_dcf = DCFAssumptions(
        revenue_base=norm_rev,
        revenue_growth_rates=assumptions.dcf_base.revenue_growth_rates,
        operating_margin=norm_margin,
        tax_rate=assumptions.dcf_base.tax_rate,
        capex_to_revenue=assumptions.normalized_capex_ratio or assumptions.dcf_base.capex_to_revenue,
        depreciation_to_revenue=assumptions.dcf_base.depreciation_to_revenue,
        nwc_change_to_revenue=assumptions.dcf_base.nwc_change_to_revenue,
        risk_free_rate=assumptions.dcf_base.risk_free_rate,
        beta=assumptions.dcf_base.beta,
        market_risk_premium=assumptions.dcf_base.market_risk_premium,
        cost_of_debt=assumptions.dcf_base.cost_of_debt,
        debt_ratio=assumptions.dcf_base.debt_ratio,
        terminal_growth_rate=min(assumptions.dcf_base.terminal_growth_rate, 0.02),  # Lower for cyclicals
        shares_outstanding=assumptions.dcf_base.shares_outstanding,
        net_debt=assumptions.dcf_base.net_debt,
        projection_years=assumptions.dcf_base.projection_years,
    )

    # Run standard DCF with normalized inputs
    dcf_result = run_dcf(adjusted_dcf, current_price)

    # Cycle adjustment commentary
    if cycle == "peak":
        adjustment = "Current revenue is above mid-cycle — fair value uses normalized (lower) revenue"
    elif cycle == "trough":
        adjustment = "Current revenue is below mid-cycle — fair value uses normalized (higher) revenue"
    else:
        adjustment = "Revenue is near mid-cycle — minimal normalization adjustment"

    upside = safe_div(
        dcf_result.fair_value_per_share - current_price, current_price
    ) if current_price > 0 else None

    result = CyclicalResult(
        fair_value_per_share=dcf_result.fair_value_per_share,
        normalized_revenue=round(norm_rev, 2),
        normalized_margin=round(norm_margin, 4),
        cycle_position=cycle,
        dcf_fair_value=dcf_result.fair_value_per_share,
        current_price=current_price,
        upside_pct=round(upside, 4) if upside else 0,
        cycle_adjustment=adjustment,
    )

    logger.info(
        "cyclical_result",
        fair_value=result.fair_value_per_share,
        cycle=cycle,
        upside=result.upside_pct,
    )
    return result


def build_cyclical_assumptions(
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
) -> CyclicalAssumptions:
    """Build cyclical DCF assumptions from data."""
    from stock_analysis.calc_engine.dcf_fcff import build_assumptions_from_data

    dcf_base = build_assumptions_from_data(sec_data, market_data, macro_data)
    financials = sec_data.get("financials", {})

    # Extract historical revenue
    rev_data = financials.get("revenue", [])
    hist_rev = [r["value"] for r in rev_data if r.get("value")]

    # Extract historical margins
    op_data = financials.get("operating_income", [])
    hist_margins = []
    for i, op in enumerate(op_data):
        if i < len(rev_data) and rev_data[i].get("value") and op.get("value"):
            margin = safe_div(op["value"], rev_data[i]["value"])
            if margin is not None:
                hist_margins.append(margin)

    return CyclicalAssumptions(
        historical_revenue=hist_rev,
        historical_margins=hist_margins,
        dcf_base=dcf_base,
    )
