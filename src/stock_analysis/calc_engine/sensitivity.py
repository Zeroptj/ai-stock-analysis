"""Sensitivity analysis — 2D tables for DCF assumptions.

Generates WACC × Terminal Growth Rate sensitivity matrix.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import structlog

from stock_analysis.calc_engine.dcf_fcff import DCFAssumptions, run_dcf
from stock_analysis.config import settings

logger = structlog.get_logger(__name__)


def build_sensitivity_table(
    base_assumptions: DCFAssumptions,
    current_price: float,
    wacc_range: list[float] | None = None,
    growth_range: list[float] | None = None,
) -> dict[str, Any]:
    """Build a 2D sensitivity table: WACC (rows) × Terminal Growth (cols).

    Returns fair value per share for each combination.
    """
    if wacc_range is None:
        wacc_range = settings.dcf.wacc_sensitivity_range
    if growth_range is None:
        growth_range = settings.dcf.terminal_growth_sensitivity_range

    logger.info(
        "building_sensitivity_table",
        wacc_range=wacc_range,
        growth_range=growth_range,
    )

    table: list[list[float]] = []

    for wacc in wacc_range:
        row = []
        for tg in growth_range:
            if wacc <= tg:
                row.append(float("inf"))
                continue

            # Override WACC by adjusting beta/debt_ratio to achieve target WACC
            # Simpler approach: directly compute with the target WACC
            modified = replace(
                base_assumptions,
                terminal_growth_rate=tg,
            )
            result = run_dcf(modified, current_price)

            # Adjust for different WACC
            base_wacc = result.wacc
            if base_wacc != wacc and base_wacc > 0:
                # Re-discount with the target WACC
                fcffs = result.projected_fcff
                pv_fcffs = [f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcffs)]

                if fcffs:
                    tv = fcffs[-1] * (1 + tg) / (wacc - tg)
                    pv_tv = tv / (1 + wacc) ** len(fcffs)
                    ev = sum(pv_fcffs) + pv_tv
                    equity = ev - base_assumptions.net_debt
                    fv = equity / base_assumptions.shares_outstanding if base_assumptions.shares_outstanding > 0 else 0
                else:
                    fv = result.fair_value_per_share
            else:
                fv = result.fair_value_per_share

            row.append(round(fv, 2))
        table.append(row)

    # Calculate upside matrix
    upside_table = []
    for row in table:
        upside_row = []
        for fv in row:
            if fv == float("inf") or current_price <= 0:
                upside_row.append(None)
            else:
                upside_row.append(round((fv - current_price) / current_price * 100, 1))
        upside_table.append(upside_row)

    return {
        "wacc_range": [round(w * 100, 1) for w in wacc_range],  # As percentages
        "growth_range": [round(g * 100, 1) for g in growth_range],
        "fair_value_table": table,
        "upside_table": upside_table,
        "current_price": current_price,
        "base_wacc_pct": round(
            (base_assumptions.risk_free_rate + base_assumptions.beta * base_assumptions.market_risk_premium) * 100, 1
        ),
        "base_growth_pct": round(base_assumptions.terminal_growth_rate * 100, 1),
    }


def format_sensitivity_text(sensitivity: dict[str, Any]) -> str:
    """Format sensitivity table as readable text."""
    wacc_range = sensitivity["wacc_range"]
    growth_range = sensitivity["growth_range"]
    table = sensitivity["fair_value_table"]

    lines = []
    lines.append("Sensitivity Analysis: Fair Value per Share")
    lines.append(f"Current Price: ${sensitivity['current_price']:.2f}")
    lines.append("")

    # Header
    header = f"{'WACC':>8s} |" + "".join(f" TG={g:.1f}% " for g in growth_range)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for i, wacc in enumerate(wacc_range):
        row_str = f"{wacc:>7.1f}% |"
        for j, fv in enumerate(table[i]):
            if fv == float("inf"):
                row_str += f"{'N/A':>9s}"
            else:
                row_str += f"${fv:>7.2f} "
        lines.append(row_str)

    return "\n".join(lines)
