"""Valuation entry point — industry-aware router + full valuation bundle.

Dispatches by `industry_classifier.classify_ticker`:
    DCF_FCFF        → dcf_fcff (Tech, Healthcare, Consumer, Industrials)
    DDM             → dcf_ddm (Utilities, stable dividend payers)
    RESIDUAL_INCOME → residual_income (Banks, Financials)
    AFFO_REIT       → affo_reit (REITs)
    DCF_CYCLICAL    → dcf_cyclical (Energy, Materials, Mining)

All numbers are Python-computed; LLM does not touch valuation math.

Usage: python -m stock_analysis.calc_engine.valuate MSFT
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from typing import Any

from stock_analysis.calc_engine.affo_reit import build_affo_assumptions, run_affo_valuation
from stock_analysis.calc_engine.comparables import run_comparables
from stock_analysis.calc_engine.dcf_cyclical import (
    build_cyclical_assumptions,
    run_cyclical_dcf,
)
from stock_analysis.calc_engine.dcf_ddm import build_ddm_assumptions, run_ddm
from stock_analysis.calc_engine.dcf_fcff import (
    DCFAssumptions,
    build_assumptions_from_data,
    run_dcf,
)
from stock_analysis.calc_engine.industry_classifier import (
    ValuationModel,
    classify_ticker,
    get_model_description,
)
from stock_analysis.calc_engine.ratios import calculate_all_ratios
from stock_analysis.calc_engine.residual_income import (
    build_ri_assumptions,
    run_residual_income,
)
from stock_analysis.calc_engine.sensitivity import build_sensitivity_table
from stock_analysis.data.fetch import fetch_all
from stock_analysis.database import init_db
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


def _flatten_latest(financials: dict[str, Any]) -> dict[str, float]:
    """Pull the most recent value for each key in SEC financial facts dict."""
    flat: dict[str, float] = {}
    for key, vals in financials.items():
        if isinstance(vals, list) and vals:
            flat[key] = vals[0].get("value") or 0
    flat["total_debt"] = flat.get("total_debt", 0)
    flat["cash"] = flat.get("cash", 0)
    return flat


def _historical_table(financials: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Build a historical table: [{period_end, revenue, net_income, ...}]."""
    period_map: dict[str, dict[str, Any]] = {}
    for key in keys:
        for entry in financials.get(key, []) or []:
            period = entry.get("period_end")
            if not period:
                continue
            period_map.setdefault(period, {"period_end": period})[key] = entry.get("value")
    rows = list(period_map.values())
    rows.sort(key=lambda r: r["period_end"], reverse=True)
    return rows[:10]


def _run_by_model(
    model: ValuationModel,
    ticker: str,
    sec_data: dict[str, Any],
    market_data: dict[str, Any],
    macro_data: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """Route to the right valuation model and return a normalized payload."""
    if model == ValuationModel.DCF_FCFF:
        a = build_assumptions_from_data(sec_data, market_data, macro_data)
        r = run_dcf(a, current_price)
        sensitivity = build_sensitivity_table(a, current_price)
        return {
            "model": model.value,
            "fair_value": r.fair_value_per_share,
            "upside_pct": r.upside_pct,
            "wacc": r.wacc,
            "enterprise_value": r.enterprise_value,
            "equity_value": r.equity_value,
            "terminal_value": r.terminal_value,
            "projected_revenue": r.projected_revenue,
            "projected_fcff": r.projected_fcff,
            "assumptions": asdict(a),
            "sensitivity": sensitivity,
        }

    if model == ValuationModel.DDM:
        a = build_ddm_assumptions(sec_data, market_data, macro_data)
        r = run_ddm(a, current_price)
        return {
            "model": model.value,
            "fair_value": r.fair_value_per_share,
            "upside_pct": r.upside_pct,
            "cost_of_equity": r.cost_of_equity,
            "projected_dividends": r.projected_dividends,
            "pv_dividends": r.pv_dividends,
            "terminal_value": r.terminal_value,
            "assumptions": asdict(a),
        }

    if model == ValuationModel.RESIDUAL_INCOME:
        a = build_ri_assumptions(sec_data, market_data, macro_data)
        r = run_residual_income(a, current_price)
        return {
            "model": model.value,
            "fair_value": r.fair_value_per_share,
            "upside_pct": r.upside_pct,
            "book_value_component": r.book_value_component,
            "pv_residual_income": r.pv_residual_income,
            "terminal_component": r.terminal_component,
            "projected_bvps": r.projected_bvps,
            "projected_eps": r.projected_eps,
            "projected_ri": r.projected_ri,
            "assumptions": asdict(a),
        }

    if model == ValuationModel.AFFO_REIT:
        a = build_affo_assumptions(sec_data, market_data, macro_data)
        r = run_affo_valuation(a, current_price)
        return {
            "model": model.value,
            "fair_value": r.fair_value_per_share,
            "upside_pct": r.upside_pct,
            "p_affo_implied": r.p_affo_implied,
            "projected_affo": r.projected_affo,
            "projected_dividends": r.projected_dividends,
            "terminal_value": r.terminal_value,
            "nav_fair_value": r.nav_fair_value,
            "assumptions": asdict(a),
        }

    if model == ValuationModel.DCF_CYCLICAL:
        a = build_cyclical_assumptions(sec_data, market_data, macro_data)
        r = run_cyclical_dcf(a, current_price)
        return {
            "model": model.value,
            "fair_value": r.fair_value_per_share,
            "upside_pct": r.upside_pct,
            "normalized_revenue": r.normalized_revenue,
            "normalized_margin": r.normalized_margin,
            "cycle_position": r.cycle_position,
            "cycle_adjustment": r.cycle_adjustment,
            "assumptions": asdict(a.dcf_base),
        }

    raise ValueError(f"Unsupported model: {model}")


async def valuate_ticker(ticker: str) -> dict[str, Any]:
    """Run industry-appropriate valuation + comparables + sensitivity."""
    init_db()

    data = await fetch_all(ticker)
    sec_data = data["sec"]
    market_data = data["market"]
    macro_data = data["macro"]

    info = market_data.get("info", {})
    current_price = info.get("current_price", 0)

    # Classify → pick model
    sector = info.get("sector", "")
    sub_industry = info.get("industry", "")
    model = classify_ticker(ticker, sector=sector, sub_industry=sub_industry)
    logger.info("valuation_route", ticker=ticker, model=model.value, sector=sector)

    # Run the appropriate model
    primary = _run_by_model(
        model, ticker, sec_data, market_data, macro_data, current_price
    )

    # Comparables (universal)
    financials_flat = _flatten_latest(sec_data.get("financials", {}))
    comps = run_comparables(ticker, financials_flat, market_data)

    # Derived ratios (profitability / returns / leverage / efficiency / multiples)
    ratios = calculate_all_ratios(
        financials={**financials_flat, "ticker": ticker},
        market_data=info,
    )

    # Historical financials table for the report
    financials_history = _historical_table(
        sec_data.get("financials", {}),
        keys=(
            "revenue", "gross_profit", "operating_income", "net_income",
            "operating_cash_flow", "capex", "total_assets", "total_liabilities",
            "stockholders_equity", "eps",
        ),
    )

    # SEC narrative context (proxy + material events)
    sec_narrative = {
        "proxy_statements": sec_data.get("proxy_statements", []),
        "material_events": sec_data.get("material_events", []),
        "annual_filings": [
            {k: v for k, v in f.items() if k != "text"}
            for f in sec_data.get("annual_filings", [])
        ],
    }

    return {
        "ticker": ticker.upper(),
        "company_name": sec_data.get("company_name") or info.get("short_name", ticker),
        "sector": sector,
        "sub_industry": sub_industry,
        "current_price": current_price,
        "model_used": model.value,
        "model_description": get_model_description(model),
        "valuation": primary,                 # model-specific payload
        "comparables": comps,
        "ratios": ratios,
        "financials_history": financials_history,
        "quarterly_financials": sec_data.get("quarterly_financials", {}),
        "sec_narrative": sec_narrative,
        "ownership": market_data.get("ownership", {}),
        "earnings_calendar": market_data.get("earnings_calendar", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Industry-aware stock valuation")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. MSFT)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()

    result = asyncio.run(valuate_ticker(args.ticker))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Valuation saved to {args.output}")
    else:
        v = result["valuation"]
        print(f"\n{'='*70}")
        print(f"  Valuation: {result['ticker']}  |  {result['sector']}")
        print(f"  Model: {result['model_description']}")
        print(f"{'='*70}")
        print(f"  Current Price:  ${result['current_price']:.2f}")
        print(f"  Fair Value:     ${v.get('fair_value', 0):.2f}")
        up = v.get('upside_pct', 0) or 0
        print(f"  Upside:         {up*100:+.1f}%")
        if "wacc" in v:
            print(f"  WACC:           {v['wacc']*100:.2f}%")
        if "cost_of_equity" in v:
            print(f"  Cost of Equity: {v['cost_of_equity']*100:.2f}%")
        if "cycle_position" in v:
            print(f"  Cycle Position: {v['cycle_position']}")


if __name__ == "__main__":
    main()
