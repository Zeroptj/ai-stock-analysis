"""Full analysis entry point — combines valuation + LLM narrative.

Usage: python -m stock_analysis.llm.analyze MSFT
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from stock_analysis.calc_engine.valuate import valuate_ticker
from stock_analysis.database import init_db
from stock_analysis.llm.groq_client import (
    generate_annual_report_summary,
    generate_industry_commentary,
    generate_meeting_synthesis,
    generate_risk_analysis,
    generate_thesis,
    run_qc_check,
)
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


def _slim_primary(primary: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields (sensitivity grids, raw assumption arrays) from the
    primary valuation payload before sending to the LLM."""
    if not primary:
        return {}
    slim = {k: v for k, v in primary.items() if k not in ("sensitivity", "assumptions")}
    a = primary.get("assumptions", {}) or {}
    slim["key_assumptions"] = {
        k: a.get(k)
        for k in (
            "wacc", "terminal_growth", "cost_of_equity", "beta",
            "revenue_growth", "operating_margin", "tax_rate",
            "roe", "payout_ratio", "dividend_growth_near", "dividend_growth_terminal",
            "shares_outstanding", "book_value_per_share",
        )
        if k in a
    }
    return slim


def _slim_valuation_bundle(valuation: dict[str, Any]) -> dict[str, Any]:
    """Strip SEC filing text + sensitivity grids from the valuation bundle."""
    b = dict(valuation)
    b["valuation"] = _slim_primary(b.get("valuation", {}))
    b["sec_narrative"] = {
        "annual_filings": [
            {k: v for k, v in (f or {}).items() if k != "text"}
            for f in (valuation.get("sec_narrative", {}) or {}).get("annual_filings", [])
        ],
    }
    comps = b.get("comparables", {}) or {}
    b["comparables"] = {
        "target_multiples": comps.get("target_multiples", {}),
        "implied_values": comps.get("implied_values", {}),
        "peers": (comps.get("comp_summary", {}) or {}).get("peers", []),
    }
    b["financials_history"] = (b.get("financials_history") or [])[:5]
    # Quarterly data is verbose (12 quarters × 30+ metrics) — drop for LLM,
    # keep only in full report bundle.
    b.pop("quarterly_financials", None)
    # Ownership & earnings calendar are for the PDF, not QC — drop to fit
    # llama-3.1-8b-instant TPM budget (6K).
    b.pop("ownership", None)
    b.pop("earnings_calendar", None)
    # Ratios are already surfaced in the PDF; QC only needs valuation math.
    b.pop("ratios", None)
    return b


async def analyze_ticker(ticker: str) -> dict[str, Any]:
    """Run full analysis: valuation + LLM narrative."""
    init_db()
    logger.info("analyze_start", ticker=ticker)

    # Step 1: Run valuation
    valuation = await valuate_ticker(ticker)

    # Extract data for LLM
    company_name = valuation.get("company_name", ticker)
    current_price = valuation.get("current_price", 0)
    primary_val = valuation.get("valuation", {})
    comps = valuation.get("comparables", {})
    sector = valuation.get("sector") or comps.get("sector", "")

    # Slimmed payloads for the LLM (keep prompts under rate-limit TPM)
    primary_slim = _slim_primary(primary_val)
    valuation_slim = _slim_valuation_bundle(valuation)

    # Step 2: LLM narrative generation
    logger.info("generating_narratives", ticker=ticker)

    thesis = generate_thesis(
        ticker=ticker,
        company_name=company_name,
        financials=primary_slim,
        valuation=valuation_slim,
        market_data={"current_price": current_price},
    )

    risks = generate_risk_analysis(
        ticker=ticker,
        company_name=company_name,
        financials=primary_slim,
        sector=sector,
    )

    industry = generate_industry_commentary(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        comparables=comps,
    )

    # Step 3: SEC narrative — 10-K summary + Meeting / material-event synthesis
    sec_narrative = valuation.get("sec_narrative", {})
    annual_filings = sec_narrative.get("annual_filings", []) or []
    proxy_statements = sec_narrative.get("proxy_statements", []) or []
    material_events = sec_narrative.get("material_events", []) or []

    try:
        annual_summary = generate_annual_report_summary(
            ticker=ticker,
            company_name=company_name,
            annual_filings=annual_filings,
        )
    except Exception as e:
        logger.warning("annual_summary_failed", ticker=ticker, error=str(e))
        annual_summary = None

    if proxy_statements or material_events:
        meetings = generate_meeting_synthesis(
            ticker=ticker,
            company_name=company_name,
            proxy_statements=proxy_statements,
            material_events=material_events,
        )
    else:
        meetings = {"proxy_summary": None, "material_events_summary": [],
                    "key_takeaways_for_investors": []}

    # Step 4: QC check (slim payload — don't re-send SEC text)
    qc = run_qc_check(
        ticker=ticker,
        valuation_result=valuation_slim,
        market_data={"current_price": current_price},
    )

    result = {
        "ticker": ticker.upper(),
        "current_price": current_price,
        "valuation": valuation,
        "thesis": thesis,
        "risks": risks,
        "industry": industry,
        "annual_summary": annual_summary,
        "meetings": meetings,
        "qc_check": qc,
    }

    logger.info("analyze_complete", ticker=ticker, qc_passed=qc.get("passed", False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Full stock analysis with LLM narrative")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. MSFT)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()

    result = asyncio.run(analyze_ticker(args.ticker))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Analysis saved to {args.output}")
    else:
        thesis = result.get("thesis", {})
        print(f"\n{'='*60}")
        print(f"  Analysis: {result['ticker']}")
        print(f"{'='*60}")
        print(f"  Rating: {thesis.get('rating', 'N/A')} ({thesis.get('conviction', 'N/A')} conviction)")
        print(f"  Summary: {thesis.get('one_line_summary', 'N/A')}")
        print(f"\n  QC Check: {'PASSED' if result['qc_check'].get('passed') else 'FLAGS FOUND'}")

        flags = result["qc_check"].get("flags", [])
        for flag in flags:
            print(f"    - [{flag['severity']}] {flag['issue']}")


if __name__ == "__main__":
    main()
