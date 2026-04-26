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


def _humanize_dollar(v: Any) -> str | None:
    """Format raw dollar amounts as compact strings before sending to the LLM,
    so the model never has to count digits or pick the right magnitude.
    Big enterprise values like 1,951,968,802,785.68 are easy for an LLM to
    misread (we've seen "$195.2B" hallucinated from $1.95T)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}${abs_v/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v/1e6:.2f}M"
    return f"{sign}${abs_v:,.2f}"


def _humanize_pct(v: Any) -> str | None:
    """Format a decimal share (0.3752) as a labeled percent string ('37.52%').
    LLM must not be left to multiply or divide by 100."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return f"{v*100:+.2f}%"


_DOLLAR_FIELDS = (
    "enterprise_value", "equity_value", "terminal_value",
    "normalized_revenue", "market_cap",
)


def _slim_primary(primary: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields (sensitivity grids, raw assumption arrays) from the
    primary valuation payload before sending to the LLM. Also reformat raw
    decimals/large numbers into human-readable strings so the model can quote
    them verbatim without arithmetic."""
    if not primary:
        return {}
    slim = {k: v for k, v in primary.items() if k not in ("sensitivity", "assumptions")}

    # Convert dollar amounts to compact strings ($1.95T) — keeps both raw and
    # display so downstream calc code still works if it imports this helper.
    for field in _DOLLAR_FIELDS:
        if slim.get(field) is not None:
            display = _humanize_dollar(slim[field])
            if display:
                slim[f"{field}_display"] = display

    # Fair value per share is normal-magnitude — quote with 2 dp.
    fv = slim.get("fair_value")
    if fv is not None:
        try:
            slim["fair_value_display"] = f"${float(fv):,.2f}"
        except (TypeError, ValueError):
            pass

    # Upside/downside is a decimal share — express as labeled percent so the
    # LLM cannot re-scale it (Bug: 0.3752 → "0.3752% downside" instead of
    # "37.52% downside").
    upside = slim.get("upside_pct")
    if upside is not None:
        try:
            up = float(upside)
            slim["upside_pct_display"] = f"{up*100:+.2f}%"
            slim["fair_value_vs_current"] = (
                f"Fair value implies a {abs(up)*100:.2f}% "
                f"{'upside' if up > 0 else 'downside'} from the current price."
            )
        except (TypeError, ValueError):
            pass

    # WACC / cost of equity are also decimals
    for k in ("wacc", "cost_of_equity", "normalized_margin"):
        if slim.get(k) is not None:
            display = _humanize_pct(slim[k])
            if display:
                slim[f"{k}_display"] = display

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
    """Strip SEC filing text + sensitivity grids from the valuation bundle and
    add display strings for the headline numbers QC tends to misread.

    The QC LLM has historically conflated per-share fair value with company-
    level enterprise value or market cap. We rename the display fields with
    explicit `_per_share` / `_company` suffixes and pre-compute the only
    valid headline comparison (per-share fair value vs per-share market price)
    so the model can quote it instead of inventing one.
    """
    b = dict(valuation)
    b["valuation"] = _slim_primary(b.get("valuation", {}))

    primary = b["valuation"]
    cur_price = b.get("current_price") or valuation.get("current_price")

    # Compute company-level market cap (shares × price) to show alongside EV.
    assumptions = (valuation.get("valuation", {}) or {}).get("assumptions", {}) or {}
    shares = assumptions.get("shares_outstanding")
    market_cap_company = None
    if shares and cur_price:
        try:
            market_cap_company = float(shares) * float(cur_price)
        except (TypeError, ValueError):
            market_cap_company = None

    def _humanize(v):
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        abs_v = abs(v)
        sign = "-" if v < 0 else ""
        if abs_v >= 1e12: return f"{sign}${abs_v/1e12:.2f}T"
        if abs_v >= 1e9:  return f"{sign}${abs_v/1e9:.2f}B"
        if abs_v >= 1e6:  return f"{sign}${abs_v/1e6:.2f}M"
        return f"{sign}${abs_v:,.2f}"

    # Self-documenting field names so the LLM cannot mix per-share with
    # company-level totals in the same comparison.
    per_share = {
        "current_price_per_share": f"${float(cur_price):,.2f}" if cur_price else None,
        "fair_value_per_share": primary.get("fair_value_display"),
        "fair_value_vs_current_price": primary.get("fair_value_vs_current"),
    }
    company_level = {
        "market_cap_company": _humanize(market_cap_company),
        "enterprise_value_company": primary.get("enterprise_value_display"),
        "equity_value_company": primary.get("equity_value_display"),
    }
    rates = {
        "wacc": primary.get("wacc_display"),
        "cost_of_equity": primary.get("cost_of_equity_display"),
    }

    b["display_summary"] = {
        "per_share": {k: v for k, v in per_share.items() if v},
        "company_level": {k: v for k, v in company_level.items() if v},
        "rates": {k: v for k, v in rates.items() if v},
        "valid_comparisons": [
            "Per-share fair value vs per-share current price → "
            + (primary.get("fair_value_vs_current") or "n/a"),
        ],
        "do_not_compare": [
            "Per-share fair value with company-level market cap or enterprise value",
            "Market cap with enterprise value (they differ by net debt; comparing them is meaningless)",
        ],
    }
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

    # Pass the thesis stance into industry commentary so the peer narrative
    # doesn't contradict the SELL/BUY rating from the thesis pass.
    industry = generate_industry_commentary(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        comparables=comps,
        thesis_rating=(thesis or {}).get("rating"),
        thesis_summary=(thesis or {}).get("one_line_summary"),
        fair_value_vs_current=primary_slim.get("fair_value_vs_current"),
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
