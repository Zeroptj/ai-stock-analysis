"""End-to-end orchestrator — screen → analyze → report.

Usage:
    python -m stock_analysis.run                    # Full pipeline
    python -m stock_analysis.run --ticker AAPL       # Single ticker
    python -m stock_analysis.run --tickers AAPL,MSFT # Multiple tickers
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from stock_analysis.calc_engine.industry_classifier import classify_ticker, get_model_description
from stock_analysis.database import init_db
from stock_analysis.llm.analyze import analyze_ticker
from stock_analysis.logging import get_logger
from stock_analysis.report.generator import generate_pdf
from stock_analysis.screening.screener import run_screener

logger = get_logger(__name__)


async def run_pipeline(
    tickers: list[str] | None = None,
    skip_screening: bool = False,
    skip_llm: bool = False,
    output_dir: str = "output",
    movers: str = "both",
) -> list[dict[str, Any]]:
    """Run the full analysis pipeline.

    1. Screen S&P 500 (or use provided tickers)
    2. For each candidate: fetch data → valuate → LLM narrative
    3. Generate PDF reports
    """
    init_db()
    logger.info("pipeline_start")

    # Step 1: Get tickers
    if tickers:
        candidates = [{"ticker": t} for t in tickers]
        logger.info("using_provided_tickers", count=len(candidates))
    elif not skip_screening:
        logger.info("running_weekly_movers_screening", movers=movers)
        candidates = run_screener(movers=movers)
        gainers = sum(1 for c in candidates if c.get("mover_type") == "gainer")
        losers = sum(1 for c in candidates if c.get("mover_type") == "loser")
        logger.info(
            "screening_done",
            total=len(candidates),
            gainers=gainers,
            losers=losers,
        )
    else:
        logger.error("no_tickers_provided")
        return []

    # Step 2: Analyze each ticker
    results = []
    for i, candidate in enumerate(candidates):
        ticker = candidate.get("ticker", candidate) if isinstance(candidate, dict) else candidate
        logger.info("analyzing_ticker", ticker=ticker, progress=f"{i+1}/{len(candidates)}")

        try:
            # Classify model type
            model_type = classify_ticker(ticker)
            logger.info("model_selected", ticker=ticker, model=model_type.value)

            # Full analysis
            analysis = await analyze_ticker(ticker)
            analysis["valuation_model"] = model_type.value
            analysis["valuation_model_description"] = get_model_description(model_type)

            # Carry screening context (mover_type, pct_change, ...) into the report
            if isinstance(candidate, dict):
                analysis["screening"] = {
                    k: candidate.get(k)
                    for k in ("mover_type", "pct_change", "price_start",
                              "avg_dollar_volume")
                    if k in candidate
                }

            # Generate PDF — sync Playwright won't run inside the asyncio loop,
            # so dispatch to a worker thread.
            output_path = await asyncio.to_thread(generate_pdf, analysis)
            analysis["report_path"] = str(output_path)

            results.append(analysis)
            logger.info("ticker_complete", ticker=ticker, report=str(output_path))

        except Exception as e:
            logger.error("ticker_failed", ticker=ticker, error=str(e))
            results.append({"ticker": ticker, "error": str(e)})

    # Summary
    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    logger.info("pipeline_complete", success=len(success), failed=len(failed))

    # Save summary
    summary_path = Path(output_dir) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total": len(results),
                "success": len(success),
                "failed": len(failed),
                "tickers": [
                    {
                        "ticker": r.get("ticker"),
                        "status": "success" if "error" not in r else "failed",
                        "report": r.get("report_path"),
                        "error": r.get("error"),
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="AI Stock Analysis — End-to-end pipeline"
    )
    parser.add_argument("--ticker", "-t", help="Single ticker to analyze")
    parser.add_argument(
        "--tickers", help="Comma-separated list of tickers (e.g. AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--skip-screening", action="store_true", help="Skip S&P 500 screening"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM narrative generation"
    )
    parser.add_argument(
        "--output", "-o", default="output", help="Output directory"
    )
    parser.add_argument(
        "--movers",
        choices=["gainers", "losers", "both"],
        default="both",
        help="Which side of the weekly movers to analyze (default: both)",
    )
    args = parser.parse_args()

    tickers = None
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    results = asyncio.run(
        run_pipeline(
            tickers=tickers,
            skip_screening=args.skip_screening,
            skip_llm=args.skip_llm,
            output_dir=args.output,
            movers=args.movers,
        )
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")

    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"  Total:   {len(results)}")
    print(f"  Success: {len(success)}")
    print(f"  Failed:  {len(failed)}")

    if success:
        print(f"\n  Reports generated:")
        for r in success:
            print(f"    - {r['ticker']}: {r.get('report_path', 'N/A')}")

    if failed:
        print(f"\n  Failed tickers:")
        for r in failed:
            print(f"    - {r['ticker']}: {r['error']}")


if __name__ == "__main__":
    main()
