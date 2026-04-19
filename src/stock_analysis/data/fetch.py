"""CLI entry point — fetch all data for a ticker.

Usage: python -m stock_analysis.data.fetch AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from stock_analysis.data.macro_data import fetch_macro_data
from stock_analysis.data.market_data_client import fetch_market_data
from stock_analysis.data.sec_edgar_client import fetch_sec_data
from stock_analysis.database import init_db
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


async def fetch_all(ticker: str) -> dict:
    """Fetch data from all sources for a given ticker."""
    init_db()
    logger.info("fetch_all_start", ticker=ticker)

    # Fetch SEC data (async) and market + macro data (sync) concurrently
    async with httpx.AsyncClient(timeout=30.0) as client:
        sec_task = asyncio.create_task(fetch_sec_data(ticker, client))

        # Run sync functions in executor
        loop = asyncio.get_event_loop()
        market_task = loop.run_in_executor(None, fetch_market_data, ticker)
        macro_task = loop.run_in_executor(None, fetch_macro_data)

        sec_data = await sec_task
        market_data = await market_task
        macro_data = await macro_task

    result = {
        "ticker": ticker.upper(),
        "sec": sec_data,
        "market": market_data,
        "macro": macro_data,
    }

    logger.info("fetch_all_complete", ticker=ticker)
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data from all sources")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    args = parser.parse_args()

    data = asyncio.run(fetch_all(args.ticker))

    json_str = json.dumps(data, indent=2 if args.pretty else None, default=str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"Data saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
