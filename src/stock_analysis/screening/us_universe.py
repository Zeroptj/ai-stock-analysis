"""US-listed stock universe — fetched from NASDAQ Trader symbol directory.

Sources (public, no auth):
    https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt   (NASDAQ)
    https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt    (NYSE / AMEX / ARCA)

Filters out ETFs, test issues, and symbols containing special chars
(preferred shares, warrants, units — e.g. "BRK.A" is kept but "BRK.A$" dropped).
"""

from __future__ import annotations

import io
from typing import Literal

import httpx
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_UNIVERSE_CACHE: list[dict] | None = None


def _parse_nasdaq_listed(text: str) -> list[dict]:
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    return [
        {
            "ticker": row["Symbol"],
            "name": row["Security Name"],
            "exchange": "NASDAQ",
        }
        for _, row in df.iterrows()
        if isinstance(row["Symbol"], str)
    ]


def _parse_other_listed(text: str) -> list[dict]:
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    exchange_map = {"N": "NYSE", "A": "AMEX", "P": "ARCA", "Z": "BATS"}
    return [
        {
            "ticker": row["ACT Symbol"],
            "name": row["Security Name"],
            "exchange": exchange_map.get(row["Exchange"], row["Exchange"]),
        }
        for _, row in df.iterrows()
        if isinstance(row["ACT Symbol"], str)
    ]


def _is_common_stock(ticker: str, name: str) -> bool:
    """Keep common stocks; drop preferred, warrants, units, rights, notes."""
    if any(ch in ticker for ch in ("$", "=", "^", "+")):
        return False
    name_lower = name.lower()
    blocklist = (
        "preferred", "warrant", " unit", "depositary", "right",
        "% note", "% debenture", "trust preferred",
    )
    return not any(kw in name_lower for kw in blocklist)


def get_us_universe(
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"),
    force_refresh: bool = False,
) -> list[dict]:
    """Return list of US-listed common stocks: [{ticker, name, exchange}, ...]."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None and not force_refresh:
        return [e for e in _UNIVERSE_CACHE if e["exchange"] in exchanges]

    logger.info("fetching_us_universe")
    try:
        with httpx.Client(timeout=30.0) as client:
            nasdaq_resp = client.get(NASDAQ_LISTED_URL)
            nasdaq_resp.raise_for_status()
            other_resp = client.get(OTHER_LISTED_URL)
            other_resp.raise_for_status()

        entries = _parse_nasdaq_listed(nasdaq_resp.text) + _parse_other_listed(other_resp.text)
        entries = [e for e in entries if _is_common_stock(e["ticker"], e["name"])]
        # dedupe by ticker (NASDAQ Symbol may duplicate in otherlisted)
        seen: set[str] = set()
        unique: list[dict] = []
        for e in entries:
            if e["ticker"] not in seen:
                seen.add(e["ticker"])
                unique.append(e)

        _UNIVERSE_CACHE = unique
        logger.info("us_universe_loaded", count=len(unique))
        return [e for e in unique if e["exchange"] in exchanges]
    except Exception as exc:
        logger.error("us_universe_error", error=str(exc))
        return []


def get_us_tickers(
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"),
) -> list[str]:
    """Convenience: return just the ticker strings (yfinance-compatible: '.' → '-')."""
    return [e["ticker"].replace(".", "-") for e in get_us_universe(exchanges)]


def main():
    universe = get_us_universe()
    by_exchange: dict[str, int] = {}
    for e in universe:
        by_exchange[e["exchange"]] = by_exchange.get(e["exchange"], 0) + 1
    print(f"Total US common stocks: {len(universe)}")
    for exch, count in sorted(by_exchange.items()):
        print(f"  {exch}: {count}")


if __name__ == "__main__":
    main()
