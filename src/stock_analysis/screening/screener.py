"""Weekly-movers screener — US-listed top gainers / losers over trailing N days.

Pipeline:
    1. Fetch US common-stock universe (NASDAQ + NYSE + AMEX).
    2. Batch-download recent price history via yfinance (multi-ticker).
    3. Compute % change from `lookback_days` ago → today.
    4. Apply liquidity / price filters.
    5. Return top N gainers + top N losers.

Usage: python -m stock_analysis.screening.screener
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import logging

import pandas as pd
import structlog
import yaml
import yfinance as yf

from stock_analysis.screening.us_universe import get_us_tickers

logger = structlog.get_logger(__name__)

# yfinance chats a lot at INFO when single tickers time out mid-batch.
# Those are expected & recoverable (the ticker is just dropped) — silence them.
logging.getLogger("yfinance").setLevel(logging.ERROR)

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "screening_criteria.yaml"
)


def load_criteria(config_path: Path | None = None) -> dict:
    path = config_path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download_chunk(
    tickers: list[str],
    period: str,
    threads: bool,
) -> pd.DataFrame:
    """Batch download via yfinance. Returns wide DataFrame (ticker × OHLCV)."""
    return yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        threads=threads,
        progress=False,
        auto_adjust=True,
    )


def _compute_mover(
    ticker: str,
    df: pd.DataFrame,
    lookback_days: int,
    min_price: float,
    min_avg_dollar_volume: float,
    min_history_days: int,
) -> dict[str, Any] | None:
    """Compute return/liquidity for a single ticker. None if filtered out."""
    if df is None or df.empty:
        return None
    closes = df["Close"].dropna()
    volumes = df["Volume"].dropna()
    if len(closes) < max(min_history_days, lookback_days + 1):
        return None

    current = float(closes.iloc[-1])
    past = float(closes.iloc[-(lookback_days + 1)])
    if past <= 0 or current <= 0:
        return None
    if current < min_price:
        return None

    avg_dollar_vol = float((closes * volumes).tail(20).mean())
    if avg_dollar_vol < min_avg_dollar_volume:
        return None

    pct = (current - past) / past * 100.0
    return {
        "ticker": ticker,
        "current_price": round(current, 2),
        "price_start": round(past, 2),
        "pct_change": round(pct, 2),
        "avg_dollar_volume": round(avg_dollar_vol, 0),
    }


def _extract_single(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extract a single ticker's OHLCV from a multi-ticker yf.download result."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        return data[ticker]
    return data


def run_weekly_movers(
    criteria: dict | None = None,
    universe: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Screen US universe for top weekly gainers / losers.

    Returns: {"gainers": [...], "losers": [...]}
    """
    criteria = criteria or load_criteria()

    uni_cfg = criteria.get("universe", {})
    mov_cfg = criteria.get("movers", {})
    flt_cfg = criteria.get("filters", {})
    bat_cfg = criteria.get("batch", {})

    lookback = int(mov_cfg.get("lookback_days", 5))
    top_gainers = int(mov_cfg.get("top_gainers", 10))
    top_losers = int(mov_cfg.get("top_losers", 10))

    min_price = float(flt_cfg.get("min_price", 5.0))
    min_adv = float(flt_cfg.get("min_avg_dollar_volume", 5_000_000))
    min_hist = int(flt_cfg.get("min_history_days", 5))

    chunk_size = int(bat_cfg.get("chunk_size", 400))
    threads = bool(bat_cfg.get("threads", True))

    if universe is None:
        exchanges = tuple(uni_cfg.get("exchanges", ["NASDAQ", "NYSE", "AMEX"]))
        universe = get_us_tickers(exchanges=exchanges)
        max_tickers = int(uni_cfg.get("max_tickers", 0))
        if max_tickers > 0:
            universe = universe[:max_tickers]

    logger.info("weekly_movers_start", universe_size=len(universe), lookback_days=lookback)

    # yfinance needs enough history to cover lookback + weekends/holidays
    period = "1mo" if lookback <= 10 else "3mo"

    rows: list[dict[str, Any]] = []
    for i in range(0, len(universe), chunk_size):
        chunk = universe[i : i + chunk_size]
        logger.info(
            "downloading_chunk",
            chunk=i // chunk_size + 1,
            total_chunks=(len(universe) + chunk_size - 1) // chunk_size,
            size=len(chunk),
        )
        try:
            data = _download_chunk(chunk, period=period, threads=threads)
        except Exception as e:
            logger.warning("chunk_download_error", error=str(e))
            continue

        for ticker in chunk:
            df = _extract_single(data, ticker)
            row = _compute_mover(
                ticker, df, lookback, min_price, min_adv, min_hist
            )
            if row:
                rows.append(row)

    rows.sort(key=lambda r: r["pct_change"], reverse=True)
    gainers = rows[:top_gainers]
    losers = list(reversed(rows[-top_losers:])) if top_losers else []

    logger.info(
        "weekly_movers_done",
        valid=len(rows),
        gainers=len(gainers),
        losers=len(losers),
    )
    return {"gainers": gainers, "losers": losers}


def run_screener(
    tickers: list[str] | None = None,
    criteria: dict | None = None,
    movers: str = "both",
) -> list[dict[str, Any]]:
    """Pipeline entry point — returns flat candidate list for run.py.

    movers: "gainers" | "losers" | "both" — which side(s) to return.
    """
    result = run_weekly_movers(criteria=criteria, universe=tickers)
    combined: list[dict[str, Any]] = []
    if movers in ("gainers", "both"):
        for r in result["gainers"]:
            combined.append({**r, "mover_type": "gainer"})
    if movers in ("losers", "both"):
        for r in result["losers"]:
            combined.append({**r, "mover_type": "loser"})
    return combined


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'#':>3}  {'Ticker':<8} {'% Chg':>8}  {'Price':>9}  {'Start':>9}  {'Avg $Vol':>14}")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>3}  {r['ticker']:<8} {r['pct_change']:>+7.2f}%  "
            f"${r['current_price']:>8.2f}  ${r['price_start']:>8.2f}  "
            f"${r['avg_dollar_volume']/1e6:>10.1f}M"
        )


def main():
    parser = argparse.ArgumentParser(description="Weekly movers — top US gainers/losers")
    parser.add_argument("--config", "-c", help="Path to criteria YAML")
    parser.add_argument("--output", "-o", help="Output CSV")
    parser.add_argument(
        "--max", type=int, default=None, help="Override max_tickers (for quick runs)"
    )
    args = parser.parse_args()

    criteria = load_criteria(Path(args.config)) if args.config else load_criteria()
    if args.max is not None:
        criteria.setdefault("universe", {})["max_tickers"] = args.max

    result = run_weekly_movers(criteria=criteria)

    if args.output:
        rows = [{**r, "type": "gainer"} for r in result["gainers"]]
        rows += [{**r, "type": "loser"} for r in result["losers"]]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        _print_table(f"Top {len(result['gainers'])} Weekly Gainers", result["gainers"])
        _print_table(f"Top {len(result['losers'])} Weekly Losers", result["losers"])


if __name__ == "__main__":
    main()
