"""Market data client — yfinance wrapper for price history and fundamentals."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
import yfinance as yf

from stock_analysis.data.cache import cached
from stock_analysis.database import Financial, PriceHistory, get_session

logger = structlog.get_logger(__name__)


def _fetch_yf_info(ticker: str) -> dict[str, Any]:
    """Raw yfinance .info fetch (intended to be wrapped in cache)."""
    return dict(yf.Ticker(ticker).info or {})


def get_stock_info(ticker: str) -> dict[str, Any]:
    """Fetch basic stock info and fundamentals from yfinance (cached 12h)."""
    logger.info("fetching_stock_info", ticker=ticker)
    info = cached(
        key=f"yf_info:{ticker.upper()}",
        ttl_hours=12,
        fetcher=lambda: _fetch_yf_info(ticker),
    ) or {}

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName", info.get("shortName", ticker)),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap", 0),
        "enterprise_value": info.get("enterpriseValue", 0),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "ev_to_revenue": info.get("enterpriseToRevenue"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "dividend_yield": info.get("dividendYield"),
        "payout_ratio": info.get("payoutRatio"),
        "beta": info.get("beta"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),
        "currency": info.get("currency", "USD"),
    }


def get_price_history(
    ticker: str, period: str = "2y", interval: str = "1d"
) -> list[dict[str, Any]]:
    """Fetch historical price data."""
    logger.info("fetching_price_history", ticker=ticker, period=period)
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)

    if hist.empty:
        logger.warning("no_price_history", ticker=ticker)
        return []

    records = []
    for date, row in hist.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        })

    # Cache to database
    session = get_session()
    try:
        for rec in records[-30:]:  # Cache last 30 days
            session.add(PriceHistory(
                ticker=ticker.upper(),
                date=rec["date"],
                open=rec["open"],
                high=rec["high"],
                low=rec["low"],
                close=rec["close"],
                volume=rec["volume"],
            ))
        session.commit()
    finally:
        session.close()

    return records


def get_financial_statements(ticker: str) -> dict[str, Any]:
    """Fetch income statement, balance sheet, and cash flow from yfinance."""
    logger.info("fetching_financial_statements", ticker=ticker)
    stock = yf.Ticker(ticker)

    def _df_to_dict(df) -> dict:
        if df is None or df.empty:
            return {}
        result = {}
        for col in df.columns:
            period_key = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            result[period_key] = {
                str(idx): (float(val) if val == val else None)  # NaN check
                for idx, val in df[col].items()
            }
        return result

    income_stmt = _df_to_dict(stock.income_stmt)
    balance_sheet = _df_to_dict(stock.balance_sheet)
    cash_flow = _df_to_dict(stock.cashflow)

    # Cache to database
    session = get_session()
    try:
        for stmt_type, data in [
            ("income", income_stmt),
            ("balance", balance_sheet),
            ("cashflow", cash_flow),
        ]:
            session.add(Financial(
                ticker=ticker.upper(),
                period="latest",
                statement_type=stmt_type,
                data_json=json.dumps(data, default=str),
                source="yfinance",
            ))
        session.commit()
    finally:
        session.close()

    return {
        "income_statement": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
    }


def _df_or_none_to_records(df) -> list[dict[str, Any]]:
    """Convert a yfinance DataFrame (or None) into a list of plain dicts."""
    if df is None:
        return []
    try:
        if df.empty:
            return []
        # Reset index so date / holder columns become regular columns
        df2 = df.reset_index()
        return json.loads(df2.to_json(orient="records", date_format="iso", default_handler=str))
    except Exception:
        return []


def get_ownership(ticker: str) -> dict[str, Any]:
    """Insider transactions + institutional / major holders (cached 24h)."""
    logger.info("fetching_ownership", ticker=ticker)

    def _fetch() -> dict[str, Any]:
        t = yf.Ticker(ticker)
        return {
            "insider_transactions": _df_or_none_to_records(getattr(t, "insider_transactions", None)),
            "insider_purchases": _df_or_none_to_records(getattr(t, "insider_purchases", None)),
            "institutional_holders": _df_or_none_to_records(getattr(t, "institutional_holders", None)),
            "major_holders": _df_or_none_to_records(getattr(t, "major_holders", None)),
        }

    return cached(
        key=f"yf_ownership:{ticker.upper()}",
        ttl_hours=24,
        fetcher=_fetch,
    ) or {
        "insider_transactions": [],
        "insider_purchases": [],
        "institutional_holders": [],
        "major_holders": [],
    }


def get_earnings_calendar(ticker: str) -> dict[str, Any]:
    """Upcoming earnings date + EPS/revenue estimates (cached 6h — near-term event)."""
    logger.info("fetching_earnings_calendar", ticker=ticker)

    def _fetch() -> dict[str, Any]:
        t = yf.Ticker(ticker)
        cal = getattr(t, "calendar", None)
        if cal is None:
            return {}
        # yfinance returns either DataFrame or dict depending on version
        if isinstance(cal, dict):
            return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in cal.items()}
        try:
            return {c: (cal[c].iloc[0].isoformat() if hasattr(cal[c].iloc[0], "isoformat") else cal[c].iloc[0])
                    for c in cal.columns}
        except Exception:
            return {}

    return cached(
        key=f"yf_calendar:{ticker.upper()}",
        ttl_hours=6,
        fetcher=_fetch,
    ) or {}


def fetch_market_data(ticker: str) -> dict[str, Any]:
    """Main entry point — fetch all market data for a ticker."""
    info = get_stock_info(ticker)
    prices = get_price_history(ticker)
    financials = get_financial_statements(ticker)
    ownership = get_ownership(ticker)
    earnings_calendar = get_earnings_calendar(ticker)

    return {
        "info": info,
        "price_history": prices,
        "financials": financials,
        "ownership": ownership,
        "earnings_calendar": earnings_calendar,
        "source": "yfinance",
        "fetched_at": datetime.utcnow().isoformat(),
    }
