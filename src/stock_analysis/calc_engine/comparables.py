"""Comparable company analysis — peer valuation multiples.

Compares target company against sector peers using EV/Revenue, EV/EBITDA, P/E.
"""

from __future__ import annotations

from typing import Any

import structlog
import yfinance as yf

from stock_analysis.calc_engine.ratios import safe_div

logger = structlog.get_logger(__name__)

# Default peer groups by sector (can be overridden)
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "ORCL"],
    "Information Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "ORCL"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW"],
    "Health Care": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "CL"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "Industrials": ["HON", "UPS", "CAT", "RTX", "DE", "LMT", "GE"],
    "Real Estate": ["PLD", "AMT", "CCI", "O", "SPG", "PSA", "EQIX"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL"],
}


def get_peer_multiples(ticker: str, sector: str | None = None) -> list[dict[str, Any]]:
    """Fetch valuation multiples for sector peers."""
    if sector and sector in SECTOR_PEERS:
        peers = [t for t in SECTOR_PEERS[sector] if t != ticker.upper()]
    else:
        # Try to find sector
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get("sector", "")
            peers = [t for t in SECTOR_PEERS.get(sector, []) if t != ticker.upper()]
        except Exception:
            peers = []

    if not peers:
        logger.warning("no_peers_found", ticker=ticker, sector=sector)
        return []

    logger.info("fetching_peer_multiples", ticker=ticker, peers=peers)
    peer_data = []

    for peer_ticker in peers[:8]:  # Limit to 8 peers
        try:
            stock = yf.Ticker(peer_ticker)
            info = stock.info

            market_cap = info.get("marketCap", 0)
            ev = info.get("enterpriseValue", 0)
            revenue = info.get("totalRevenue", 0)
            ebitda = info.get("ebitda", 0)
            net_income = info.get("netIncomeToCommon", 0)

            peer_data.append({
                "ticker": peer_ticker,
                "name": info.get("longName", peer_ticker),
                "market_cap": market_cap,
                "ev_to_revenue": safe_div(ev, revenue),
                "ev_to_ebitda": safe_div(ev, ebitda),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": safe_div(market_cap, revenue),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
            })
        except Exception as e:
            logger.warning("peer_fetch_error", peer=peer_ticker, error=str(e))

    return peer_data


def calc_comp_summary(
    target_multiples: dict[str, float | None],
    peer_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate summary statistics and implied valuation from comps."""
    import numpy as np

    metrics = ["ev_to_revenue", "ev_to_ebitda", "pe_ratio", "forward_pe", "price_to_book"]
    summary: dict[str, Any] = {"peers": peer_data}

    for metric in metrics:
        values = [p[metric] for p in peer_data if p.get(metric) is not None and p[metric] > 0]
        if values:
            summary[f"{metric}_median"] = round(float(np.median(values)), 2)
            summary[f"{metric}_mean"] = round(float(np.mean(values)), 2)
            summary[f"{metric}_min"] = round(min(values), 2)
            summary[f"{metric}_max"] = round(max(values), 2)
            summary[f"{metric}_target"] = target_multiples.get(metric)

    return summary


def calc_implied_values(
    comp_summary: dict[str, Any],
    financials: dict[str, Any],
    shares_outstanding: float,
) -> dict[str, float | None]:
    """Calculate implied share price from median peer multiples."""
    revenue = financials.get("revenue", 0)
    ebitda = financials.get("ebitda", 0)
    net_income = financials.get("net_income", 0)
    book_value = financials.get("stockholders_equity", 0)
    net_debt = (financials.get("total_debt", 0) or 0) - (financials.get("cash", 0) or 0)

    implied = {}

    # EV/Revenue implied
    median_ev_rev = comp_summary.get("ev_to_revenue_median")
    if median_ev_rev and revenue and shares_outstanding:
        ev = revenue * median_ev_rev
        equity = ev - net_debt
        implied["ev_revenue_implied"] = round(equity / shares_outstanding, 2)

    # EV/EBITDA implied
    median_ev_ebitda = comp_summary.get("ev_to_ebitda_median")
    if median_ev_ebitda and ebitda and shares_outstanding:
        ev = ebitda * median_ev_ebitda
        equity = ev - net_debt
        implied["ev_ebitda_implied"] = round(equity / shares_outstanding, 2)

    # P/E implied
    median_pe = comp_summary.get("pe_ratio_median")
    if median_pe and net_income and shares_outstanding:
        eps = net_income / shares_outstanding
        implied["pe_implied"] = round(eps * median_pe, 2)

    # P/B implied
    median_pb = comp_summary.get("price_to_book_median")
    if median_pb and book_value and shares_outstanding:
        bvps = book_value / shares_outstanding
        implied["pb_implied"] = round(bvps * median_pb, 2)

    return implied


def run_comparables(
    ticker: str,
    financials: dict[str, Any],
    market_data: dict[str, Any],
    sector: str | None = None,
) -> dict[str, Any]:
    """Run full comparable company analysis."""
    logger.info("running_comparables", ticker=ticker)

    info = market_data.get("info", {})
    sector = sector or info.get("sector", "")
    shares = info.get("shares_outstanding", 0) or 0

    target_multiples = {
        "ev_to_revenue": info.get("ev_to_revenue"),
        "ev_to_ebitda": info.get("ev_to_ebitda"),
        "pe_ratio": info.get("pe_ratio"),
        "forward_pe": info.get("forward_pe"),
        "price_to_book": info.get("price_to_book"),
    }

    peer_data = get_peer_multiples(ticker, sector)
    comp_summary = calc_comp_summary(target_multiples, peer_data)
    implied = calc_implied_values(comp_summary, financials, shares)

    return {
        "ticker": ticker,
        "sector": sector,
        "target_multiples": target_multiples,
        "comp_summary": comp_summary,
        "implied_values": implied,
    }
