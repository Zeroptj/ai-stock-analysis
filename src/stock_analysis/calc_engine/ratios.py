"""Financial ratios calculator — margin, ROIC, leverage, efficiency, etc.

All calculations are pure Python — no LLM involved.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Safe division that returns None instead of raising."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def calc_profitability_ratios(financials: dict[str, Any]) -> dict[str, float | None]:
    """Calculate profitability ratios from financial data."""
    revenue = financials.get("revenue")
    gross_profit = financials.get("gross_profit")
    operating_income = financials.get("operating_income")
    net_income = financials.get("net_income")
    cost_of_revenue = financials.get("cost_of_revenue")

    gross_margin = safe_div(gross_profit, revenue)
    operating_margin = safe_div(operating_income, revenue)
    net_margin = safe_div(net_income, revenue)

    return {
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "cost_ratio": safe_div(cost_of_revenue, revenue),
    }


def calc_return_ratios(financials: dict[str, Any]) -> dict[str, float | None]:
    """Calculate return ratios (ROE, ROA, ROIC)."""
    net_income = financials.get("net_income")
    total_assets = financials.get("total_assets")
    total_equity = financials.get("stockholders_equity")
    total_debt = financials.get("total_debt", 0) or 0
    cash = financials.get("cash", 0) or 0
    interest_expense = financials.get("interest_expense", 0) or 0
    tax_rate = financials.get("effective_tax_rate", 0.21)

    roe = safe_div(net_income, total_equity)
    roa = safe_div(net_income, total_assets)

    # ROIC = NOPAT / Invested Capital
    operating_income = financials.get("operating_income")
    if operating_income is not None:
        nopat = operating_income * (1 - tax_rate)
        invested_capital = (total_equity or 0) + total_debt - cash
        roic = safe_div(nopat, invested_capital) if invested_capital > 0 else None
    else:
        roic = None

    return {
        "roe": roe,
        "roa": roa,
        "roic": roic,
    }


def calc_leverage_ratios(financials: dict[str, Any]) -> dict[str, float | None]:
    """Calculate leverage and solvency ratios."""
    total_assets = financials.get("total_assets")
    total_liabilities = financials.get("total_liabilities")
    total_equity = financials.get("stockholders_equity")
    total_debt = financials.get("total_debt")
    ebitda = financials.get("ebitda")
    interest_expense = financials.get("interest_expense")
    operating_income = financials.get("operating_income")

    return {
        "debt_to_equity": safe_div(total_debt, total_equity),
        "debt_to_assets": safe_div(total_debt, total_assets),
        "debt_to_ebitda": safe_div(total_debt, ebitda),
        "equity_multiplier": safe_div(total_assets, total_equity),
        "interest_coverage": safe_div(operating_income, interest_expense),
    }


def calc_efficiency_ratios(financials: dict[str, Any]) -> dict[str, float | None]:
    """Calculate efficiency/activity ratios."""
    revenue = financials.get("revenue")
    total_assets = financials.get("total_assets")
    receivables = financials.get("receivables")
    inventory = financials.get("inventory")
    cost_of_revenue = financials.get("cost_of_revenue")

    return {
        "asset_turnover": safe_div(revenue, total_assets),
        "receivables_turnover": safe_div(revenue, receivables),
        "days_sales_outstanding": safe_div(receivables, safe_div(revenue, 365)),
        "inventory_turnover": safe_div(cost_of_revenue, inventory),
    }


def calc_growth_rates(historical: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    """Calculate YoY growth rates for a given metric."""
    if len(historical) < 2:
        return {"yoy_growth": None, "cagr_3y": None, "cagr_5y": None}

    values = [h.get(metric) for h in historical if h.get(metric) is not None]
    if len(values) < 2:
        return {"yoy_growth": None, "cagr_3y": None, "cagr_5y": None}

    # YoY growth (most recent vs previous)
    yoy = safe_div(values[0] - values[1], abs(values[1]))

    # CAGR
    def _cagr(start_val: float, end_val: float, years: int) -> float | None:
        if start_val <= 0 or end_val <= 0 or years <= 0:
            return None
        return (end_val / start_val) ** (1 / years) - 1

    cagr_3y = _cagr(values[min(3, len(values) - 1)], values[0], min(3, len(values) - 1)) if len(values) >= 3 else None
    cagr_5y = _cagr(values[min(5, len(values) - 1)], values[0], min(5, len(values) - 1)) if len(values) >= 4 else None

    return {
        "yoy_growth": yoy,
        "cagr_3y": cagr_3y,
        "cagr_5y": cagr_5y,
    }


def calc_valuation_multiples(
    market_data: dict[str, Any], financials: dict[str, Any]
) -> dict[str, float | None]:
    """Calculate valuation multiples."""
    price = market_data.get("current_price")
    shares = market_data.get("shares_outstanding")
    market_cap = market_data.get("market_cap")
    total_debt = financials.get("total_debt", 0) or 0
    cash = financials.get("cash", 0) or 0

    ev = (market_cap or 0) + total_debt - cash

    revenue = financials.get("revenue")
    ebitda = financials.get("ebitda")
    net_income = financials.get("net_income")
    eps = financials.get("eps")
    book_value = financials.get("stockholders_equity")

    return {
        "market_cap": market_cap,
        "enterprise_value": ev if ev > 0 else None,
        "pe_ratio": safe_div(price, eps),
        "ev_to_revenue": safe_div(ev, revenue),
        "ev_to_ebitda": safe_div(ev, ebitda),
        "price_to_book": safe_div(market_cap, book_value),
        "price_to_sales": safe_div(market_cap, revenue),
        "earnings_yield": safe_div(eps, price),
        "fcf_yield": safe_div(
            financials.get("free_cash_flow"),
            market_cap
        ),
    }


def calculate_all_ratios(
    financials: dict[str, Any],
    market_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate all financial ratios for a company."""
    logger.info("calculating_ratios", ticker=financials.get("ticker", ""))

    result = {
        "profitability": calc_profitability_ratios(financials),
        "returns": calc_return_ratios(financials),
        "leverage": calc_leverage_ratios(financials),
        "efficiency": calc_efficiency_ratios(financials),
    }

    if market_data:
        result["valuation"] = calc_valuation_multiples(market_data, financials)

    return result
