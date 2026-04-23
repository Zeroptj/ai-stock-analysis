"""SEC EDGAR API client — fetch 10-K, 10-Q, financial facts.

Uses the SEC EDGAR XBRL API (no API key needed, just User-Agent header).
Docs: https://efts.sec.gov/LATEST/search-index?q=
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import structlog

from stock_analysis.config import settings
from stock_analysis.database import Filing, Financial, get_session

logger = structlog.get_logger(__name__)

SEC_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_DATA_URL = "https://data.sec.gov"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_SUBMISSIONS_URL = f"{SEC_DATA_URL}/submissions"
SEC_COMPANY_FACTS_URL = f"{SEC_DATA_URL}/api/xbrl/companyfacts"
SEC_COMPANY_CONCEPT_URL = f"{SEC_DATA_URL}/api/xbrl/companyconcept"

# Max characters kept when extracting filing text for LLM consumption.
MAX_FILING_TEXT_CHARS = 50_000

# Mapping of common ticker → CIK (can be expanded or fetched dynamically)
_CIK_CACHE: dict[str, str] = {}


def _headers() -> dict[str, str]:
    return {
        "User-Agent": settings.api_keys.sec_user_agent,
        "Accept": "application/json",
    }


async def get_cik(ticker: str, client: httpx.AsyncClient) -> str:
    """Lookup CIK — accepts yfinance style (BRK-B), SEC style (BRKB), or dot (BRK.B)."""
    t_upper = ticker.upper()
    if t_upper in _CIK_CACHE:
        return _CIK_CACHE[t_upper]

    # Match variants: as-is, dash→dot, strip separators entirely
    variants = {
        t_upper,
        t_upper.replace("-", "."),
        t_upper.replace(".", "-"),
        t_upper.replace("-", "").replace(".", ""),
    }

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    data = resp.json()

    for entry in data.values():
        if entry["ticker"].upper() in variants:
            cik = str(entry["cik_str"]).zfill(10)
            _CIK_CACHE[t_upper] = cik
            logger.info("resolved_cik", ticker=ticker, matched=entry["ticker"], cik=cik)
            return cik

    raise ValueError(f"CIK not found for ticker: {ticker} (tried {sorted(variants)})")


async def get_company_facts(
    ticker: str,
    client: httpx.AsyncClient,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Fetch all XBRL facts for a company (revenue, net income, assets, etc.).

    Set use_cache=False to bypass the SQLite cache entirely (always hits SEC EDGAR).
    """
    import json as _json
    from datetime import timedelta

    key = f"sec_facts:{ticker.upper()}"

    if use_cache:
        try:
            from stock_analysis.database import CacheEntry, get_session, init_db
            init_db()
            session = get_session()
            try:
                entry = session.query(CacheEntry).filter_by(key=key).first()
                if entry and (datetime.utcnow() - entry.fetched_at) < timedelta(hours=24):
                    logger.debug("cache_hit", key=key)
                    return _json.loads(entry.value_json)
            finally:
                session.close()
        except Exception as exc:
            logger.warning("cache_unavailable", key=key, error=str(exc))
            use_cache = False

    cik = await get_cik(ticker, client)
    url = f"{SEC_COMPANY_FACTS_URL}/CIK{cik}.json"
    logger.info("fetching_company_facts", ticker=ticker, url=url)
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    data = resp.json()

    if use_cache:
        try:
            from stock_analysis.database import CacheEntry, get_session
            session = get_session()
            try:
                entry = session.query(CacheEntry).filter_by(key=key).first()
                payload = _json.dumps(data, default=str)
                if entry:
                    entry.value_json = payload
                    entry.fetched_at = datetime.utcnow()
                else:
                    session.add(CacheEntry(key=key, value_json=payload, fetched_at=datetime.utcnow()))
                session.commit()
            finally:
                session.close()
        except Exception as exc:
            logger.warning("cache_write_failed", key=key, error=str(exc))

    return data


async def get_submissions(ticker: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch recent filings list (10-K, 10-Q, 8-K, etc.)."""
    cik = await get_cik(ticker, client)
    url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"

    logger.info("fetching_submissions", ticker=ticker, url=url)
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


async def get_filing_list(
    ticker: str,
    client: httpx.AsyncClient,
    filing_type: str | list[str] = "10-K",
    count: int = 5,
) -> list[dict[str, Any]]:
    """Get list of recent filings of a given type (or list of types)."""
    submissions = await get_submissions(ticker, client)
    cik = await get_cik(ticker, client)
    recent = submissions.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    wanted = {filing_type} if isinstance(filing_type, str) else set(filing_type)
    filings = []
    for i, form in enumerate(forms):
        if form in wanted and len(filings) < count:
            acc = accessions[i]
            primary = primary_docs[i] if i < len(primary_docs) else None
            filings.append({
                "form": form,
                "filing_date": dates[i],
                "accession_number": acc,
                "primary_document": primary,
                "document_url": _build_document_url(cik, acc, primary) if primary else None,
            })

    logger.info(
        "found_filings",
        ticker=ticker,
        types=list(wanted),
        count=len(filings),
    )
    return filings


def _build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Construct the URL to a filing's primary document."""
    cik_int = str(int(cik))  # strip leading zeros for archive path
    acc_nodash = accession_number.replace("-", "")
    return f"{SEC_ARCHIVE_URL}/{cik_int}/{acc_nodash}/{primary_document}"


def _html_to_text(html: str, max_chars: int = MAX_FILING_TEXT_CHARS) -> str:
    """Strip HTML tags, collapse whitespace, truncate. Minimal deps."""
    import re

    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                        ("&gt;", ">"), ("&#160;", " "), ("&#8217;", "'"),
                        ("&#8211;", "-"), ("&#8212;", "-")):
        text = text.replace(entity, char)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


async def fetch_filing_text(
    filing: dict[str, Any],
    client: httpx.AsyncClient,
    max_chars: int = MAX_FILING_TEXT_CHARS,
) -> str:
    """Download and extract plain text from a filing's primary document."""
    url = filing.get("document_url")
    if not url:
        return ""
    try:
        resp = await client.get(url, headers=_headers(), timeout=60.0)
        resp.raise_for_status()
        return _html_to_text(resp.text, max_chars=max_chars)
    except Exception as e:
        logger.warning(
            "fetch_filing_text_error",
            accession=filing.get("accession_number"),
            error=str(e),
        )
        return ""


def _extract_financial_facts(facts: dict[str, Any], frequency: str = "annual") -> dict[str, Any]:
    """Extract key financial metrics from XBRL company facts.

    frequency: "annual" (10-K / FY) or "quarterly" (10-Q / Q1-Q3 + FY Q4).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    def _units_for(concept: str) -> list[dict]:
        concept_data = us_gaap.get(concept, {})
        units = concept_data.get("units", {})
        # XBRL uses different units: USD for dollars, USD/shares for per-share
        # metrics, shares for share counts, pure for ratios/percentages.
        return (
            units.get("USD")
            or units.get("USD/shares")
            or units.get("shares")
            or units.get("pure")
            or []
        )

    def _get_annual_values(concept: str, max_years: int = 10) -> list[dict]:
        data = _units_for(concept)
        annual = [
            item for item in data
            if item.get("form") == "10-K" and item.get("fp") == "FY"
        ]
        annual.sort(key=lambda x: x.get("end", ""), reverse=True)
        return annual[:max_years]

    def _get_quarterly_values(concept: str, max_quarters: int = 12) -> list[dict]:
        """Quarters from 10-Q (Q1-Q3) + 10-K Q4 implied. Dedupe by end date."""
        data = _units_for(concept)
        quarters = [
            item for item in data
            if (
                (item.get("form") == "10-Q" and item.get("fp") in ("Q1", "Q2", "Q3"))
                or (item.get("form") == "10-K" and item.get("fp") == "FY")
            )
        ]
        # Keep only items with a "start" field (duration / quarterly), not
        # instant snapshots — unless this concept is balance-sheet (instant).
        has_start = [q for q in quarters if q.get("start")]
        if has_start:
            quarters = has_start
        quarters.sort(key=lambda x: x.get("end", ""), reverse=True)
        return quarters[:max_quarters]

    key_concepts = {
        # --- Income statement ---
        "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
        "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
        "gross_profit": ["GrossProfit"],
        "rd_expense": ["ResearchAndDevelopmentExpense"],
        "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
        "operating_income": ["OperatingIncomeLoss"],
        "ebitda": ["EarningsBeforeInterestTaxesDepreciationAndAmortization"],
        "interest_expense": ["InterestExpense"],
        "interest_income": ["InvestmentIncomeInterest", "InterestIncomeOperating"],
        "income_tax_expense": ["IncomeTaxExpenseBenefit"],
        "effective_tax_rate": ["EffectiveIncomeTaxRateContinuingOperations"],
        "net_income": ["NetIncomeLoss", "ProfitLoss"],
        "eps_basic": ["EarningsPerShareBasic"],
        "eps_diluted": ["EarningsPerShareDiluted"],
        "eps": ["EarningsPerShareBasic", "EarningsPerShareDiluted"],

        # --- Balance sheet ---
        "total_assets": ["Assets"],
        "current_assets": ["AssetsCurrent"],
        "cash": ["CashAndCashEquivalentsAtCarryingValue"],
        "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
        "accounts_receivable": ["AccountsReceivableNetCurrent"],
        "inventory": ["InventoryNet"],
        "goodwill": ["Goodwill"],
        "intangibles": ["IntangibleAssetsNetExcludingGoodwill",
                        "FiniteLivedIntangibleAssetsNet"],
        "ppe_net": ["PropertyPlantAndEquipmentNet"],
        "total_liabilities": ["Liabilities"],
        "current_liabilities": ["LiabilitiesCurrent"],
        "accounts_payable": ["AccountsPayableCurrent"],
        "deferred_revenue": ["DeferredRevenue", "ContractWithCustomerLiabilityCurrent",
                             "ContractWithCustomerLiability"],
        "short_term_debt": ["LongTermDebtCurrent", "ShortTermBorrowings",
                            "DebtCurrent"],
        "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
        "stockholders_equity": ["StockholdersEquity",
                                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
        "shares_outstanding": ["CommonStockSharesOutstanding",
                               "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],

        # --- Cash flow ---
        "operating_cash_flow": ["NetCashProvidedByOperatingActivities"],
        "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
        "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "depreciation": ["DepreciationDepletionAndAmortization", "Depreciation"],
        "stock_based_comp": ["ShareBasedCompensation"],
        "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
        "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
        "dividends_per_share": ["CommonStockDividendsPerShareDeclared"],
    }

    result: dict[str, list[dict]] = {}
    max_periods = 12 if frequency == "quarterly" else 10
    fetch = _get_quarterly_values if frequency == "quarterly" else _get_annual_values

    for metric_name, concept_names in key_concepts.items():
        # Union across all tag variants (companies switch tags over time),
        # dedupe by period_end, newest period first.
        merged: dict[str, float] = {}
        for concept in concept_names:
            for v in fetch(concept, max_periods):
                end = v.get("end")
                if end and end not in merged:
                    merged[end] = v.get("val")
        if merged:
            rows = [{"period_end": e, "value": v} for e, v in merged.items()]
            rows.sort(key=lambda r: r["period_end"], reverse=True)
            result[metric_name] = rows[:max_periods]

    return result


async def fetch_sec_data(
    ticker: str,
    client: httpx.AsyncClient,
    include_filing_text: bool = True,
) -> dict[str, Any]:
    """Main entry point — fetch SEC data for a ticker.

    Returns financials + recent filings of 10-K, 10-Q, DEF 14A (proxy /
    shareholder meeting), and 8-K (material events). When `include_filing_text`
    is True, the latest DEF 14A and latest 8-K have their primary documents
    downloaded and converted to plain text for LLM consumption.
    """
    logger.info("fetching_sec_data", ticker=ticker)

    facts = await get_company_facts(ticker, client)
    annual = await get_filing_list(ticker, client, "10-K", count=3)
    quarterly = await get_filing_list(ticker, client, "10-Q", count=4)
    proxy = await get_filing_list(ticker, client, "DEF 14A", count=2)
    events = await get_filing_list(ticker, client, "8-K", count=5)

    financials = _extract_financial_facts(facts, frequency="annual")
    quarterly_financials = _extract_financial_facts(facts, frequency="quarterly")
    company_name = facts.get("entityName", ticker)

    # Fetch primary document text for the most useful narrative filings
    if include_filing_text:
        if annual:
            # 10-K is huge — allow a bigger char budget for MD&A / risk factors
            annual[0]["text"] = await fetch_filing_text(annual[0], client, max_chars=80_000)
        if proxy:
            proxy[0]["text"] = await fetch_filing_text(proxy[0], client)
        if events:
            events[0]["text"] = await fetch_filing_text(events[0], client)

    # Cache to database
    session = get_session()
    try:
        for filing in annual + quarterly + proxy + events:
            existing = session.query(Filing).filter_by(
                accession_number=filing["accession_number"]
            ).first()
            if not existing:
                session.add(Filing(
                    ticker=ticker.upper(),
                    filing_type=filing["form"],
                    filing_date=filing["filing_date"],
                    accession_number=filing["accession_number"],
                    content_json=json.dumps(filing),
                ))

        session.add(Financial(
            ticker=ticker.upper(),
            period="latest",
            statement_type="all",
            data_json=json.dumps(financials, default=str),
            source="sec",
        ))
        session.commit()
    finally:
        session.close()

    return {
        "company_name": company_name,
        "ticker": ticker.upper(),
        "annual_filings": annual,
        "quarterly_filings": quarterly,
        "proxy_statements": proxy,      # DEF 14A — shareholder meeting materials
        "material_events": events,      # 8-K — earnings, M&A, etc.
        "financials": financials,
        "quarterly_financials": quarterly_financials,
        "source": "SEC EDGAR",
        "fetched_at": datetime.utcnow().isoformat(),
    }
