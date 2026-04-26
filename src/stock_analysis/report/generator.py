"""PDF report generator — Jinja2 HTML → Playwright Chromium → PDF.

Produces a finance-standard research note with industry-routed valuation
(DCF / DDM / RI / AFFO / Cyclical), historical financials, peer comps,
SEC meeting/events synthesis, risks, and an appendix with DCF assumptions.
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

from stock_analysis.config import settings
from stock_analysis.report.charts import generate_all_charts

logger = structlog.get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _fmt_currency(value: Any, precision: int = 2) -> str:
    try:
        return f"${float(value):,.{precision}f}"
    except Exception:
        return "n/a"


def _fmt_bn(value: Any, precision: int = 1) -> str:
    try:
        v = float(value)
        if abs(v) >= 1e9:
            return f"${v/1e9:,.{precision}f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:,.{precision}f}M"
        return f"${v:,.0f}"
    except Exception:
        return "n/a"


def _fmt_pct(value: Any, precision: int = 1) -> str:
    try:
        return f"{float(value)*100:+.{precision}f}%"
    except Exception:
        return "n/a"


_BIG_NUM_RE = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+|\d{7,})(\.\d+)?\b")


def _humanize_nums(text: str | None) -> str:
    """Convert raw dollar amounts like $12,345,678,900 → $12.3B in LLM-generated text."""
    if not text:
        return ""

    def _sub(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw + (m.group(2) or ""))
        except ValueError:
            return m.group(0)
        if val >= 1e12:
            return f"${val/1e12:.1f}T"
        if val >= 1e9:
            return f"${val/1e9:.1f}B"
        if val >= 1e6:
            return f"${val/1e6:.1f}M"
        return m.group(0)

    return _BIG_NUM_RE.sub(_sub, text)


def _fmt_date(s: str | None) -> str:
    if not s:
        return ""
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%b %d, %Y")
    except (ValueError, TypeError):
        return s


def build_context(analysis: dict[str, Any]) -> dict[str, Any]:
    """Flatten the analyze_ticker result into a Jinja-friendly context."""
    bundle = analysis.get("valuation", {}) or {}
    primary = bundle.get("valuation", {}) or {}
    comps = bundle.get("comparables", {}) or {}
    history = bundle.get("financials_history", []) or []
    quarterly = bundle.get("quarterly_financials", {}) or {}
    ratios = bundle.get("ratios", {}) or {}
    sec_narrative = bundle.get("sec_narrative", {}) or {}
    ownership = bundle.get("ownership", {}) or {}
    earnings_calendar = bundle.get("earnings_calendar", {}) or {}

    thesis = analysis.get("thesis", {}) or {}
    risks = analysis.get("risks", {}) or {}
    industry = analysis.get("industry", {}) or {}
    meetings = analysis.get("meetings", {}) or {}
    annual_summary = analysis.get("annual_summary") or {}
    qc_check = analysis.get("qc_check", {}) or {}
    screening = analysis.get("screening", {}) or {}

    target_mult = comps.get("target_multiples", {}) or {}
    val_ratios = (ratios.get("valuation") if isinstance(ratios, dict) else None) or {}
    charts = generate_all_charts(analysis)

    # Single source of truth for headline multiples — calc_engine ratios first
    # (computed from SEC EPS / EBITDA), yfinance .info as fallback. Without
    # this, page 1 (yfinance) and page 5 (ratios) showed different P/E values.
    pe_unified = val_ratios.get("pe_ratio") or target_mult.get("pe_ratio")
    ev_ebitda_unified = val_ratios.get("ev_to_ebitda") or target_mult.get("ev_to_ebitda")
    ps_unified = val_ratios.get("price_to_sales") or target_mult.get("price_to_sales")

    # Market cap: shares × price (SEC shares + current price)
    shares = primary.get("assumptions", {}).get("shares_outstanding", 0) or 0
    current_price = bundle.get("current_price", 0) or analysis.get("current_price", 0) or 0
    market_cap = shares * current_price if shares and current_price else 0

    return {
        "ticker": bundle.get("ticker", analysis.get("ticker", "")),
        "company_name": bundle.get("company_name") or bundle.get("ticker", ""),
        "sector": bundle.get("sector", ""),
        "sub_industry": bundle.get("sub_industry", ""),
        "report_date": datetime.now().strftime("%B %d, %Y"),
        "current_price": current_price,

        # Valuation summary
        "model_used": bundle.get("model_used", ""),
        "model_description": bundle.get("model_description", ""),
        "fair_value": primary.get("fair_value", 0),
        "upside_pct": primary.get("upside_pct", 0),
        "primary": primary,                # model-specific payload
        "assumptions": primary.get("assumptions", {}),

        # Key metrics
        "market_cap": market_cap,
        "enterprise_value": primary.get("enterprise_value"),
        "wacc": primary.get("wacc"),
        "cost_of_equity": primary.get("cost_of_equity"),
        "ev_ebitda": ev_ebitda_unified,
        "pe_ratio": pe_unified,
        "ps_ratio": ps_unified,

        # Comparables
        "comps": comps,
        "peers": (comps.get("comp_summary", {}) or {}).get("peers", []),
        "implied_values": comps.get("implied_values", {}) or {},

        # Historical financials
        "financials_history": history,
        "quarterly_financials": quarterly,
        "ratios": ratios,

        # Narrative
        "thesis": thesis,
        "risks": risks,
        "industry_commentary": industry,
        "meetings": meetings,
        "annual_summary": annual_summary,
        "qc_check": qc_check,

        # SEC filings context (metadata only on the cover; text is inside `meetings`)
        "annual_filings": sec_narrative.get("annual_filings", []),
        "proxy_statements": sec_narrative.get("proxy_statements", []),
        "material_events": sec_narrative.get("material_events", []),

        # Ownership & catalysts
        "ownership": ownership,
        "insider_transactions": ownership.get("insider_transactions", []),
        "institutional_holders": ownership.get("institutional_holders", []),
        "major_holders": ownership.get("major_holders", []),
        "earnings_calendar": earnings_calendar,

        # Screening context
        "screening": screening,

        # Charts (base64 data URIs)
        "charts": charts,
    }


def render_html(analysis: dict[str, Any]) -> str:
    """Render analysis into HTML using the report template."""
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATES_DIR), str(STATIC_DIR)]),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["fmt_currency"] = _fmt_currency
    env.filters["fmt_bn"] = _fmt_bn
    env.filters["fmt_pct"] = _fmt_pct
    env.filters["fmt_date"] = _fmt_date
    env.filters["humanize_nums"] = _humanize_nums

    template = env.get_template("report.html")
    return template.render(**build_context(analysis))


def generate_pdf(
    analysis: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Generate PDF via headless Chromium (Playwright)."""
    ticker = analysis.get("ticker", "UNKNOWN")
    logger.info("generating_pdf", ticker=ticker)

    html_content = render_html(analysis)

    if output_path is None:
        output_dir = Path(settings.report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = output_dir / f"{ticker}_report_{date_str}.pdf"
    else:
        output_path = Path(output_path)

    # Write HTML to a temp file so Chromium can resolve relative asset URLs
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        temp_html_path = Path(f.name)
        f.write(html_content)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"file:///{temp_html_path.as_posix()}")
            page.emulate_media(media="print")
            page.pdf(
                path=str(output_path),
                format="A4",
                print_background=True,
                margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                display_header_footer=True,
                header_template=(
                    '<div style="font-size:8px; width:100%; padding:0 12mm; '
                    'color:#6B7280; display:flex; justify-content:space-between;">'
                    f'<span>{ticker} — Equity Research</span>'
                    f'<span>{datetime.now().strftime("%b %d, %Y")}</span>'
                    '</div>'
                ),
                footer_template=(
                    '<div style="font-size:8px; width:100%; padding:0 12mm; '
                    'color:#6B7280; display:flex; justify-content:space-between;">'
                    '<span>Generated by AI Stock Analysis</span>'
                    '<span>Page <span class="pageNumber"></span>'
                    ' of <span class="totalPages"></span></span></div>'
                ),
            )
            browser.close()
    finally:
        try:
            temp_html_path.unlink(missing_ok=True)
        except Exception:
            pass

    logger.info("pdf_generated", path=str(output_path))
    return output_path
