"""Groq API client — LLM integration for narrative generation.

Handles retries, rate limiting, and structured output parsing.
LLM is used ONLY for writing and interpretation, NOT calculations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog
from groq import Groq, RateLimitError

from stock_analysis.config import settings

logger = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# --- Rate limiter ---
# Groq free tier: ~30 RPM. We run 6 LLM calls / ticker now, so when multiple
# tickers run back-to-back we need a sliding-window throttle to avoid 429s.
_RPM_LIMIT = 25              # keep buffer under Groq free-tier 30 RPM
_MIN_GAP_SECONDS = 2.0        # minimum gap between two consecutive calls
_REQUEST_LOG: list[float] = []  # monotonic timestamps of recent calls


def _throttle() -> None:
    """Sliding-window throttle before each LLM call."""
    global _REQUEST_LOG
    now = time.monotonic()
    _REQUEST_LOG = [t for t in _REQUEST_LOG if now - t < 60]

    # Sleep until the 60s window has room
    if len(_REQUEST_LOG) >= _RPM_LIMIT:
        wait = 60 - (now - _REQUEST_LOG[0]) + 0.5
        logger.info("llm_throttle_wait", wait_s=round(wait, 1), queue=len(_REQUEST_LOG))
        time.sleep(max(0.0, wait))
        now = time.monotonic()
        _REQUEST_LOG = [t for t in _REQUEST_LOG if now - t < 60]

    # Minimum gap between successive calls
    if _REQUEST_LOG:
        gap = _MIN_GAP_SECONDS - (now - _REQUEST_LOG[-1])
        if gap > 0:
            time.sleep(gap)

    _REQUEST_LOG.append(time.monotonic())


def _get_client() -> Groq:
    """Get Groq client instance."""
    return Groq(api_key=settings.api_keys.groq_api_key)


def _load_prompt(template_name: str) -> str:
    """Load a prompt template from file."""
    path = PROMPTS_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float | None = None,
    response_format: str = "text",  # "text" or "json"
) -> str:
    """Call Groq LLM with retry logic.

    Args:
        prompt: User prompt
        system_prompt: System instructions
        model: Model to use (defaults to config)
        max_tokens: Max response tokens
        temperature: Sampling temperature
        response_format: "text" or "json"

    Returns:
        LLM response text
    """
    client = _get_client()
    model = model or settings.llm.model_main
    temperature = temperature if temperature is not None else settings.llm.temperature

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(settings.llm.max_retries):
        try:
            _throttle()
            logger.info(
                "llm_call",
                model=model,
                attempt=attempt + 1,
                max_tokens=max_tokens,
            )
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            logger.info("llm_response", tokens=response.usage.total_tokens if response.usage else 0)
            return content

        except RateLimitError as e:
            # Honor Retry-After when the SDK exposes it; fallback to exp-backoff.
            retry_after = None
            try:
                retry_after = float(getattr(e, "response", None).headers.get("Retry-After"))  # type: ignore[union-attr]
            except Exception:
                retry_after = None
            wait = retry_after if retry_after else min(2 ** attempt * 10, 90)
            logger.warning("rate_limited", wait=wait, attempt=attempt + 1)
            time.sleep(wait)

        except Exception as e:
            logger.error("llm_error", error=str(e), attempt=attempt + 1)
            if attempt == settings.llm.max_retries - 1:
                raise

    raise RuntimeError("LLM call failed after all retries")


def generate_thesis(
    ticker: str,
    company_name: str,
    financials: dict[str, Any],
    valuation: dict[str, Any],
    market_data: dict[str, Any],
) -> dict[str, Any]:
    """Generate investment thesis (bull/base/bear cases)."""
    template = _load_prompt("thesis_generator.txt")

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        financials=json.dumps(financials, indent=2, default=str),
        valuation=json.dumps(valuation, indent=2, default=str),
        market_data=json.dumps(market_data, indent=2, default=str),
    )

    system = (
        "You are a senior buy-side equity analyst. Generate investment thesis "
        "based ONLY on the data provided. Never invent numbers — cite or don't claim. "
        "Respond in JSON format."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        max_tokens=settings.llm.max_tokens_thesis,
        response_format="json",
    )

    return json.loads(response)


def generate_risk_analysis(
    ticker: str,
    company_name: str,
    financials: dict[str, Any],
    sector: str,
) -> dict[str, Any]:
    """Generate risk factors summary."""
    template = _load_prompt("risk_synthesizer.txt")

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        financials=json.dumps(financials, indent=2, default=str),
        sector=sector,
    )

    system = (
        "You are a risk analyst. Identify key risks based ONLY on the provided data. "
        "Never speculate beyond the data. Respond in JSON format."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        max_tokens=settings.llm.max_tokens_risk,
        response_format="json",
    )

    return json.loads(response)


def generate_industry_commentary(
    ticker: str,
    company_name: str,
    sector: str,
    comparables: dict[str, Any],
) -> dict[str, Any]:
    """Generate industry and competitive positioning commentary."""
    template = _load_prompt("industry_commentary.txt")

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        comparables=json.dumps(comparables, indent=2, default=str),
    )

    system = (
        "You are an industry analyst. Provide competitive positioning analysis "
        "based ONLY on the data provided. Respond in JSON format."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        max_tokens=settings.llm.max_tokens_industry,
        response_format="json",
    )

    return json.loads(response)


def generate_annual_report_summary(
    ticker: str,
    company_name: str,
    annual_filings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize latest 10-K MD&A / risk factors / segment narrative into a structured JSON."""
    template = _load_prompt("annual_report_summary.txt")

    latest = annual_filings[0] if annual_filings else {}
    text = (latest.get("text") or "").strip()
    if not text:
        return {
            "business_overview": None,
            "segment_highlights": [],
            "mdna_themes": [],
            "key_risk_factors": [],
            "strategic_initiatives": [],
            "investor_takeaways": [],
        }

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        filing_date=latest.get("filing_date", "n/a"),
        filing_url=latest.get("document_url", "n/a"),
        filing_text=text[:20_000],  # hard cap for Groq TPM
    )

    system = (
        "You extract facts from SEC 10-K filings. Never invent numbers or names. "
        "Return JSON only."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        max_tokens=settings.llm.max_tokens_annual,
        response_format="json",
    )

    return json.loads(response)


def generate_meeting_synthesis(
    ticker: str,
    company_name: str,
    proxy_statements: list[dict[str, Any]],
    material_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize DEF 14A + 8-K filings into a structured section for the report."""
    template = _load_prompt("meeting_synthesizer.txt")

    latest_proxy = proxy_statements[0] if proxy_statements else {}
    latest_event = material_events[0] if material_events else {}
    other_events = [
        {"date": e.get("filing_date"), "accession": e.get("accession_number")}
        for e in material_events[1:]
    ]

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        proxy_date=latest_proxy.get("filing_date", "n/a"),
        proxy_url=latest_proxy.get("document_url", "n/a"),
        proxy_text=(latest_proxy.get("text") or "(no proxy text available)")[:15_000],
        event_date=latest_event.get("filing_date", "n/a"),
        event_url=latest_event.get("document_url", "n/a"),
        event_text=(latest_event.get("text") or "(no 8-K text available)")[:5_000],
        other_events=json.dumps(other_events, default=str),
    )

    system = (
        "You summarize SEC filings factually. Never invent names, dates, or numbers. "
        "Return JSON only."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        max_tokens=settings.llm.max_tokens_meetings,
        response_format="json",
    )

    return json.loads(response)


def run_qc_check(
    ticker: str,
    valuation_result: dict[str, Any],
    market_data: dict[str, Any],
) -> dict[str, Any]:
    """QC / sanity check on valuation numbers — flag anomalies."""
    template = _load_prompt("qc_checker.txt")

    prompt = template.format(
        ticker=ticker,
        valuation=json.dumps(valuation_result, indent=2, default=str),
        market_data=json.dumps(market_data, indent=2, default=str),
    )

    system = (
        "You are a quantitative analyst performing sanity checks. "
        "Flag any numbers that seem unreasonable. Respond in JSON format."
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=system,
        model=settings.llm.model_qc,
        max_tokens=settings.llm.max_tokens_qc,
        response_format="json",
    )

    return json.loads(response)
