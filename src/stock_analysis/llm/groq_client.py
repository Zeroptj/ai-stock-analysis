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
# Groq free tier: 30 RPM AND 12K TPM on llama-3.3-70b. The TPM cap is what
# actually trips first when payloads are large, so we throttle conservatively
# on RPM to give TPM more breathing room — and back off harder for ~90s after
# any 429 we observe (suggests we're brushing the TPM ceiling).
_RPM_LIMIT = 18              # buffer well under 30 RPM
_MIN_GAP_SECONDS = 4.0        # baseline gap between two consecutive calls
_COOLDOWN_GAP_SECONDS = 8.0   # gap during the cooldown window after a 429
_COOLDOWN_DURATION = 90.0     # how long to apply the elevated gap
_REQUEST_LOG: list[float] = []  # monotonic timestamps of recent calls
_LAST_429_AT: float = 0.0       # monotonic timestamp of the most recent 429


def _note_rate_limit() -> None:
    """Record a 429 so subsequent calls space themselves out further."""
    global _LAST_429_AT
    _LAST_429_AT = time.monotonic()


def _current_min_gap() -> float:
    """Use the cooldown gap if a 429 happened in the last _COOLDOWN_DURATION."""
    if _LAST_429_AT and (time.monotonic() - _LAST_429_AT) < _COOLDOWN_DURATION:
        return _COOLDOWN_GAP_SECONDS
    return _MIN_GAP_SECONDS


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

    # Minimum gap between successive calls — elevated during 429 cooldown
    min_gap = _current_min_gap()
    if _REQUEST_LOG:
        gap = min_gap - (now - _REQUEST_LOG[-1])
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
            _note_rate_limit()  # spaces subsequent calls further apart for 90s
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
    thesis_rating: str | None = None,
    thesis_summary: str | None = None,
    fair_value_vs_current: str | None = None,
) -> dict[str, Any]:
    """Generate industry and competitive positioning commentary.

    Optional thesis_* args let the LLM align its peer narrative with the
    overall recommendation — without them, peer commentary often contradicts
    the SELL/BUY rating produced by the thesis pass.
    """
    template = _load_prompt("industry_commentary.txt")

    prompt = template.format(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        comparables=json.dumps(comparables, indent=2, default=str),
        thesis_rating=thesis_rating or "(not yet determined)",
        thesis_summary=thesis_summary or "(not yet determined)",
        fair_value_vs_current=fair_value_vs_current or "(not yet determined)",
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


def _extract_proxy_for_llm(text: str, total_cap: int = 25_000) -> str:
    """Build a proxy-statement payload for the LLM that prioritizes the
    Director Nominees section. DEF 14A documents are huge (Microsoft's runs
    into the hundreds of pages), and a flat truncation usually drops the
    nominee list entirely — yielding the "1 of 12 directors" bug."""
    if not text:
        return "(no proxy text available)"

    headings = (
        "director nominees",
        "nominees for director",
        "nominees for election as director",
        "election of directors",
        "biographical information",
        "our board of directors",
    )
    lower = text.lower()
    director_idx = -1
    for h in headings:
        idx = lower.find(h)
        if idx != -1:
            director_idx = idx
            break

    if director_idx == -1:
        return text[:total_cap]

    # Carve out a generous window around the directors section so all nominees
    # land in the same chunk. Keep the start of the doc too (cover, agenda).
    director_window = text[max(0, director_idx - 500): director_idx + 15_000]
    head = text[:8_000]
    combined = (
        head
        + "\n\n[…document truncated — DIRECTOR NOMINEES SECTION FOLLOWS…]\n\n"
        + director_window
    )
    return combined[:total_cap]


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
        proxy_text=_extract_proxy_for_llm(latest_proxy.get("text") or ""),
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
