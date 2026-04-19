# Building This Project From Scratch — AI Stock Analysis

> This guide walks someone who has never built a project like this → through to the current state (weekly US movers screener + 5 valuation models + LLM narrative + PDF report).
>
> **This is not a copy-paste script** — it's a layer-by-layer walkthrough that explains *why* each piece exists and *what problem* it solves for the next layer.

---

## 📑 Table of Contents

1. [Prerequisites — what to know first](#0-prerequisites)
2. [Project skeleton + venv + pyproject](#1-project-skeleton)
3. [Config + logging + settings](#2-config--logging)
4. [Database layer + TTL cache](#3-database--cache)
5. [Data ingestion — SEC / yfinance / FRED](#4-data-ingestion)
6. [Screening — NASDAQ universe + weekly movers](#5-screening)
7. [Calc engine — 5 valuation models + industry router](#6-calc-engine)
8. [LLM layer — Groq + prompts + rate limit](#7-llm-layer)
9. [Report generation — Jinja2 + charts + Playwright PDF](#8-report-generation)
10. [Orchestration — wire the pipeline together](#9-orchestration)
11. [MCP server (optional sidecar)](#10-mcp-server)
12. [Common pitfalls + debugging](#11-pitfalls)
13. [Next steps](#12-next)

---

## 0) Prerequisites

### What you should already know

**Python skills:**
- Python 3.11+ syntax
- `async` / `await` / `asyncio` — **critical**; the whole project is async because it's network-bound
- Type hints (`list[dict]`, `str | None`, `@dataclass`)
- Context managers (`with`, `async with`)
- Package structure (`src/` layout + `__init__.py`)

**Finance concepts (basic):**
- What DCF (Discounted Cash Flow) is — at minimum understand FCFF, terminal value, WACC
- Multiples: P/E, EV/EBITDA, P/B, P/S
- Balance sheet vs Income statement vs Cash flow statement
- Dividend Discount Model, Residual Income (for banks), AFFO (for REITs)

> If you're not comfortable here yet, read the first few chapters of *Damodaran on Valuation* before continuing.

**APIs / Web:**
- HTTP status codes (200, 429, 413, 5xx)
- JSON structure
- Rate limits + retry strategies
- REST vs streaming

**Tools:**
- Git
- CLI (bash / PowerShell)
- SQL (basic SELECT / INSERT)
- Markdown

---

## 1) Project skeleton

### Set up venv

```bash
mkdir ai-stock-analysis && cd ai-stock-analysis
python -m venv .venv
# Windows
source .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
```

### `pyproject.toml` (using hatchling)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "stock-analysis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "yfinance>=0.2.40",
    "pandas",
    "numpy",
    "pandas-ta",
    "fredapi",
    "SQLAlchemy>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv",
    "pyyaml",
    "structlog",
    "groq",
    "jinja2",
    "playwright>=1.40",
    "matplotlib",
    "seaborn",
    "asyncio-throttle",
    "mcp>=1.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]

[project.scripts]
sec-mcp-server = "stock_analysis.mcp_servers.sec_edgar_server:main"

[tool.hatch.build.targets.wheel]
packages = ["src/stock_analysis"]
```

```bash
pip install -e ".[dev]"
python -m playwright install chromium
```

### Directory layout

```
ai-stock-analysis/
├── .env                       # secrets
├── config/
│   ├── settings.yaml
│   └── screening_criteria.yaml
├── data/                      # SQLite lives here
├── output/                    # PDFs go here
├── src/stock_analysis/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── logging.py
│   ├── run.py                 # orchestrator
│   ├── screening/
│   ├── data/
│   ├── calc_engine/
│   ├── llm/
│   ├── report/
│   └── mcp_servers/
└── pyproject.toml
```

**Checkpoint:** `pip install -e .` succeeds → package imports → `python -c "import stock_analysis; print('ok')"`

---

## 2) Config + logging

### `src/stock_analysis/config.py` — pydantic-settings

Why pydantic-settings?
- Loads `.env` automatically
- Validates types
- Merges YAML + env cleanly

**Key idea:** separate *secrets* (`.env`) from *parameters* (`settings.yaml`). Secrets must never be committed; parameters are version-controlled.

```python
# Trimmed from the real file — just the pattern
class APIKeysConfig(BaseSettings):
    groq_api_key: str = ""
    fred_api_key: str = ""
    sec_user_agent: str = "StockAnalysis you@example.com"

    model_config = SettingsConfigDict(env_file=".env")

class LLMConfig(BaseModel):
    model_main: str = "llama-3.3-70b-versatile"
    model_qc: str = "llama-3.1-8b-instant"
    max_tokens_thesis: int = 2000
    # ...

class Settings(BaseSettings):
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # ...

settings = load_settings()   # singleton
```

### `logging.py` — structlog (JSON)

Why structlog? You want **traceability for every number**. JSON logs are trivial to grep/filter later.

```python
import structlog
structlog.configure(processors=[
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
def get_logger(name): return structlog.get_logger(name)
```

**Checkpoint:** `python -c "from stock_analysis.config import settings; print(settings.llm.model_main)"`

---

## 3) Database + Cache

### Why do we need a DB at all?

- **Cache** — prevent yfinance (429) + SEC (10 req/s) rate limits
- **Audit trail** — every number must be traceable back to its source

### `database.py` — SQLAlchemy models

Six tables:

```python
class CacheEntry(Base):
    __tablename__ = "cache"
    key = Column(String(200), primary_key=True)     # "yf_info:AAPL"
    value_json = Column(Text, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)

class PriceHistory(Base): ...
class Financial(Base): ...
class Filing(Base): ...
class Valuation(Base): ...
class Report(Base): ...

def init_db():
    engine = create_engine(f"sqlite:///{settings.database.path}")
    Base.metadata.create_all(engine)
```

### `data/cache.py` — TTL wrapper

Key pattern — **stale-fallback**: if the fetcher fails, return the stale cached value rather than crashing.

```python
def cached(key: str, ttl_hours: float, fetcher: Callable) -> Any:
    session = get_session()
    entry = session.query(CacheEntry).filter_by(key=key).first()
    if entry and (datetime.utcnow() - entry.fetched_at) < timedelta(hours=ttl_hours):
        return json.loads(entry.value_json)

    try:
        value = fetcher()
    except Exception as e:
        if entry: return json.loads(entry.value_json)  # fallback
        return None

    # upsert
    ...
    return value
```

**Lesson:** every external API call should be wrapped in `cached()`. It's the simplest defense.

**Checkpoint:** running creates `data/stock_analysis.db` with a `cache` table.

---

## 4) Data ingestion

### Three sources — each with different personality

| Source | Library | Rate limit | Pain points |
|---|---|---|---|
| SEC EDGAR | httpx + custom | 10 req/s (User-Agent required) | XBRL tags change across fiscal years |
| yfinance | yfinance | no public limit but 429s often | unstable, return types shift |
| FRED | fredapi | has key, low rate | lightly used |

### SEC EDGAR — two primary endpoints

**1. Company Facts** (XBRL): `/api/xbrl/companyfacts/CIK{cik}.json`
- Must resolve **CIK** first via `company_tickers.json`
- Ticker variants: `BRK-B` (yf), `BRK.B` (common), `BRKB` (SEC) — try all
- Returns nested: `facts.us-gaap.Revenues.units.USD[].val`

**2. Submissions**: `/submissions/CIK{cik}.json`
- List of 10-K, 10-Q, DEF 14A, 8-K
- Use accession numbers to fetch filing text

**XBRL tag union** — critical insight:
```python
# MSFT's revenue tag changed over time — must merge variants
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
# Merge all variants, then dedupe by period_end
```

**Watch out for:**
- SEC requires a User-Agent with a real name + email (`"MyApp you@example.com"`)
- Omit it and you'll be blocked immediately
- Parallel fetch — stay under 10 req/s using `asyncio-throttle`

### yfinance wrapper

**What to pull:**
```python
t = yf.Ticker(ticker)
t.info                      # fundamentals (dict) — cache 12h
t.history(period="2y")      # price OHLCV
t.income_stmt / balance_sheet / cashflow
t.insider_transactions      # ownership — cache 24h
t.institutional_holders
t.calendar                  # next earnings — cache 6h
```

**Gotchas:** yfinance sometimes returns `DataFrame`, sometimes `dict`, sometimes `None`. Code must be defensive.

```python
def _df_or_none_to_records(df) -> list[dict]:
    if df is None: return []
    try:
        if df.empty: return []
        return json.loads(df.reset_index().to_json(orient="records"))
    except Exception:
        return []
```

### `data/fetch.py` — orchestrator

```python
async def fetch_all(ticker: str) -> dict:
    sec, market, macro = await asyncio.gather(
        fetch_sec_data(ticker),
        asyncio.to_thread(fetch_market_data, ticker),  # yfinance is sync
        asyncio.to_thread(fetch_macro_data),
    )
    return {"sec": sec, "market": market, "macro": macro}
```

**Lesson:** yfinance has no async API → wrap it in `asyncio.to_thread` so it doesn't block the event loop.

**Checkpoint:** `python -m stock_analysis.data.fetch AAPL` → JSON dump from all three sources.

---

## 5) Screening

### Goal: pick the strongest weekly US gainers / losers

### `screening/us_universe.py`

Pull the universe from the **NASDAQ Trader Symbol Directory** — fast, free, updated daily:
- `nasdaqlisted.txt` — NASDAQ
- `otherlisted.txt` — NYSE, AMEX

~5,700 tickers after filtering out ETFs, warrants, test issues.

### `screening/screener.py`

```python
def run_screener(lookback_days=5, top_n=3):
    universe = get_us_tickers()                 # ~5,700 tickers
    # yfinance batch download (400 tickers/chunk)
    data = yf.download(universe, period=f"{lookback_days+5}d", group_by="ticker")
    # compute % change + liquidity filter + rank
    rows = []
    for t in universe:
        df = data[t]
        if df.empty: continue
        if df["Close"].mean() < 5: continue      # drop penny stocks
        pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        adv = (df["Close"] * df["Volume"]).mean()
        if adv < 5e6: continue                   # drop low volume
        rows.append({"ticker": t, "pct_change": pct, ...})

    gainers = sorted(rows, key=lambda r: -r["pct_change"])[:top_n]
    losers  = sorted(rows, key=lambda r: +r["pct_change"])[:top_n]
    return [{"mover_type":"gainer", **g} for g in gainers] + \
           [{"mover_type":"loser",  **l} for l in losers]
```

**Lesson:** batch download beats per-ticker loop. ~5,700 tickers takes 2-3 minutes.

**Checkpoint:** `python -m stock_analysis.screening.screener` → prints top movers.

---

## 6) Calc engine

### Industry classifier — why?

**Insight:** DCF-FCFF does not fit every industry:
- Banks → Residual Income Model (BVPS + ROE-Ke)
- REITs → AFFO-based
- Energy / Mining → Normalized DCF (cyclical)
- Utilities → DDM (dividend-focused)

### `calc_engine/industry_classifier.py`

```python
SECTOR_MODEL_MAP = {
    "Information Technology": ValuationModel.DCF_FCFF,
    "Financials": ValuationModel.RESIDUAL_INCOME,
    "Real Estate": ValuationModel.AFFO_REIT,
    "Energy": ValuationModel.DCF_CYCLICAL,
    "Utilities": ValuationModel.DDM,
    # ...
}

TICKER_OVERRIDES = {
    "BRK-B": ValuationModel.RESIDUAL_INCOME,  # Berkshire is unique
    "O": ValuationModel.AFFO_REIT,
}

def classify_ticker(ticker, sector="", sub_industry="") -> ValuationModel:
    # 1. ticker override → 2. sub-industry → 3. sector → 4. fallback
```

### DCF-FCFF (core model)

**What to compute:**
```
WACC = E/V * Ke + D/V * Kd * (1-t)
Ke   = Rf + β * ERP                              (CAPM)
FCFF = EBIT*(1-t) + D&A - CapEx - ΔWC
Years 1-5: project FCFF at the growth rate
Terminal: FCFF_5 * (1+g) / (WACC-g)
EV       = Σ FCFF_t / (1+WACC)^t + TV/(1+WACC)^5
Equity   = EV - Debt + Cash
Fair Value/share = Equity / Shares Outstanding
```

**Real code (trimmed):**
```python
@dataclass
class DCFAssumptions:
    revenue_base: float
    revenue_growth_rates: list[float]
    operating_margin: float
    tax_rate: float
    risk_free_rate: float
    market_risk_premium: float
    beta: float
    terminal_growth_rate: float
    projection_years: int = 5
    # ...

def run_dcf(a: DCFAssumptions, current_price: float) -> DCFResult:
    # Project revenues with growth fading toward terminal
    revenues = [a.revenue_base]
    for i in range(a.projection_years):
        g = a.revenue_growth_rates[i] if i < len(a.revenue_growth_rates) \
            else fade_to_terminal(i, a)
        revenues.append(revenues[-1] * (1 + g))
    # EBIT → NOPAT → FCFF → discount → sum
    ...
```

**Hard rule:** the LLM **never** touches a number — every value is computed in Python.

### Comparables + Sensitivity

**Comparables** — compare peers in the same GICS sub-industry:
```python
# Pull EV/EBITDA, P/E, P/S from yfinance for 5-10 peers
# Compute median/mean → target multiple
# Apply to our ticker → implied fair value
```

**Sensitivity** — 2D grid:
```python
# WACC × Terminal Growth = 5×5 table
# Each cell = re-run DCF with those params
# Shows fair value across scenarios
```

### Ratios dashboard

**Profitability / Returns / Leverage / Efficiency / Multiples** — computed from income + balance + price (see `calc_engine/ratios.py`).

**Checkpoint:** `python -m stock_analysis.calc_engine.valuate AAPL` → prints fair value, WACC, upside.

---

## 7) LLM layer (Groq)

### Architecture rule: LLM = writer, not calculator

**6 prompts:**

| Prompt | Model | Purpose |
|---|---|---|
| `thesis_generator.txt` | 70b | bull/base/bear + catalysts + rating |
| `risk_synthesizer.txt` | 70b | financial/operational/market/macro risks |
| `industry_commentary.txt` | 70b | competitive positioning |
| `meeting_synthesizer.txt` | 70b | DEF 14A + 8-K synthesis |
| `annual_report_summary.txt` | 70b | 10-K → segments, MD&A, strategic initiatives |
| `qc_checker.txt` | 8b-instant | sanity-check valuation numbers |

### `llm/groq_client.py` — essentials

**Core `call_llm()`** with throttle + retry:
```python
_RPM_LIMIT = 25                  # Groq free = 30, keep a buffer
_MIN_GAP_SECONDS = 2.0
_REQUEST_LOG: list[float] = []

def _throttle():
    now = time.monotonic()
    _REQUEST_LOG[:] = [t for t in _REQUEST_LOG if now - t < 60]
    if len(_REQUEST_LOG) >= _RPM_LIMIT:
        time.sleep(60 - (now - _REQUEST_LOG[0]) + 0.5)
    if _REQUEST_LOG and (now - _REQUEST_LOG[-1]) < _MIN_GAP_SECONDS:
        time.sleep(_MIN_GAP_SECONDS - (now - _REQUEST_LOG[-1]))
    _REQUEST_LOG.append(time.monotonic())

def call_llm(prompt, system_prompt="", ...):
    for attempt in range(max_retries):
        try:
            _throttle()
            return client.chat.completions.create(...).choices[0].message.content
        except RateLimitError as e:
            wait = float(e.response.headers.get("Retry-After", 2**attempt*10))
            time.sleep(wait)
```

### Anti-hallucination tactics

1. **Prompt bans**: "cite or don't claim — never invent numbers"
2. **JSON mode**: `response_format: {"type": "json_object"}` forces structured output
3. **Slim payload**: strip data the LLM doesn't need before sending (`_slim_valuation_bundle`)
4. **Humanize filter (Jinja)**: regex catches `$12,000,000,000` → `$12.0B` in case the LLM still emits long digits

### Prompt template pattern

```
Analyze {company_name} ({ticker}) ...

## Financial Data
{financials}

## Valuation Results
{valuation}

## Instructions
1. Bull Case — ...
2. Base Case — ...
3. Bear Case — ...

IMPORTANT:
- ONLY reference numbers from the provided data
- Do NOT invent financial figures
- Format large amounts as $XB, not full digit strings

Respond in this JSON format:
{{
  "bull_case": {{ "thesis": "...", "key_drivers": [...] }},
  ...
}}
```

**Hard-won lesson:** LLMs love to hallucinate revenue numbers → you need to **slim the payload as much as possible** before sending. The QC model (8b) has only 6,000 TPM, so the slimming must be especially aggressive.

**Checkpoint:** `python -m stock_analysis.llm.analyze AAPL` → returns JSON with thesis/risks/industry/meetings/annual_summary/qc.

---

## 8) Report generation

### Flow: analysis dict → HTML → PDF

```
analyze_ticker() returns dict
     ↓
build_context()  — flatten for Jinja
     ↓
render_html()    — Jinja2 + filters + includes
     ↓
temp HTML file
     ↓
Playwright Chromium (headless, media="print")
     ↓
page.pdf(A4, margins, header/footer)
     ↓
output/TICKER_report_YYYYMMDD.pdf
```

### `report/charts.py`

```python
import matplotlib
matplotlib.use("Agg")   # headless

def chart_revenue_history(history, projected=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["period_end"][:4] for r in history], [r["revenue"]/1e9 for r in history])
    if projected:
        ax.bar([f"F{i+1}" for i in range(len(projected))], [p/1e9 for p in projected])
    ax.set_title("Revenue ($B)")
    return _fig_to_base64(fig)

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
```

**Lesson:** base64-embedding PNG beats saving separate files — Jinja renders → PDF always includes the image.

### `report/generator.py` essentials

```python
def build_context(analysis):
    # flatten the nested valuation bundle
    return {"ticker": ..., "fair_value": ..., "thesis": ..., "charts": charts, ...}

def render_html(analysis):
    env = Environment(loader=FileSystemLoader([TEMPLATES_DIR, STATIC_DIR]),
                      autoescape=select_autoescape(["html"]))
    env.filters["fmt_bn"] = _fmt_bn
    env.filters["humanize_nums"] = _humanize_nums   # regex $X,XXX,XXX,XXX → $X.XB
    template = env.get_template("report.html")
    return template.render(**build_context(analysis))

def generate_pdf(analysis, output_path=None):
    html = render_html(analysis)
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w") as f:
        f.write(html); f.flush()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"file:///{f.name}")
            page.emulate_media(media="print")
            page.pdf(path=output_path, format="A4",
                     margin={"top":"14mm","bottom":"14mm","left":"12mm","right":"12mm"},
                     display_header_footer=True,
                     header_template='...<span class="pageNumber"></span>...')
            browser.close()
```

### `report.html` — finance-grade PDF

Structure — up to 10 pages:
1. **Cover** — rating chip / 4-card summary / screening banner / earnings banner / KV
2. **Investment Thesis** — bull/base/bear blocks + catalysts (uses `humanize_nums` filter)
3. **Business Overview** (10-K) — segments / MD&A / strategy (from `annual_summary` LLM call)
4. **Financial Analysis** — 10y history table + revenue chart + quarterly + ratios
5. **Valuation** — branches on `primary.model` (FCFF/DDM/RI/AFFO/Cyclical)
6. **Peers** — EV/EBITDA bar
7. **Meetings & Events** — DEF 14A + 8-K
8. **Ownership** — insider + institutional
9. **Risks** — severity / likelihood
10. **Appendix** — assumptions + SEC filings + QC

### `report.css` key details

```css
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }

/* finance detail — numbers always align by digit position */
.fin-table td, .kv-table td, .data-table td {
    font-variant-numeric: tabular-nums;
}

@page { size: A4; margin: 0; }
.page { page-break-after: always; padding: 18mm 14mm; }
@media print {
    table, figure, .thesis-block { page-break-inside: avoid; }
}
```

**Key lessons:**
- Use **Playwright Chromium** instead of WeasyPrint because WeasyPrint needs GTK on Windows and struggles with flex/grid
- Playwright's sync API must not be called inside an event loop → use `asyncio.to_thread(generate_pdf, analysis)`

**Checkpoint:** `python -m stock_analysis.run --tickers AAPL` → PDF appears in `output/`.

---

## 9) Orchestration

### `run.py` — tying it all together

```python
async def run_pipeline(tickers=None, movers="both", ...):
    init_db()

    # Step 1: get tickers
    if tickers:
        candidates = [{"ticker": t} for t in tickers]
    else:
        candidates = run_screener(movers=movers)

    # Step 2: analyze each
    results = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            analysis = await analyze_ticker(ticker)
            analysis["screening"] = candidate   # carry metadata
            pdf_path = await asyncio.to_thread(generate_pdf, analysis)
            results.append({"ticker": ticker, "report_path": str(pdf_path)})
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    # Step 3: summary JSON
    save_summary(results, output_dir)
    return results
```

**CLI args:**
```python
parser.add_argument("--tickers")              # CSV list
parser.add_argument("--movers", choices=["gainers","losers","both"])
parser.add_argument("--output", "-o")
```

**Checkpoint:** `python -m stock_analysis.run` → picks tickers itself + produces all PDFs.

---

## 10) MCP server (optional)

### Why have one?

So Claude Desktop / IDE agents can call SEC EDGAR tools directly, without going through a Python script.

### `mcp_servers/sec_edgar_server.py`

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("sec-edgar")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="get_company_facts", description="...", inputSchema={...}),
        Tool(name="get_filings", ...),
        # ...
    ]

@app.call_tool()
async def call_tool(name: str, args: dict):
    if name == "get_company_facts":
        result = await get_company_facts(args["ticker"], httpx_client)
        return [TextContent(type="text", text=json.dumps(result))]
    # ...

def main():
    import asyncio
    asyncio.run(app.run_stdio())
```

Register in Claude Desktop `config.json`:
```json
{"mcpServers": {"sec-edgar": {"command": "sec-mcp-server"}}}
```

---

## 11) Common pitfalls

### Pitfall 1: ticker format mismatch

**Symptom:** `BRK-B` works on yfinance but fails on SEC
**Fix:** try 4 variants in `get_cik`:
```python
variants = {t, t.replace("-","."), t.replace(".","-"), t.replace("-","").replace(".","")}
```

### Pitfall 2: XBRL sparse data (small-cap / new IPO)

**Symptom:** `KeyError: 'value'` because `financials["total_debt"] = []`
**Fix:** defensive access
```python
total_debt = financials.get("total_debt") or []
debt_val = (total_debt[0].get("value") if total_debt else 0) or 0
```

### Pitfall 3: LLM 413 Payload Too Large

**Symptom:** QC LLM (8b, 6,000 TPM) receives a bundle with 50 rows of institutional holders → boom
**Fix:** slim the bundle before sending:
```python
b.pop("ownership", None)
b.pop("earnings_calendar", None)
b.pop("ratios", None)
```

### Pitfall 4: Groq 429 rate limit

**Symptom:** 6 calls/ticker × 6 tickers = 36 calls back-to-back → rate limited
**Fix:** sliding-window throttle (shown above) + honor `Retry-After`

### Pitfall 5: yfinance cache is stale

**Symptom:** a stock rallied 10% but prices still look old
**Fix:** TTL 12h for `info`, 6h for `calendar` (not every field needs to be fresh)

### Pitfall 6: Playwright inside asyncio

**Symptom:** `RuntimeError: Cannot nest sync_playwright() inside asyncio`
**Fix:** `await asyncio.to_thread(generate_pdf, analysis)`

### Pitfall 7: SEC blocks without User-Agent

**Symptom:** 403 Forbidden
**Fix:** `headers={"User-Agent": "MyApp you@example.com"}` — show them you're a human, not a bot

### Pitfall 8: yfinance return type flips

**Symptom:** today `.calendar` returns a dict, tomorrow it returns a DataFrame
**Fix:** defensive wrapper:
```python
if isinstance(cal, dict): ...
elif hasattr(cal, "columns"): ...
else: return {}
```

---

## 12) Next steps (extending the project)

Once you've reached the current state, try building:

### Short-term
- [ ] **Unit tests** — pytest for calc_engine (valuation math)
- [ ] **Backtest** — record ratings every week and compare price at +3m, +6m
- [ ] **ESG section** — Yahoo sustainability scores
- [ ] **Email delivery** — send yourself the PDFs every Friday night

### Mid-term
- [ ] **Streamlit dashboard** — interactive view of the pipeline
- [ ] **GitHub Actions schedule** — weekly run + commit reports
- [ ] **Cost tracking** — log Groq tokens per run
- [ ] **Ticker watchlist** — separate DB table for user-pinned names

### Long-term
- [ ] **ML earnings model** — predict EPS surprise
- [ ] **Alternative data** — Reddit / news sentiment
- [ ] **Portfolio view** — optimize allocation across ratings
- [ ] **Multi-LLM** — Claude/GPT fallback when Groq is down

---

## Summary — the learning path

1. **Python async** — not just `asyncio.sleep`, but async HTTP + mixed sync bridging
2. **API hardening** — cache + retry + backoff + stale-fallback
3. **Finance domain** — 5 valuation models, each with its own assumptions
4. **LLM engineering** — prompt design + anti-hallucination + rate limits
5. **PDF pipeline** — picking the right tool (Playwright > WeasyPrint on Windows)
6. **Orchestration** — per-ticker error handling so one failure doesn't kill the batch
7. **Observability** — structlog JSON, every number traceable


Happy building 🚀
