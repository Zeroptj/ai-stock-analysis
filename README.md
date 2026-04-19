# AI Stock Analysis — Automated US Equity Research

An end-to-end system that scans the **entire US stock universe**, picks the week's top movers, pulls filings from SEC EDGAR, routes each name to the right valuation model for its industry, writes narrative with an LLM, and produces a **buy-side-style PDF research report** via headless Chromium.

> Note: `project.md` is the original concept document (S&P 500 + TradingView MCP + WeasyPrint) and **no longer matches** the actual implementation. This README describes the code as it is today.

---

## 📐 System Architecture

### Layer Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     LAYER 1: ORCHESTRATION                        │
│   Python asyncio Controller  (run.py / analyze.py / valuate.py)   │
└──────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  LAYER 2A      │   │  LAYER 2B          │   │  LAYER 2C          │
│  Screening     │   │  Data Ingestion    │   │  Calc Engine       │
│                │   │                    │   │                    │
│ • NASDAQ       │   │ • SEC EDGAR        │   │ • Industry Router  │
│   Trader list  │   │   (10-K / 10-Q /   │   │ • DCF-FCFF         │
│   (~5,700 US)  │   │    DEF 14A / 8-K)  │   │ • DDM              │
│ • yfinance     │   │ • yfinance quotes  │   │ • Residual Income  │
│   batch prices │   │ • FRED (macro)     │   │ • AFFO (REIT)      │
│ • weekly %     │   │                    │   │ • DCF Cyclical     │
│   → top 10/10  │   │ → httpx async      │   │ • Comparables      │
└────────────────┘   └────────────────────┘   │ • 2D Sensitivity   │
                                              └────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │   LAYER 3: STORAGE       │
                    │   SQLite (SQLAlchemy)    │
                    │   TTL cache (6h/12h/24h) │
                    │   + audit trail          │
                    └──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │   LAYER 4: LLM (Groq)    │
                    │   - Investment thesis    │
                    │   - Risk synthesizer     │
                    │   - Industry commentary  │
                    │   - Meeting synthesis    │
                    │     (DEF 14A + 8-K)      │
                    │   - 10-K annual summary  │
                    │     (MD&A + segments)    │
                    │   - QC / sanity check    │
                    └──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │   LAYER 5: OUTPUT        │
                    │   Jinja2 HTML →          │
                    │   Playwright Chromium →  │
                    │   A4 PDF (print CSS)     │
                    └──────────────────────────┘

          [ Sidecar: SEC EDGAR MCP Server (stdio) ]
          exposes 6 tools callable directly from Claude / IDE agents
```

### Why not the original `project.md`?

| Original (project.md) | Actual (in repo) | Reason |
|---|---|---|
| TradingView MCP → filter S&P 500 | yfinance batch download → weekly movers across the whole US universe | No free TradingView MCP exists; we want broader coverage than S&P 500 |
| WeasyPrint | Playwright Chromium | WeasyPrint needs GTK on Windows and struggles with modern CSS; Chromium renders flex / grid / web fonts faithfully |
| Single DCF model | 5 models routed by GICS sector | DCF-FCFF doesn't fit Banks / REITs / Cyclicals |
| Technical-indicator filter | Weekly % change + liquidity floor | User wants "what moved hard this week", not an EMA/RSI filter |

---

## 🧱 Layer-by-Layer Detail

### LAYER 1 — Orchestration

Three entry points at different levels of completeness — pick whichever matches your need:

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│  valuate.py        │◀───▶│  analyze.py        │◀───▶│  run.py            │
│  ─────────────     │     │  ─────────────     │     │  ─────────────     │
│  numbers only      │     │  numbers + LLM     │     │  full end-to-end   │
│  (no LLM calls)    │     │  narrative         │     │  screening → PDF   │
└────────────────────┘     └────────────────────┘     └────────────────────┘
```

- **Framework:** Python 3.11+ / `asyncio` + `httpx`
- **Config:** `pydantic-settings` — secrets in `.env`, parameters in `config/*.yaml`
- **Logging:** `structlog` (JSON) — every number can be traced back to its source
- **Threading:** Playwright's sync API is dispatched via `asyncio.to_thread` so it doesn't collide with the event loop

---

### LAYER 2A — Screening (Weekly Movers)

Replaces TradingView MCP with an in-house pipeline that's more stable and broader:

```
Input:  NASDAQ Trader Symbol Directory (nasdaqlisted.txt + otherlisted.txt)
        ~5,700 US common stocks (ETFs, warrants, test issues filtered out)

Step 1: yfinance batch download (chunks of 400 tickers)
        → price history over `lookback_days` (default = 5 trading days)

Step 2: Filter  — min_price ≥ $5, avg daily $ volume ≥ $5M

Step 3: Rank    — compute % change, sort to find top gainers / losers

Output: top N gainers + top N losers (default 10/10)
```

**Key files:**
- `screening/us_universe.py` — fetches the NASDAQ Trader lists, caches at module level
- `screening/screener.py` — batch fetch, return computation, returns a flat list tagged with `mover_type`

**Why build it ourselves:** No public/free TradingView MCP is available; yfinance batch download is fast enough (~2–3 min for all 5,700 tickers) and isn't rate-limited like an API.

---

### LAYER 2B — Data Ingestion

Three data sources, each wrapped as a module with the same `get_data(ticker) → dict` interface:

| Module | Source | Used for | Cache TTL |
|---|---|---|---|
| `data/sec_edgar_client.py` | SEC EDGAR XBRL Company Facts + Submissions | 10-K, 10-Q, DEF 14A, 8-K — 10y annual + quarterly, plus filing text | 24h |
| `data/market_data_client.py` | yfinance `.info` / `.history` / `.financials` | Current price, basic info, sector, fundamentals | 12h |
| `data/market_data_client.py` | yfinance `.insider_transactions` / `.institutional_holders` / `.major_holders` | Ownership panel in PDF | 24h |
| `data/market_data_client.py` | yfinance `.calendar` | Next earnings date + EPS/Revenue estimates | 6h |
| `data/macro_data.py` | FRED API | Risk-free rate (DGS10), CPI | (in-memory) |
| `data/cache.py` | SQLite `cache` table | Generic TTL-backed JSON cache with stale-fallback | — |
| `data/fetch.py` | — | Orchestrator: merges all sources in a single call | — |

**Key behaviors of the SEC client:**
- **Tag union per metric** — XBRL tags change over time (e.g. MSFT switched `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax`). We merge every tag variant and dedupe by `period_end` so recent years aren't lost.
- **10 years + quarterly** — annual data back to 10 years; quarterly data also extracted for the Quarterly Trend table.
- **Filing text** — fetches the primary document HTML → regex-strips tags → passes to the LLM (truncated to 20K chars for 10-K, 15K for DEF 14A, 5K for 8-K to fit under Groq's TPM cap).
- **Flexible CIK lookup** — `get_cik()` tries `BRK-B`, `BRK.B`, and `BRKB` variants so yfinance-format tickers still resolve on EDGAR.
- **SEC User-Agent** — mandatory per SEC policy. Set via `settings.yaml → api_keys.sec_user_agent` (name + email).

**Caching:**
- `cached(key, ttl_hours, fetcher)` wraps every external call. If the stored entry is fresh it returns immediately; if the fetcher throws (rate-limited / offline) it falls back to stale cache rather than failing. This is the single most effective tool against yfinance 429s.

---

### LAYER 2B+ — SEC EDGAR MCP Server

A sidecar MCP stdio server that exposes the SEC client as tools, so Claude Desktop or IDE agents can call EDGAR directly instead of piping through a Python script.

**File:** `mcp_servers/sec_edgar_server.py` (installed as console script `sec-mcp-server`)

**Tools exposed:**
| Tool | Purpose |
|---|---|
| `get_company_facts` | Full XBRL blob for a ticker |
| `get_financial_summary` | Key metrics, 5 years |
| `get_filings` | List of filings by form type |
| `get_proxy_statements` | DEF 14A list with metadata |
| `get_material_events` | 8-K list with metadata |
| `get_filing_text` | Full text of a filing (HTML → plain text) |

**Claude Desktop config entry:**
```json
{
  "mcpServers": {
    "sec-edgar": { "command": "sec-mcp-server" }
  }
}
```

---

### LAYER 2C — Calc Engine (Industry-Routed)

```
                    ┌──────────────────────────────┐
                    │  classify_ticker()            │
                    │  ticker / sector / sub_ind    │
                    │  → ValuationModel             │
                    └──────────────┬───────────────┘
                                   │
  ┌──────────┬─────────────┬───────┴───────┬──────────────┬──────────┐
  ▼          ▼             ▼               ▼              ▼
┌────────┐┌────────┐ ┌───────────────┐ ┌──────────┐ ┌──────────────┐
│ DCF    ││  DDM   │ │  Residual     │ │  AFFO    │ │  DCF         │
│ FCFF   ││ (2-stg)│ │  Income       │ │  (REIT)  │ │  Cyclical    │
├────────┤├────────┤ ├───────────────┤ ├──────────┤ ├──────────────┤
│Tech    ││Utility │ │Banks /        │ │Equity /  │ │Energy /      │
│Health  ││Stable  │ │Financials     │ │Mortgage  │ │Materials /   │
│Consumer││dividend│ │(BVPS + ROE    │ │REITs     │ │Mining        │
│Industr.││payers  │ │−Ke×BV)        │ │(P/AFFO   │ │(normalized   │
│        ││        │ │               │ │+ NAV)    │ │revenue)      │
└────────┘└────────┘ └───────────────┘ └──────────┘ └──────────────┘
```

**Files:**
- `calc_engine/industry_classifier.py` — ticker override → sub-industry → sector → fallback
- `calc_engine/dcf_fcff.py` — 2-stage FCFF + WACC + Gordon terminal
- `calc_engine/dcf_ddm.py` — 2-stage DDM, CAGR from SEC `CommonStockDividendsPerShareDeclared`
- `calc_engine/residual_income.py` — RI = (ROE − Ke) × BVPS, driven by SEC equity + net income
- `calc_engine/affo_reit.py` — FFO from `NetIncome + Depreciation`, dividend from SEC / DEF 14A
- `calc_engine/dcf_cyclical.py` — normalized revenue / margin by cycle position
- `calc_engine/comparables.py` — EV/EBITDA, P/E, P/S against the peer set
- `calc_engine/sensitivity.py` — 2D grid: WACC × terminal growth
- `calc_engine/valuate.py` — the router + normalized output payload

**Hard rule:** the LLM never touches a number — every value is computed in Python.

---

### LAYER 3 — Storage

SQLite via SQLAlchemy at `data/stock_analysis.db` (configurable).

- **TTL-backed cache** (`cache` table) — keyed by `yf_info:{ticker}`, `yf_ownership:{ticker}`, `yf_calendar:{ticker}`, `sec_facts:{ticker}`. Values stay fresh for 6h / 12h / 24h depending on how fast the underlying data decays. On fetcher failure we fall back to the stale value rather than crashing.
- **Audit trail** — `prices` and `financials` tables capture the raw values used for each run.
- **Schema summary:**

| Table | Purpose | Wired? |
|---|---|---|
| `cache` | Generic TTL-backed JSON blob — used by every external API wrapper | ✅ |
| `prices` | Daily OHLCV snapshot | ✅ (write-only audit) |
| `financials` | Income / balance / cash-flow dump from yfinance | ✅ (write-only audit) |
| `filings` | SEC filing metadata | schema ready |
| `valuations` | Valuation result per run | schema ready |
| `reports` | Generated PDF metadata | schema ready |

Clear caches:
```bash
sqlite3 data/stock_analysis.db "DELETE FROM cache WHERE key LIKE 'yf_%';"   # yfinance only
sqlite3 data/stock_analysis.db "DELETE FROM cache;"                          # all API cache
```

---

### LAYER 4 — LLM (Groq)

LLM is used only for writing and interpretation — never for calculation.

6 calls per ticker, routed to the model best-fit for each task:

| Task | Model | Token budget | Prompt file |
|---|---|---|---|
| Investment thesis (bull/base/bear + catalysts) | `llama-3.3-70b-versatile` | 2,000 | `thesis_generator.txt` |
| Risk synthesizer (financial / operational / market / macro) | `llama-3.3-70b-versatile` | 1,500 | `risk_synthesizer.txt` |
| Industry commentary (positioning, strengths, weaknesses) | `llama-3.3-70b-versatile` | 1,500 | `industry_commentary.txt` |
| **Meeting synthesis** (DEF 14A + 8-K → structured JSON) | `llama-3.3-70b-versatile` | 1,500 | `meeting_synthesizer.txt` |
| **10-K annual summary** (business overview / MD&A / segments / risks) | `llama-3.3-70b-versatile` | 2,000 | `annual_report_summary.txt` |
| QC / sanity check | `llama-3.1-8b-instant` | 500 | `qc_checker.txt` |

**Anti-hallucination techniques:**
- Every prompt enforces "cite or don't claim — never invent numbers"
- Payloads are slimmed via `_slim_primary` + `_slim_valuation_bundle` — sensitivity grids, SEC text, ownership lists, earnings calendar, and ratios dashboard are stripped before being sent to QC. This saves tokens and reduces the chance the LLM picks stray numbers out of an appendix.
- `response_format: json_object` forces structured output
- Thesis prompt explicitly instructs compact units ("$12.3B" not "$12,300,000,000"); a Jinja `humanize_nums` post-filter in `generator.py` acts as a safety net that regex-converts any 7-plus-digit amount that slips through.

**Rate-limit handling:**
- **Sliding-window throttle** in `groq_client._throttle()` — caps at 25 RPM with a 2s minimum gap between calls (Groq free tier is 30 RPM; we keep a buffer).
- **Retry-After-aware retries** — 429 responses honour the `Retry-After` header; other errors use exponential backoff (max 90s).
- **Per-model TPM awareness** — `llama-3.1-8b-instant` (QC) has only 6K TPM, so the QC bundle drops `ownership`/`earnings_calendar`/`ratios` to stay under budget.
- `413 Payload Too Large` triggers proxy/8-K truncation.

---

### LAYER 5 — Output (Playwright PDF)

```
analysis dict
   │
   ▼
build_context()  — flattens bundle into a Jinja-friendly dict
   │
   ▼
render_html()  — Jinja2 template (report.html + includes report.css)
   │
   ▼
temp HTML file
   │
   ▼
Playwright Chromium (headless, emulate_media='print')
   │
   ▼
page.pdf(format='A4', margins, header/footer template)
   │
   ▼
output/TICKER_report_YYYYMMDD.pdf
```

**Template (`report/templates/report.html`)** — up to 9 pages (conditional on data availability):
1. **Cover + Executive Summary** — rating chip, 4-card summary, screening banner, **next-earnings banner** (if calendar data), key multiples
2. **Investment Thesis** — Bull / Base / Bear + Catalysts (big-dollar amounts auto-humanized to $X.XB)
3. **Business Overview (10-K)** — business summary, segment highlights, MD&A themes, strategic initiatives, disclosed risks (from `annual_summary` LLM call)
4. **Financial Analysis** — 10-year historical table, revenue bar chart with projection caption, quarterly trend, key ratios dashboard
5. **Valuation** — branches on `primary.model` (DCF-FCFF / DDM / RI / AFFO / Cyclical each get a different layout); sensitivity heatmap with read-me caption
6. **Peers & Industry Context** — EV/EBITDA peer bar with caption
7. **Shareholder Meeting & Material Events** — DEF 14A + 8-K → structured section
8. **Ownership & Insider Activity** — holder breakdown, insider transactions, top institutional holders (only when data exists)
9. **Risks** — financial / operational / market / macro
10. **Appendix** — assumptions dump, SEC filings referenced, QC flags, disclaimer

**Chart captions (non-LLM):** every chart has a `.chart-caption` below it that explains the data source and math. F1–F5 in the revenue chart is labelled as Forecast Years 1–5, with the actual growth rates used printed inline. The sensitivity heatmap caption explains how to read the WACC × terminal-growth grid. Peer chart caption explains the median line.

**Stylesheet (`report/static/report.css`)** — finance-grade:
- Unified sans stack: Segoe UI → Helvetica Neue → Arial
- `font-variant-numeric: tabular-nums` on every table — columns line up by digit position
- Navy accent `#0B3D91`, serif-free, print-safe
- `@page` + `display_header_footer` → ticker + page number on every page

**Charts (`report/charts.py`)** — matplotlib → base64 PNG embed:
- Revenue history (historical + projected F1–F5)
- Sensitivity heatmap (WACC × terminal growth — FCFF only)
- Peer comparison (EV/EBITDA bar)

---

## 🚀 Installation & Usage

### Install

```bash
# 1. Clone
git clone <repo>
cd ai-stock-analysis

# 2. Create and activate a virtual environment
#    (do NOT skip this — otherwise pip will install into your global Python)
python -m venv .venv

#    Activate:
#    Windows (Git Bash / bash):   source .venv/Scripts/activate
#    Windows (PowerShell):        .venv\Scripts\Activate.ps1
#    Windows (cmd.exe):           .venv\Scripts\activate.bat
#    macOS / Linux:               source .venv/bin/activate

# 3. Install project + dev deps in editable mode
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 4. Install Playwright Chromium (required for PDF rendering)
python -m playwright install chromium

# 5. Copy .env template and fill in API keys
cp .env.example .env
# Edit .env — set GROQ_API_KEY and FRED_API_KEY
```

> **Prefer [`uv`](https://github.com/astral-sh/uv)?** It's much faster:
> ```bash
> uv venv
> source .venv/Scripts/activate     # same activate step as above
> uv pip install -e ".[dev]"
> python -m playwright install chromium
> ```

Once the venv is active your shell prompt is prefixed with `(.venv)`. Every `python` / `pip` / `pytest` command from that point on uses the project's isolated environment. Run `deactivate` to leave.

### Config

**Secrets (`.env`):**
```
GROQ_API_KEY=gsk_xxx
FRED_API_KEY=xxx
```

**Parameters (`config/settings.yaml`):**
- `api_keys.sec_user_agent` — required by SEC (name + email)
- `llm.*` — model selection, token budget, temperature
- `dcf.*` — projection years, terminal growth, WACC sensitivity range
- `report.output_dir` — default `output/`

**Screening (`config/screening_criteria.yaml`):**
- `universe.exchanges` — `[NASDAQ, NYSE, AMEX]`
- `movers.lookback_days` — default 5 (1 trading week)
- `movers.top_gainers / top_losers` — default **3/3** (kept tight to respect LLM rate limits)
- `filters.min_price`, `min_avg_dollar_volume`

### Entry Points

> ⚠️ **Groq free tier users — read this first.**
>
> The pipeline makes **6 LLM calls per ticker** (thesis / risk / industry / meeting / 10-K summary / QC). The free tier caps at **12K TPM** for the 70b model and **6K TPM** for the 8b-instant QC model. In practice this is fine for a **single ticker at a time**, but the **full screening run** (3 gainers + 3 losers = 6 tickers = 36 LLM calls back-to-back) almost always hits 413 / 429 errors, especially because large-cap tickers like KKR / JPM / MSFT have big payloads.
>
> **Recommended on free tier:**
> ```bash
> python -m stock_analysis.run --tickers AAPL          # one ticker — safe
> python -m stock_analysis.run --tickers AAPL,MSFT     # 2–3 tickers — usually OK
> ```
>
> **If you want to run the full pipeline on free tier, pick at least one:**
> 1. Lower `llm.max_tokens_*` in `config/settings.yaml` (thesis 2000 → 1200, annual 2000 → 1200, etc.)
> 2. Drop `movers.top_gainers` / `top_losers` to `1/1` or `2/2` in `config/screening_criteria.yaml`
> 3. Shorten text caps in `groq_client.py` (10-K `text[:20_000]` → `text[:10_000]`, proxy `[:15_000]` → `[:8_000]`)
> 4. Skip the 10-K LLM call (it's already wrapped in try/except so failures don't kill the run, but you can remove it entirely from `analyze.py`)
> 5. Switch `llm.model_main` from `llama-3.3-70b-versatile` to a model with higher TPM on your tier
> 6. **Upgrade to Groq Dev Tier** (~$30–50/mo) — by far the simplest fix
>
> The SQLite cache softens the blow: a failed run can be re-run and all SEC/yfinance data comes from cache — only the LLM calls hit the API again.

| Command | Purpose | Example |
|---|---|---|
| `python -m stock_analysis.data.fetch MSFT` | Fetch all sources → JSON | smoke-test data layer |
| `python -m stock_analysis.calc_engine.valuate MSFT` | Valuation only (no LLM) | inspect DCF numbers |
| `python -m stock_analysis.llm.analyze MSFT` | Valuation + LLM narrative → JSON | debug prompts |
| `python -m stock_analysis.run --tickers MSFT` | Single ticker end-to-end → PDF **(safe on free tier)** | one-off report |
| `python -m stock_analysis.run` | Full pipeline: weekly movers → PDFs **(free tier: see warning above)** | weekly production run |
| `sec-mcp-server` | Start the SEC EDGAR MCP server (stdio) | plug into Claude Desktop |

All commands accept `--output / -o` to override the output path.

### Selecting gainers, losers, or both

`python -m stock_analysis.run` takes a `--movers` flag that restricts which side of the weekly movers gets analyzed. Useful when you're short on time or API quota and only want one side:

| Flag | Behavior |
|---|---|
| `--movers both` *(default)* | Analyze the top gainers **and** top losers |
| `--movers gainers` | Analyze only the top gainers |
| `--movers losers` | Analyze only the top losers |

**Examples:**
```bash
# Full weekly pipeline — both sides
python -m stock_analysis.run

# Top gainers only
python -m stock_analysis.run --movers gainers

# Top losers only, custom output folder
python -m stock_analysis.run --movers losers -o output/losers_week_17

# A fixed watchlist (flag ignored when --tickers is set)
python -m stock_analysis.run --tickers AAPL,MSFT,NVDA
```

The count for each side is controlled by `movers.top_gainers` / `movers.top_losers` in `config/screening_criteria.yaml` (default 3 each).

---

## 📊 Sample Output

```
output/
├── MSFT_report_20260419.pdf       # full PDF (8 pages)
├── JPM_report_20260419.pdf
├── O_report_20260419.pdf
└── pipeline_summary.json          # per-ticker success/fail summary
```

**PDF contents:**
- Cover: rating chip (BUY / HOLD / SELL), 4-card summary, screening banner, next-earnings banner, one-liner thesis, KV table
- Thesis: three colored blocks (bull green, base blue, bear red) — big-dollar amounts auto-formatted as $X.XB
- Business Overview (from 10-K): segment highlights, MD&A themes, strategic initiatives
- Financial Analysis: 10-year SEC history, quarterly trend, ratios dashboard, revenue chart with F1–F5 projection caption
- Valuation: big-number fair value + model-specific table (FCFF has a sensitivity heatmap with read-me caption, REIT shows NAV, Cyclical shows cycle position)
- Meeting & Events: structured content from DEF 14A (voting proposals, directors, exec comp) + 8-K timeline
- Ownership: holder breakdown, insider transactions, top institutional holders
- Appendix: every assumption + SEC filing accessions + QC flags

---

## 🗓️ Status & Known Limitations

### ✅ Done
- Weekly movers screening (full US universe, ~5,700 tickers)
- SEC EDGAR client (XBRL 10y + quarterly + DEF 14A + 8-K + 10-K text)
- SEC MCP server (6 tools)
- Flexible CIK lookup (accepts BRK-B / BRK.B / BRKB variants)
- 5 valuation models + industry router + 2D sensitivity + ratios dashboard
- 6 LLM narrative sections (thesis / risk / industry / meeting / 10-K annual summary / QC)
- **SQLite TTL cache** — keyed per source, with stale-fallback on fetcher errors
- **Ownership panel** (insider transactions + institutional holders) and **earnings calendar banner**
- **Sliding-window LLM throttle** + Retry-After-aware retries
- Playwright PDF generation (up to 10 finance-standard pages with chart captions)

### ⚠️ Limitations / Things to know
- **Groq free tier is single-ticker-friendly, not batch-friendly** — 70b: 12K TPM, 8b-instant (QC): 6K TPM. Use `--tickers SYMBOL` mode (1–3 tickers at a time) on the free tier. Running `python -m stock_analysis.run` without `--tickers` (screener mode, 6 tickers × 6 LLM calls = 36 back-to-back calls) routinely trips 413 / 429. Either lower `llm.max_tokens_*` + `movers.top_gainers/top_losers`, trim text caps in `groq_client.py`, swap models, or upgrade to Dev Tier. Cache absorbs the SEC/yfinance side of failures; only LLM calls need to be re-run.
- **yfinance is not 100% stable** — batch downloads occasionally miss tickers; the cache absorbs transient 429s, and `min_history_days` filters out incomplete data.
- **SEC rate limit = 10 req/sec** — the client uses `asyncio-throttle`; don't fan out too many parallel tickers
- **LLM hallucination risk** — despite strict constraints, always QC manually before acting on a report
- **ESG section not implemented** — mentioned in `project.md` but skipped for now

### 🔜 Not yet built
- Backtest framework (measure recommendation accuracy over time)
- Cron / GitHub Actions scheduling
- Email delivery
- Streamlit dashboard
- Solid unit-test coverage (currently light)
- ESG snapshot section

---

## 📦 Actual Stack

```
Language:        Python 3.11+
Package mgr:     pip + pyproject.toml (hatchling)
Async:           asyncio + httpx + asyncio-throttle
Data:            pandas, numpy, pandas-ta
Financial:       yfinance, fredapi, + custom SEC EDGAR client
LLM:             groq-python (Llama 3.3 70B + 3.1 8B)
MCP:             mcp SDK ≥ 1.0 (stdio server)
Templating:      Jinja2
PDF:             Playwright Chromium   (not WeasyPrint)
Charts:          matplotlib, seaborn
DB:              SQLite + SQLAlchemy 2.x
Config:          pydantic-settings + python-dotenv + YAML
Logging:         structlog (JSON)
Testing:         pytest + pytest-asyncio (optional)
```

---

## 🗂️ Project Layout

```
ai-stock-analysis/
├── .env                        # secrets (gitignored)
├── .env.example
├── config/
│   ├── settings.yaml           # params: SEC UA, LLM, DCF, report
│   └── screening_criteria.yaml # universe + movers + filters
├── pyproject.toml
├── README.md                   # this file
├── project.md                  # original design doc (outdated)
├── ROADMAP.md
└── src/stock_analysis/
    ├── config.py               # pydantic settings loader
    ├── database.py             # SQLAlchemy init
    ├── logging.py              # structlog config
    ├── run.py                  # ★ end-to-end orchestrator
    │
    ├── screening/
    │   ├── us_universe.py      # NASDAQ Trader list
    │   └── screener.py         # weekly movers
    │
    ├── data/
    │   ├── fetch.py            # all-sources orchestrator
    │   ├── sec_edgar_client.py # XBRL + filings + 10-K text + CIK variants
    │   ├── market_data_client.py  # yf info / price / ownership / earnings cal
    │   ├── macro_data.py       # FRED
    │   └── cache.py            # TTL-backed JSON cache (stale-fallback)
    │
    ├── calc_engine/
    │   ├── valuate.py          # ★ industry router
    │   ├── industry_classifier.py
    │   ├── dcf_fcff.py
    │   ├── dcf_ddm.py
    │   ├── residual_income.py
    │   ├── affo_reit.py
    │   ├── dcf_cyclical.py
    │   ├── comparables.py
    │   ├── sensitivity.py
    │   └── ratios.py
    │
    ├── llm/
    │   ├── groq_client.py      # 6 prompt fns + sliding-window throttle
    │   ├── analyze.py          # ★ valuation + narrative
    │   └── prompts/
    │       ├── thesis_generator.txt
    │       ├── risk_synthesizer.txt
    │       ├── industry_commentary.txt
    │       ├── meeting_synthesizer.txt
    │       ├── annual_report_summary.txt
    │       └── qc_checker.txt
    │
    ├── mcp_servers/
    │   └── sec_edgar_server.py # stdio MCP, 6 tools
    │
    └── report/
        ├── generator.py        # ★ Playwright PDF pipeline
        ├── charts.py           # matplotlib → base64
        ├── templates/
        │   └── report.html     # 8-section Jinja template
        └── static/
            └── report.css      # finance-grade print CSS
```

---

## 💰 Approximate Cost (per month)

| Item | Cost |
|---|---|
| Groq API (free tier: 12K TPM 70b / 6K TPM 8b, 14.4K req/day) | **$0** (fits ~15–20 reports/day at 6 LLM calls each) |
| SEC EDGAR | free |
| yfinance | free |
| FRED API | free |
| Playwright Chromium | free (runs locally) |
| SQLite | free |
| **Total** | **$0** (within free-tier limits) |

Upgrade to the Groq Dev Tier (~$30–50/mo) if you need higher throughput.

---

## ⚠️ Disclaimer

This system generates research reports from publicly available data (SEC EDGAR, yfinance, FRED) for educational / prototyping purposes only. **It is not investment advice** — always do your own due diligence before making any investment decision.

All numbers are computed in Python; narrative is written by an LLM using only the data passed to it.
