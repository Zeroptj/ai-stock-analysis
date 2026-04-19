# คู่มือสร้างโปรเจ็กต์นี้จาก 0 — AI Stock Analysis

> คู่มือนี้พาคนที่ยังไม่เคยเขียน project แบบนี้ → สร้างให้ได้เทียบเท่ากับ state ปัจจุบัน (weekly US movers screener + 5 valuation models + LLM narrative + PDF report)
>
> **ไม่ใช่การ copy-paste code** — แต่เป็นการไล่สร้าง layer ทีละชั้น เข้าใจว่าทำไมต้องเป็นอย่างนั้น และแต่ละชั้นแก้ปัญหาอะไรให้กัน

---

## 📑 สารบัญ

1. [ต้องรู้อะไรก่อน (Prerequisites)](#0-prerequisites)
2. [Project skeleton + venv + pyproject](#1-project-skeleton)
3. [Config + logging + settings](#2-config--logging)
4. [Database layer + TTL cache](#3-database--cache)
5. [Data ingestion — SEC / yfinance / FRED](#4-data-ingestion)
6. [Screening — NASDAQ universe + weekly movers](#5-screening)
7. [Calc engine — 5 valuation models + industry router](#6-calc-engine)
8. [LLM layer — Groq + prompts + rate limit](#7-llm-layer)
9. [Report generation — Jinja2 + charts + Playwright PDF](#8-report-generation)
10. [Orchestration — ร้อย pipeline ทั้งหมด](#9-orchestration)
11. [MCP server (optional sidecar)](#10-mcp-server)
12. [Common pitfalls + debugging](#11-pitfalls)
13. [ขั้นต่อไป](#12-next)

---

## 0) Prerequisites

### ต้องรู้อะไรก่อนเริ่ม

**Python skills:**
- Python 3.11+ syntax
- `async` / `await` / `asyncio` — **สำคัญมาก** ทั้ง project ใช้ async เพราะ network-bound
- Type hints (`list[dict]`, `str | None`, dataclass)
- Context managers (`with`, `async with`)
- Package structure (`src/` layout + `__init__.py`)

**Finance concepts (เบื้องต้น):**
- DCF (Discounted Cash Flow) คืออะไร — project ต่ำสุดต้องเข้าใจ FCFF, terminal value, WACC
- Multiples: P/E, EV/EBITDA, P/B, P/S
- Balance sheet vs Income statement vs Cash flow
- Dividend Discount Model, Residual Income (สำหรับธนาคาร), AFFO (สำหรับ REIT)

> ถ้ายังไม่แข็งด้านนี้ แนะนำอ่าน *Damodaran on Valuation* บทแรกๆ ก่อน

**APIs / Web:**
- HTTP status codes (200, 429, 413, 5xx)
- JSON structure
- Rate limits + retry strategies
- REST vs streaming

**Tools:**
- Git
- CLI (bash / PowerShell)
- SQL (SELECT/INSERT พอ)
- Markdown

---

## 1) Project skeleton

### ตั้ง venv

```bash
mkdir ai-stock-analysis && cd ai-stock-analysis
python -m venv .venv
# Windows
source .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
```

### `pyproject.toml` (ใช้ hatchling)

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

**Checkpoint:** `pip install -e .` ผ่าน → import ได้ → `python -c "import stock_analysis; print('ok')"`

---

## 2) Config + logging

### `src/stock_analysis/config.py` — pydantic-settings

ทำไมต้อง pydantic-settings? เพราะ:
- โหลด `.env` อัตโนมัติ
- Validate type ให้
- Merge YAML + env ได้

**แนวคิด:** แยก *secrets* (`.env`) ออกจาก *parameters* (`settings.yaml`). Secret ห้าม commit, parameter commit ได้

```python
# ย่อจากของจริง — เห็น pattern พอ
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

ทำไม structlog? — ต้องการ **traceability ทุกตัวเลข**. JSON log ทำให้ grep/filter ได้ง่าย

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

### ทำไมต้องมี DB?

- **Cache** — กัน rate limit จาก yfinance (429) + SEC (10 req/s)
- **Audit trail** — ทุกตัวเลขต้อง trace กลับได้

### `database.py` — SQLAlchemy models

หัวใจ 6 ตาราง:

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

Pattern สำคัญ — **stale-fallback**: ถ้า fetcher fail → คืน stale cache แทน crash

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

**Lesson:** ทุก API call ที่ยิง external ควรห่อด้วย `cached()` — ง่ายที่สุด

**Checkpoint:** รันแล้ว `data/stock_analysis.db` ถูกสร้าง, มีตาราง cache

---

## 4) Data ingestion

### สามแหล่ง — แต่ละที่มีนิสัยต่างกัน

| Source | Library | Rate limit | Pain points |
|---|---|---|---|
| SEC EDGAR | httpx + custom | 10 req/s (ต้อง User-Agent) | XBRL tag เปลี่ยนตาม fiscal year |
| yfinance | yfinance | ไม่มี public limit แต่ 429 บ่อย | unstable, return type เปลี่ยน |
| FRED | fredapi | มี key, rate ต่ำ | ใช้น้อย |

### SEC EDGAR — สอง endpoint หลัก

**1. Company Facts** (XBRL): `/api/xbrl/companyfacts/CIK{cik}.json`
- ต้อง resolve **CIK** ก่อน จาก `company_tickers.json`
- Ticker variant: `BRK-B` (yf), `BRK.B` (common), `BRKB` (SEC) — ต้องลองทั้งหมด
- Return nested: `facts.us-gaap.Revenues.units.USD[].val`

**2. Submissions**: `/submissions/CIK{cik}.json`
- List ของ 10-K, 10-Q, DEF 14A, 8-K
- ไปดึง filing text ต่อจาก accession number

**XBRL tag union** — critical insight:
```python
# MSFT revenue tag เปลี่ยนตามปี — ต้อง merge ทุก variant
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
# รวมทุก variant แล้ว dedupe ด้วย period_end
```

**ข้อควรระวัง:**
- SEC บังคับ User-Agent ที่ระบุชื่อ + email ชัดเจน (`"MyApp you@example.com"`)
- ถ้าไม่ใส่ → ถูก block ทันที
- Parallel fetch → อย่าเกิน 10 req/s, ใช้ `asyncio-throttle`

### yfinance wrapper

**ของที่ต้องดึง:**
```python
t = yf.Ticker(ticker)
t.info                      # fundamentals (dict) — cache 12h
t.history(period="2y")      # price OHLCV
t.income_stmt / balance_sheet / cashflow
t.insider_transactions      # ownership — cache 24h
t.institutional_holders
t.calendar                  # next earnings — cache 6h
```

**ข้อควรระวัง:** yfinance คืน `DataFrame` บางครั้ง คืน `dict` บางครั้ง คืน `None` → ต้อง defensive มากๆ

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
        asyncio.to_thread(fetch_market_data, ticker),  # yfinance sync
        asyncio.to_thread(fetch_macro_data),
    )
    return {"sec": sec, "market": market, "macro": macro}
```

**Lesson:** yfinance ไม่มี async → ห่อด้วย `asyncio.to_thread` ให้ไม่ block event loop

**Checkpoint:** `python -m stock_analysis.data.fetch AAPL` → dump JSON เห็นข้อมูล 3 แหล่งครบ

---

## 5) Screening

### เป้าหมาย: เลือกหุ้น US ที่ขึ้น/ลงแรงสุดประจำสัปดาห์

### `screening/us_universe.py`

ดึง universe จาก **NASDAQ Trader Symbol Directory** (เร็ว, ฟรี, update รายวัน):
- `nasdaqlisted.txt` — NASDAQ
- `otherlisted.txt` — NYSE, AMEX

~5,700 ตัว หลัง filter ETF, warrant, test issue ออก

### `screening/screener.py`

```python
def run_screener(lookback_days=5, top_n=3):
    universe = get_us_tickers()                 # ~5700 tickers
    # yfinance batch download (400 ticker/chunk)
    data = yf.download(universe, period=f"{lookback_days+5}d", group_by="ticker")
    # compute % change + filter liquidity + rank
    rows = []
    for t in universe:
        df = data[t]
        if df.empty: continue
        if df["Close"].mean() < 5: continue      # penny stocks out
        pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        adv = (df["Close"] * df["Volume"]).mean()
        if adv < 5e6: continue                   # low volume out
        rows.append({"ticker": t, "pct_change": pct, ...})

    gainers = sorted(rows, key=lambda r: -r["pct_change"])[:top_n]
    losers  = sorted(rows, key=lambda r: +r["pct_change"])[:top_n]
    return [{"mover_type":"gainer", **g} for g in gainers] + \
           [{"mover_type":"loser",  **l} for l in losers]
```

**Lesson:** batch download > loop per ticker. ~5,700 tickers ใช้ 2-3 นาที

**Checkpoint:** `python -m stock_analysis.screening.screener` → print รายชื่อ top movers

---

## 6) Calc engine

### Industry classifier — why?

**Insight:** DCF-FCFF ไม่เหมาะกับทุกอุตสาหกรรม:
- ธนาคาร → Residual Income Model (BVPS + ROE-Ke)
- REIT → AFFO-based
- Energy/Mining → Normalized DCF (cyclical)
- Utility → DDM (dividend-focused)

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
    "BRK-B": ValuationModel.RESIDUAL_INCOME,  # Berkshire unique
    "O": ValuationModel.AFFO_REIT,
}

def classify_ticker(ticker, sector="", sub_industry="") -> ValuationModel:
    # 1. ticker override → 2. sub-industry → 3. sector → 4. fallback
```

### DCF-FCFF (core model)

**สิ่งที่ต้อง compute:**
```
WACC = E/V * Ke + D/V * Kd * (1-t)
Ke   = Rf + β * ERP                              (CAPM)
FCFF = EBIT*(1-t) + D&A - CapEx - ΔWC
Year 1-5: project FCFF by growth rate
Terminal: FCFF_5 * (1+g) / (WACC-g)
EV       = Σ FCFF_t / (1+WACC)^t + TV/(1+WACC)^5
Equity   = EV - Debt + Cash
Fair Value/share = Equity / Shares Outstanding
```

**ของจริงใน code (ย่อ):**
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

**Hard rule:** LLM **ห้ามยุ่ง** กับตัวเลข — ทุกค่า compute ใน Python เท่านั้น

### Comparables + Sensitivity

**Comparables** — เทียบ peer ใน GICS sub-industry เดียวกัน:
```python
# Pull EV/EBITDA, P/E, P/S จาก yfinance ของ 5-10 peer
# Compute median/mean → target multiple
# Apply กับ ticker ของเรา → implied fair value
```

**Sensitivity** — 2D grid:
```python
# WACC × Terminal Growth = 5×5 table
# แต่ละ cell = re-run DCF ด้วย param นั้น
# เห็น fair value ภายใต้ scenario ต่างๆ
```

### Ratios dashboard

**Profitability / Returns / Leverage / Efficiency / Multiples** — compute จาก income + balance + price ทั้งหมด (ดู `calc_engine/ratios.py`)

**Checkpoint:** `python -m stock_analysis.calc_engine.valuate AAPL` → print fair value, WACC, upside

---

## 7) LLM layer (Groq)

### Architecture rule: LLM = writer, not calculator

**6 prompts:**

| Prompt | Model | หน้าที่ |
|---|---|---|
| `thesis_generator.txt` | 70b | bull/base/bear + catalysts + rating |
| `risk_synthesizer.txt` | 70b | financial/operational/market/macro risks |
| `industry_commentary.txt` | 70b | competitive positioning |
| `meeting_synthesizer.txt` | 70b | DEF 14A + 8-K synthesis |
| `annual_report_summary.txt` | 70b | 10-K → segments, MD&A, strategic initiatives |
| `qc_checker.txt` | 8b-instant | sanity-check valuation numbers |

### `llm/groq_client.py` — essentials

**Core `call_llm()`** พร้อม throttle + retry:
```python
_RPM_LIMIT = 25                  # Groq free = 30, keep buffer
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
2. **JSON mode**: `response_format: {"type": "json_object"}` บังคับ output
3. **Slim payload**: ตัดข้อมูลที่ LLM ไม่ต้องใช้ก่อน send (`_slim_valuation_bundle`)
4. **Humanize filter (Jinja)**: regex จับ `$12,000,000,000` → `$12.0B` ในกรณี LLM ยังใส่เลขยาว

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
- Format large amounts as $XB, not full digits

Respond in this JSON format:
{{
  "bull_case": {{ "thesis": "...", "key_drivers": [...] }},
  ...
}}
```

**Lesson จริงจากการทำ:** LLM ชอบ hallucinate เลข revenue → ต้อง **slim payload ให้เหลือน้อยที่สุด** ก่อน send. ตัว QC model (8b) มี TPM แค่ 6000 → ตัดยิ่งต้องกว้าง

**Checkpoint:** `python -m stock_analysis.llm.analyze AAPL` → return JSON มี thesis/risks/industry/meetings/annual_summary/qc

---

## 8) Report generation

### Flow: analysis dict → HTML → PDF

```
analyze_ticker() returns dict
     ↓
build_context()  — flatten ให้ Jinja กิน
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

**Lesson:** base64-embed PNG ดีกว่า save file แยก — Jinja render → PDF เห็นรูปแน่

### `report/generator.py` essentials

```python
def build_context(analysis):
    # flatten nested valuation bundle
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

โครงสร้าง ~10 pages:
1. **Cover** — rating chip / 4-card summary / screening banner / earnings banner / KV
2. **Investment Thesis** — bull/base/bear blocks + catalysts (ใช้ `humanize_nums` filter)
3. **Business Overview** (10-K) — segment / MD&A / strategy (จาก annual_summary LLM)
4. **Financial Analysis** — 10y history table + revenue chart + quarterly + ratios
5. **Valuation** — branch ตาม `primary.model` (FCFF/DDM/RI/AFFO/Cyclical)
6. **Peers** — EV/EBITDA bar
7. **Meetings & Events** — DEF 14A + 8-K
8. **Ownership** — insider + institutional
9. **Risks** — severity/likelihood
10. **Appendix** — assumptions + SEC filings + QC

### `report.css` key details

```css
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }

/* finance detail — ตัวเลขเรียงตามหลักหน่วยเสมอ */
.fin-table td, .kv-table td, .data-table td {
    font-variant-numeric: tabular-nums;
}

@page { size: A4; margin: 0; }
.page { page-break-after: always; padding: 18mm 14mm; }
@media print {
    table, figure, .thesis-block { page-break-inside: avoid; }
}
```

**Lesson สำคัญ:**
- ใช้ **Playwright Chromium** ไม่ใช่ WeasyPrint เพราะ WeasyPrint ต้อง GTK บน Windows + ไม่ support flex/grid ดี
- Playwright sync API ห้ามเรียกใน event loop ตรงๆ → ใช้ `asyncio.to_thread(generate_pdf, analysis)`

**Checkpoint:** `python -m stock_analysis.run --tickers AAPL` → ได้ PDF ใน `output/`

---

## 9) Orchestration

### `run.py` — ร้อยทั้งหมด

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

**Checkpoint:** `python -m stock_analysis.run` → เลือกหุ้นเอง + PDF ออกมาครบ

---

## 10) MCP server (optional)

### ทำไมต้องมี?

ให้ Claude Desktop / IDE agent เรียก SEC EDGAR tools ตรง ไม่ต้องเปิด Python script

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

Register ใน Claude Desktop `config.json`:
```json
{"mcpServers": {"sec-edgar": {"command": "sec-mcp-server"}}}
```

---

## 11) Common pitfalls

### Pitfall 1: ticker format mismatch

**อาการ:** `BRK-B` → yfinance OK, SEC fail
**แก้:** ลอง 4 variants ใน `get_cik`:
```python
variants = {t, t.replace("-","."), t.replace(".","-"), t.replace("-","").replace(".","")}
```

### Pitfall 2: XBRL sparse data (small-cap / new IPO)

**อาการ:** `KeyError: 'value'` เพราะ `financials["total_debt"] = []`
**แก้:** defensive access
```python
total_debt = financials.get("total_debt") or []
debt_val = (total_debt[0].get("value") if total_debt else 0) or 0
```

### Pitfall 3: LLM 413 Payload Too Large

**อาการ:** QC LLM (8b, TPM 6000) ได้ bundle ที่มี institutional holders 50 rows → boom
**แก้:** slim bundle ก่อน send:
```python
b.pop("ownership", None)
b.pop("earnings_calendar", None)
b.pop("ratios", None)
```

### Pitfall 4: Groq 429 rate limit

**อาการ:** ยิง 6 calls/ticker × 6 tickers = 36 calls ติดกัน → rate limit
**แก้:** sliding-window throttle (ข้างบน) + honor `Retry-After`

### Pitfall 5: yfinance cache stale

**อาการ:** หุ้นขึ้น 10% แล้วแต่ราคายังเก่า
**แก้:** TTL 12h สำหรับ info, 6h สำหรับ calendar (ไม่ใช่ทุก field ต้อง fresh)

### Pitfall 6: Playwright ใน asyncio

**อาการ:** `RuntimeError: Cannot nest sync_playwright() inside asyncio`
**แก้:** `await asyncio.to_thread(generate_pdf, analysis)`

### Pitfall 7: SEC ไม่ให้ User-Agent = block

**อาการ:** 403 Forbidden
**แก้:** `headers={"User-Agent": "MyApp you@example.com"}` — ต้องดูมือไม่ใช่ bot

### Pitfall 8: yfinance info สลับ type

**อาการ:** วันนี้ `.calendar` คืน dict, พรุ่งนี้คืน DataFrame
**แก้:** defensive wrapper:
```python
if isinstance(cal, dict): ...
elif hasattr(cal, "columns"): ...
else: return {}
```

---

## 12) Next steps (ฝึกต่อ)

เมื่อถึงจุดปัจจุบันแล้ว ลองสร้างต่อ:

### Short-term
- [ ] **Unit tests** — pytest ของ calc_engine (valuation math)
- [ ] **Backtest** — บันทึก rating ทุก week + เทียบ price +3m, +6m
- [ ] **ESG section** — Yahoo sustainability scores
- [ ] **Email delivery** — ส่ง PDF ให้ตัวเองทุกคืนศุกร์

### Mid-term
- [ ] **Streamlit dashboard** — interactive view ของ pipeline
- [ ] **GitHub Actions schedule** — weekly run + commit report
- [ ] **Cost tracking** — log Groq tokens ต่อรัน
- [ ] **Ticker watchlist** — DB table แยก user-pinned

### Long-term
- [ ] **ML earnings model** — predict EPS surprise
- [ ] **Alternative data** — Reddit / news sentiment
- [ ] **Portfolio view** — optimize allocation across ratings
- [ ] **Multi-LLM** — Claude/GPT fallback เมื่อ Groq ล่ม

---

## สรุป — learning path ที่ได้

1. **Python async** — ไม่ใช่แค่ `asyncio.sleep` แต่ async HTTP + mixed sync bridging
2. **API hardening** — cache + retry + backoff + stale-fallback
3. **Finance domain** — 5 valuation models คนละ assumption
4. **LLM engineering** — prompt design + anti-hallucination + rate limit
5. **PDF pipeline** — เลือก tool ให้ถูก (Playwright > WeasyPrint on Windows)
6. **Orchestration** — error handling per-ticker, don't fail the batch
7. **Observability** — structlog JSON, ทุก number traceable


Happy building 🚀
