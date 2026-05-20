# StocksKE — Full System Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Flow](#3-data-flow)
4. [Module Reference](#4-module-reference)
   - 4.1 [importer.py](#41-importerpy--price-data-ingestion)
   - 4.2 [nse_predictor/collector.py](#42-nse_predictorcollectorpy--news-collection)
   - 4.3 [nse_predictor/extractor.py](#43-nse_predictorextractorpy--llm-analysis)
   - 4.4 [nse_predictor/validator.py](#44-nse_predictorvalidatorpy--prediction-quality-gate)
   - 4.5 [nse_predictor/aligner.py](#45-nse_predictoralignerpy--ground-truth-labelling)
   - 4.6 [nse_predictor/pipeline.py](#46-nse_predictorpipelinepy--orchestrator)
   - 4.7 [nse_predictor/backtest.py](#47-nse_predictorbacktestpy--accuracy-evaluation)
   - 4.8 [nse_predictor/companies.py](#48-nse_predictorcompaniespy--company-registry)
   - 4.9 [nse_predictor/config.py](#49-nse_predictorconfigpy--runtime-configuration)
   - 4.10 [StocksKE_architecture/news_ingestion.py](#410-stockske_architecturenews_ingestionpy--sqlite-news-store)
   - 4.11 [StocksKE_architecture/app/main.py](#411-stockske_architectureappmainpy--fastapi-server)
   - 4.12 [StocksKE_architecture/app/ingestion/news_fetcher.py](#412-appingestionsnews_fetcherpy)
   - 4.13 [StocksKE_architecture/app/ingestion/price_scraper.py](#413-appingestionprice_scraperpy)
   - 4.14 [StocksKE_architecture/app/processing/sentiment_analyzer.py](#414-appprocessingsentiment_analyzerpy)
   - 4.15 [StocksKE_architecture/app/prediction/features.py](#415-apppredictionfeaturespy)
   - 4.16 [StocksKE_architecture/app/prediction/train.py](#416-apppredictiontrainpy)
   - 4.17 [StocksKE_architecture/app/prediction/predict.py](#417-apppredictionpredictpy)
   - 4.18 [StocksKE_architecture/config/settings.py](#418-configsettingspy)
5. [Test Suite](#5-test-suite)
   - 5.1 [test_validator.py](#51-test_validatorpy)
   - 5.2 [test_aligner.py](#52-test_alignerpy)
   - 5.3 [test_extractor.py](#53-test_extractorpy)
   - 5.4 [test_importer.py](#54-test_importerpy)
6. [Configuration Reference](#6-configuration-reference)
7. [How to Run](#7-how-to-run)

---

## 1. Project Overview

StocksKE is a two-layer system for predicting price movements on the **Nairobi Securities Exchange (NSE)**:

| Layer | Location | Purpose |
|---|---|---|
| **Research pipeline** | `nse_predictor/` | Collects news, runs LLM analysis, aligns predictions against real prices, measures accuracy |
| **Production system** | `StocksKE_architecture/` | Dockerised FastAPI + Celery service: ingests prices & news at scale, trains XGBoost models, sends Telegram alerts |
| **Price bridge** | `importer.py` | Downloads raw price lists from Innova, compiles per-security XLSX files, exports a unified CSV for both layers |

---

## 2. System Architecture

### 2.1 — Full System

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          StocksKE System                                 │
│                                                                          │
│  ┌─────────────────────────┐        ┌──────────────────────────────────┐ │
│  │   PRICE DATA LAYER      │        │   NEWS DATA LAYER                │ │
│  │                         │        │                                  │ │
│  │  Innova Price Lists      │        │  newsapi.org/v2/everything       │ │
│  │  (XLS files via HTTP)   │        │  Business Daily (scraper)        │ │
│  │         │               │        │         │                        │ │
│  │   importer.py           │        │   collector.py                   │ │
│  │   ├── download          │        │   ├── fetch_news_api()           │ │
│  │   ├── compile           │        │   └── scrape_business_daily()    │ │
│  │   └── build_prices_csv  │        │         │                        │ │
│  │         │               │        │   extractor.py (LLM)             │ │
│  │   prices.csv            │        │         │                        │ │
│  └──────────┬──────────────┘        │   validator.py                   │ │
│             │                       │         │                        │ │
│             └──────────────────────►│   aligner.py                     │ │
│                                     │         │                        │ │
│                                     │   labeled dataset (.jsonl)       │ │
│                                     │         │                        │ │
│                                     │   backtest.py                    │ │
│                                     │         │                        │ │
│                                     │   Accuracy Report                │ │
│                                     └──────────────────────────────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │               PRODUCTION LAYER  (StocksKE_architecture/)          │  │
│  │                                                                   │  │
│  │  NewsFetcher ──► Redis queue ──► SentimentAnalyzer (LLM)         │  │
│  │  PriceScraper ──► TimescaleDB                                     │  │
│  │  features.py ──► ModelTrainer (XGBoost) ──► Predictor            │  │
│  │  Predictor ──► Alert ──► Telegram / Email                        │  │
│  │  FastAPI (/feedback, /health) ──► PostgreSQL                     │  │
│  │  Prometheus (/metrics) ──► Grafana                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 — Research Pipeline (nse_predictor/)

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     pipeline.py (orchestrator)               │
  │                                                              │
  │  Step 0          Step 1        Step 2       Step 3  Step 4  │
  │                                                              │
  │  importer    ►  collector  ►  extractor  ►  validator  ►    │
  │  prices.csv     news API      LLM calls     hallucination    │
  │                 BD scraper    JSON parse     filter          │
  │                                                   │          │
  │                                             aligner          │
  │                                             (labeled.jsonl)  │
  │                                                   │          │
  │                                            backtest.py       │
  └──────────────────────────────────────────────────────────────┘
```

### 2.3 — Production System (StocksKE_architecture/)

```
  ┌──────────────────────────────────────────────────────────────┐
  │  External Sources           Celery Workers                   │
  │                                                              │
  │  newsapi.org  ──────────►  NewsFetcher                      │
  │                               │                             │
  │  NSE website  ──────────►  PriceScraper ──► TimescaleDB     │
  │                               │                             │
  │                          Redis Queue                        │
  │                               │                             │
  │                          SentimentAnalyzer (OpenAI)         │
  │                               │                             │
  │                          PostgreSQL (SentimentScore)        │
  │                               │                             │
  │                          features.py                        │
  │                         (ret_1d, RSI-14, score_trend_3d)   │
  │                               │                             │
  │                    ┌──────────┴──────────┐                  │
  │                    │                     │                  │
  │               ModelTrainer          Predictor               │
  │              (XGBoost, weekly)     (hourly)                 │
  │                                         │                   │
  │                                    Alert → Telegram         │
  │                                                             │
  │  FastAPI (/feedback, /health, /metrics)                     │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### Price Data

```
  Innova website (HTTPS)
       │  GET /files/Price%20List%20-DD-Month-YYYY.xls
       ▼
  importer.download_price_lists()
       │  saves: downloaded_price_lists/DD-Month-YYYY.xls
       ▼
  importer.compile_securities()
       │  reads each .xls, extracts "Ord" rows
       │  saves: compiled_securities/<Security Name>.xlsx
       ▼
  importer.build_prices_csv()
       │  strips "Ord..." suffix → matches NAME_MAP → ticker
       │  normalises dates → YYYY-MM-DD
       ▼
  prices.csv   (Stock Code | Date | Day's Final Price)
       │
       └──────────────────────► aligner.load_prices()
```

### News → Prediction Data

```
  newsapi.org  +  businessdailyafrica.com
       │
       ▼
  collector.collect_all()
       │  saves: nse_dataset/news/news_YYYY-MM-DD_to_YYYY-MM-DD.jsonl
       ▼
  extractor.extract_all()
       │  LLM → {directly_affected, indirectly_affected, direction, confidence}
       │  saves: nse_dataset/extractions/extractions_TIMESTAMP.jsonl
       ▼
  validator.filter_predictions()
       │  drops unknown tickers, invalid directions, confidence < 0
       ▼
  aligner.align()
       │  joins predictions → prices.csv → pct_change → price_label
       │  sets correct=True/False
       │  saves: nse_dataset/labeled/labeled_TIMESTAMP.jsonl
       ▼
  backtest.py
       │  loads labeled/*.jsonl → accuracy, precision, recall, F1
       ▼
  Accuracy Report (stdout or JSON)
```

---

## 4. Module Reference

---

### 4.1 `importer.py` — Price Data Ingestion

**Location:** `stocks analyse/importer.py`

**Purpose:** Three-stage price pipeline: (1) download raw XLS files from Innova's website, (2) compile them into one XLSX per security, (3) export a single unified `prices.csv` that the prediction pipeline can consume.

#### Architecture

```
  CLI (argparse)
    ├── download   → download_price_lists()
    ├── compile    → compile_securities()
    ├── build-csv  → build_prices_csv()
    └── all        → all three in sequence

  Internal helpers
    ├── _download_one()      retry loop for a single URL
    ├── _parse_row()         extract OHLCV from one DataFrame row
    ├── _read_sheet()        try Sheet1 then index-0 fallback
    ├── _strip_ord_suffix()  "KCB Group Plc Ord 1_00" → "KCB Group Plc"
    ├── _normalise_date()    "01-June-2020" → "2020-06-01"
    └── _load_ticker_map()   {name.lower(): ticker} from companies.py
```

#### Key Functions

**`download_price_lists(start_date, end_date, output_folder, base_url)`**

Downloads one XLS file per calendar day. Skips files already on disk.

```python
saved = download_price_lists(
    datetime(2024, 1, 1), datetime(2024, 12, 31),
    output_folder="downloaded_price_lists"
)
```

**`_download_one(session, url, dest)`**

Retries up to `MAX_RETRIES` (3) times with exponential backoff. Returns `True` on success, `False` on 404 or exhausted retries.

```python
# Returns True when HTTP 200 and file written, False otherwise
ok = _download_one(session, "https://innova.co.ke/files/Price%20List%20-01-June-2024.xls", dest)
```

**`compile_securities(input_folder, output_folder)`**

Reads every `*.xls*` in `input_folder`. For each row that contains "Ord" in column 0, detects the optional ISIN column and extracts: Total Shares, Market Cap, VWAP, High, Low.

```python
# Returns {security_name: Path("compiled_securities/KCB Group Plc Ord 1_00.xlsx")}
written = compile_securities("downloaded_price_lists", "compiled_securities")
```

**`_parse_row(row, date)`**

Handles the two possible column layouts (with ISIN and without) by checking whether column 1 starts with `"KE"` and is 12 characters long.

```python
# With ISIN: cols[1]=ISIN, cols[2]=shares, cols[3]=mktcap, cols[4]=VWAP ...
# Without:   cols[1]=shares, cols[2]=mktcap, cols[3]=VWAP ...
has_isin = col1.startswith("KE") and len(col1) == 12
```

**`build_prices_csv(compiled_dir, output_csv, ticker_map)`**

The integration bridge. Reads compiled XLSX files, resolves security names to NSE tickers, normalises dates, and writes the CSV that `aligner.py` expects.

```python
ticker_map = {"kcb group plc": "KCB", "safaricom plc": "SCOM", ...}
out = build_prices_csv("compiled_securities", "prices.csv", ticker_map)
```

Output CSV format:
```
Stock Code,Date,Day's Final Price
KCB,2024-01-02,45.50
SCOM,2024-01-02,18.75
```

**`_strip_ord_suffix(name)`**

Removes the "Ord X.XX" suffix that appears in both file names and raw security names.

```python
_strip_ord_suffix("KCB Group Plc Ord 1_00")   # → "KCB Group Plc"
_strip_ord_suffix("Equity Group Holdings Plc Ord 0.50")  # → "Equity Group Holdings Plc"
```

**`_normalise_date(raw)`**

Accepts any of the four date formats the system produces and returns a canonical `YYYY-MM-DD` string.

```python
_normalise_date("01-June-2020")   # → "2020-06-01"
_normalise_date("2024-03-15")     # → "2024-03-15"
```

**`_load_ticker_map(companies_json)`**

Auto-discovers `nse_predictor/companies.py` at runtime. Falls back to a JSON file if provided.

```python
# Inserts nse_predictor/ into sys.path, then imports NAME_MAP
ticker_map = _load_ticker_map(None)   # auto-discover
ticker_map = _load_ticker_map("my_map.json")  # explicit file
```

#### CLI Reference

```
python importer.py download  --start 2024-01-01 --end 2024-12-31
python importer.py compile   --in downloaded_price_lists --out compiled_securities
python importer.py build-csv --in compiled_securities   --out prices.csv
python importer.py all       --start 2024-01-01 --end 2024-12-31 --prices-csv prices.csv
```

---

### 4.2 `nse_predictor/collector.py` — News Collection

**Purpose:** Pulls raw news articles from two sources — the **News API** (per-company queries) and the **Business Daily Africa** website (scraper) — deduplicates by URL, and saves a JSONL file.

#### Architecture

```
  collect_all(days_back)
    │
    ├── fetch_news_api(name, ticker, from_date, to_date)
    │      for each company in NSE_COMPANIES
    │      rate-limit aware: 429 → sleep 60s → retry
    │      returns [{ticker, company, title, description, content, source,
    │               published_at, url}]
    │
    └── scrape_business_daily(max_pages)
           robots.txt check first
           BeautifulSoup anchor extraction
           returns [{ticker:"UNKNOWN", ...}]

  Output: nse_dataset/news/news_YYYY-MM-DD_to_YYYY-MM-DD.jsonl
```

#### Key Functions

**`fetch_news_api(company_name, ticker, from_date, to_date)`**

Queries `newsapi.org/v2/everything` with `"{company_name}" OR "{ticker}"`. Handles 429 rate-limits and network errors with up to 3 retries.

```python
# Returns list of normalised article dicts; empty list on failure
items = fetch_news_api("KCB Group Plc", "KCB", "2024-01-01", "2024-01-31")
```

**`scrape_business_daily(max_pages)`**

Reads `robots.txt` before scraping. Extracts `<a href>` links and nearby `<time>` tags from the companies section.

```python
# All articles get ticker="UNKNOWN"; extractor resolves the ticker later
articles = scrape_business_daily(max_pages=5)
```

**`collect_all(days_back)`**

Iterates all companies in `NSE_COMPANIES`, deduplicates by URL, appends Business Daily results, and saves the combined JSONL.

```python
articles = collect_all(days_back=30)
# → nse_dataset/news/news_2024-01-01_to_2024-01-31.jsonl
```

---

### 4.3 `nse_predictor/extractor.py` — LLM Analysis

**Purpose:** Sends each article to an OpenAI-compatible LLM and parses the structured JSON response describing which NSE companies are affected, in which direction, and with what confidence.

#### Architecture

```
  extract_all(articles)
    │
    └── call_llm(article)  for each article (min length 50 chars)
           │
           ├── build_user_message(article)
           │      → JSON: {article_text, article_date, source}
           │
           ├── POST /chat/completions
           │      model=MODEL_NAME, temp=0.0
           │      system=SYSTEM_PROMPT (contains company list + rules)
           │
           ├── _strip_fences(content)
           │      removes ```json ... ``` wrappers
           │
           └── json.loads() → prediction dict

  Output: nse_dataset/extractions/extractions_TIMESTAMP.jsonl
  Each line: {"article": {...}, "prediction": {...}}
```

#### LLM Output Schema

```json
{
  "article_summary": "KCB reports 20% profit increase.",
  "event_type": "earnings",
  "primary_sector": "Banking",
  "directly_affected": [
    {"ticker": "KCB", "direction": "UP", "confidence": 0.91,
     "impact_type": "direct", "reasoning": "..."}
  ],
  "indirectly_affected": [...],
  "macro_flags": {"currency_risk": false, "interest_rate_sensitive": false},
  "data_quality": {"article_date": "2024-03-01", "source_credibility": "high"}
}
```

#### Key Functions

**`build_user_message(article)`**

Picks the best text from `content → description → title`, truncates to 2000 chars, wraps in JSON.

```python
msg = build_user_message({"title": "KCB posts profits", "content": "Full text..."})
# → '{"article_text": "Full text...", "article_date": "...", "source": "..."}'
```

**`_strip_fences(s)`**

Removes accidental markdown code fences that LLMs occasionally add around JSON.

```python
_strip_fences("```json\n{\"key\": 1}\n```")   # → '{"key": 1}'
```

**`call_llm(article, retries=3)`**

Posts to `OPENAI_BASE_URL/chat/completions`. On 429 sleeps 60 s; on other errors uses `2^attempt` backoff. Falls back to regex JSON extraction if the response has prose around the JSON.

```python
pred = call_llm(article)     # returns dict or None
```

---

### 4.4 `nse_predictor/validator.py` — Prediction Quality Gate

**Purpose:** Enforces strict rules on every LLM prediction to catch hallucinations — unknown tickers, invalid directions, bad confidence values — before they reach the alignment step.

#### Architecture

```
  filter_predictions(predictions)
    │
    └── validate(prediction)  for each prediction
           │
           ├── event_type ∈ VALID_EVENT_TYPES?
           ├── all tickers ∈ VALID_TICKERS?
           ├── all directions ∈ {"UP","DOWN","NEUTRAL"}?
           ├── all impact_types ∈ VALID_IMPACT_TYPES?
           ├── all confidence ∈ [0.0, 1.0]?
           └── no ticker in both directly + indirectly affected?

  Returns: (valid_list, invalid_list_with_errors)
```

#### Validation Rules

| Rule | What it catches |
|---|---|
| Ticker not in `VALID_TICKERS` | Hallucinated companies (e.g. `"FAKE"`, `"AAPL"`) |
| Direction not in `{UP, DOWN, NEUTRAL}` | Typos like `"SIDEWAYS"`, `"RISE"` |
| `impact_type` not in allowed set | LLM inventing impact categories |
| Confidence outside `[0.0, 1.0]` | Out-of-range floats |
| Ticker in both groups | Contradictory prediction |
| `event_type` not in allowed set | Unrecognised event categories |

#### Key Functions

**`validate(prediction)`**

Returns a list of error strings. An empty list means the prediction passed all checks.

```python
errors = validate(pred)
if errors:
    print("Rejected:", errors)   # ["invalid ticker: FAKE", ...]
```

**`filter_predictions(predictions)`**

Splits a list into `(valid, invalid)`. Invalid items are wrapped with their errors for logging.

```python
valid, invalid = filter_predictions(all_predictions)
logger.info("Valid: %d  Rejected: %d", len(valid), len(invalid))
```

---

### 4.5 `nse_predictor/aligner.py` — Ground-Truth Labelling

**Purpose:** Joins validated predictions against the realised price CSV. For every predicted ticker it looks up the stock price on the prediction date and `lookahead_days` trading days later, computes the % change, assigns a ground-truth label (`UP`/`DOWN`/`NEUTRAL`), and marks whether the model was correct.

#### Architecture

```
  align(predictions, prices_csv, lookahead_days, threshold)
    │
    ├── load_prices(csv_path)
    │      reads "Stock Code", "Date", "Day's Final Price"
    │      → {(ticker, "YYYY-MM-DD"): float}
    │
    └── for each prediction → for each ticker in affected groups:
           │
           ├── get_price_change(prices, ticker, article_date, lookahead_days)
           │      searches ±5 calendar days for nearest trading day
           │      returns (price_t0, price_tn, pct_change)
           │
           ├── generate_label(pct_change, threshold)
           │      pct ≥ +threshold → "UP"
           │      pct ≤ -threshold → "DOWN"
           │      else             → "NEUTRAL"
           │
           └── correct = (predicted_direction == price_label)

  Output: nse_dataset/labeled/labeled_TIMESTAMP.jsonl
```

#### Key Functions

**`load_prices(csv_path)`**

Reads the unified CSV from `importer.build_prices_csv()`. Strips commas from formatted numbers (`"1,234.50"` → `1234.50`).

```python
prices = load_prices("prices.csv")
price = prices[("KCB", "2024-03-01")]   # → 45.0
```

**`get_price_change(prices, ticker, from_date, lookahead_days)`**

Searches up to 5 calendar days forward from both the start date and the lookahead date to handle weekends and public holidays.

```python
t0, tn, pct = get_price_change(prices, "KCB", "2024-03-01", lookahead_days=3)
# t0=45.0, tn=47.25, pct=5.0
```

**`generate_label(pct_change, threshold=1.5)`**

The `threshold` (default 1.5 %) filters out noise — small moves are labelled NEUTRAL.

```python
generate_label(2.0)    # "UP"
generate_label(-0.5)   # "NEUTRAL"
generate_label(-3.0)   # "DOWN"
```

**`align(predictions_jsonl, prices_csv, lookahead_days, threshold)`**

Accepts either a JSONL file path or a Python list (both wrapped `{article, prediction}` and raw prediction dicts).

```python
labeled = align(valid_preds, "prices.csv", lookahead_days=3, threshold=1.5)
# each item: {ticker, predicted_direction, price_label, correct, pct_change, ...}
```

---

### 4.6 `nse_predictor/pipeline.py` — Orchestrator

**Purpose:** Ties all modules into a single runnable pipeline. Handles both full runs (download prices + fetch news + analyse + label) and incremental runs (skip collection, use existing data).

#### Architecture

```
  run(days_back, skip_collection, import_from, prices_csv_override)
    │
    ├── Step 0: resolve prices CSV
    │      if import_from or COMPILED_DIR set:
    │          _ensure_prices_csv() → build_prices_csv() from importer.py
    │      else use PRICE_CSV_PATH from config
    │
    ├── Step 1: collect news
    │      skip_collection=False → collector.collect_all(days_back)
    │      skip_collection=True  → load latest news/*.jsonl
    │
    ├── Step 2: extract predictions
    │      extractor.extract_all(articles)
    │
    ├── Step 3: validate
    │      validator.filter_predictions(predictions)
    │
    └── Step 4: align
           aligner.align(valid, prices_csv, lookahead_days, threshold)
           logs: accuracy on labeled set
```

#### CLI Reference

```
# Full run — build prices CSV from importer output, collect 30 days of news
python pipeline.py --import-from compiled_securities --days 30

# Use an already-built prices CSV, skip news re-download
python pipeline.py --prices-csv prices.csv --skip-collect

# Point at compiled dir via env var
export COMPILED_DIR=compiled_securities
python pipeline.py
```

---

### 4.7 `nse_predictor/backtest.py` — Accuracy Evaluation

**Purpose:** Loads one or more labeled JSONL files produced by `aligner.align()` and computes a full accuracy report including per-class precision/recall/F1 and breakdowns by confidence band, impact type, and ticker.

#### Architecture

```
  main()
    │
    ├── --demo:         _generate_synthetic_rows(500) → rows
    ├── --labeled PATH: load_labeled_rows(path)       → rows
    └── (default):      load_all_labeled(output_dir)  → rows (all files)
           │
           ▼
    compute_metrics(rows)
    │
    ├── overall accuracy
    ├── per-class TP/FP/FN → precision, recall, F1
    ├── by confidence band  (<0.50 / 0.50-0.70 / >0.70)
    ├── by impact_type
    └── by ticker (top 15 by volume)
           │
           ▼
    print_report(metrics)   or   --json → raw JSON
```

#### Metrics Produced

| Metric | Description |
|---|---|
| `overall_accuracy` | Fraction of predictions where direction matched ground truth |
| `precision[class]` | Of all times the model said UP/DOWN/NEUTRAL, how often it was right |
| `recall[class]` | Of all actual UP/DOWN/NEUTRAL moves, how many the model caught |
| `f1[class]` | Harmonic mean of precision and recall |
| `by_confidence_band` | Whether higher-confidence predictions are more accurate |
| `by_impact_type` | Accuracy split by `direct` vs `competitor` vs `sector_spillover` etc |
| `by_ticker` | Per-company accuracy (top 15 by prediction volume) |

#### CLI Reference

```
python backtest.py --demo                          # synthetic data demo
python backtest.py                                 # all labeled/*.jsonl files
python backtest.py --labeled nse_dataset/labeled/labeled_20240301T120000.jsonl
python backtest.py --json                          # output raw JSON metrics
```

---

### 4.8 `nse_predictor/companies.py` — Company Registry

**Purpose:** Single source of truth for all NSE-listed equities and their relationships. Every other module that needs a ticker or company name imports from here.

#### Contents

| Export | Type | Description |
|---|---|---|
| `NSE_COMPANIES` | `list[dict]` | 46 companies with `ticker`, `name`, `sector` |
| `COMPETITOR_RELATIONSHIPS` | `list[tuple]` | Known competitor pairs (used in extractor prompt) |
| `VALID_TICKERS` | `set[str]` | All valid ticker symbols (used by validator) |
| `SECTOR_MAP` | `dict[str, str]` | `{ticker: sector}` |
| `NAME_MAP` | `dict[str, str]` | `{ticker: full_name}` |
| `get_sector_peers(ticker)` | function | All NSE tickers in same sector |
| `get_competitors(ticker)` | function | NSE-listed competitor tickers |

```python
from companies import VALID_TICKERS, NAME_MAP, get_sector_peers
peers = get_sector_peers("KCB")   # ["ABSA", "EQTY", "NCBA", ...]
```

---

### 4.9 `nse_predictor/config.py` — Runtime Configuration

**Purpose:** All tunable parameters loaded from environment variables (or `.env` via `python-dotenv`). Import from here; never hardcode values elsewhere.

#### Full Parameter Table

| Variable | Default | Description |
|---|---|---|
| `NEWS_API_KEY` | `""` | newsapi.org API key |
| `OPENAI_API_KEY` | `""` | LLM API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Swap to a local LLM endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model identifier |
| `TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |
| `MAX_TOKENS` | `1500` | Max LLM response tokens |
| `OUTPUT_DIR` | `nse_dataset` | Root directory for all outputs |
| `PRICE_CSV_PATH` | `""` | Explicit prices CSV path (optional) |
| `COMPILED_DIR` | `""` | Path to importer's compiled_securities/ |
| `CONFIDENCE_THRESHOLD` | `0.3` | Minimum confidence to include in output |
| `PRICE_CHANGE_THRESHOLD` | `1.5` | % move to classify as UP or DOWN |
| `LOOKAHEAD_DAYS` | `3` | Days after article to check price |

```bash
# .env example
NEWS_API_KEY=abc123
OPENAI_API_KEY=sk-...
COMPILED_DIR=C:/data/compiled_securities
PRICE_CHANGE_THRESHOLD=2.0
```

---

### 4.10 `StocksKE_architecture/news_ingestion.py` — SQLite News Store

**Purpose:** Standalone script for the architecture folder. Loads company names from a Knowledge Graph JSON, queries the News API in batches of 15, and stores raw articles in a local SQLite database with deduplication.

#### Architecture

```
  main()
    │
    ├── setup_database()      CREATE TABLE IF NOT EXISTS news_articles
    ├── load_knowledge_graph() loads companies[] from JSON
    └── fetch_and_store_news(conn, companies)
           │
           ├── chunks companies into groups of 15
           ├── builds query: ("Safaricom" OR "KCB") AND Kenya
           ├── _fetch_with_retry(params)
           │      3-attempt retry, 60s sleep on 429, exponential backoff
           └── INSERT OR IGNORE (deduplication by URL UNIQUE constraint)
                  matched_tickers = which company names appeared in article text
```

#### Key Functions

**`setup_database(db_path)`**

Creates the `news_articles` table with a `UNIQUE` constraint on `url` for automatic deduplication.

```python
conn = setup_database("nse_news_database.db")
```

**`fetch_and_store_news(conn, companies)`**

Returns the number of new articles inserted. Handles batch chunking, ticker matching, and commit/rollback.

```python
inserted = fetch_and_store_news(conn, companies_list)
```

**`_fetch_with_retry(params)`**

Wraps the News API call with 3-attempt retry and 429 backoff, using `X-Api-Key` header authentication.

```python
data = _fetch_with_retry({"q": "KCB AND Kenya", "from": "2024-03-01", ...})
```

---

### 4.11 `StocksKE_architecture/app/main.py` — FastAPI Server

**Purpose:** Provides two HTTP endpoints: `/feedback` for users to rate alert quality, and `/health` for liveness probes. Mounts Prometheus metrics at `/metrics`.

```
  GET  /health     → {"status": "ok"}
  POST /feedback   → records user's UP/DOWN/NEUTRAL rating on an alert
  GET  /metrics    → Prometheus exposition format (for Grafana)
```

**`POST /feedback`** — payload: `{alert_id: int, feedback_type: str}`

Looks up the Alert by ID, creates a `Feedback` record, and updates `alert.user_feedback`. Returns 404 if alert not found.

```python
# Request body
{"alert_id": 42, "feedback_type": "correct"}
# Response
{"status": "success", "message": "Feedback recorded."}
```

---

### 4.12 `app/ingestion/news_fetcher.py`

**Purpose:** Celery worker class that fetches NSE-related news from News API, deduplicates by MD5 content hash, persists to PostgreSQL, and pushes new articles into a Redis queue for the sentiment analyser.

#### Architecture

```
  NewsFetcher.fetch_recent_news()
    │
    ├── GET newsapi.org (q="NSE OR Kenya stock", pageSize=50)
    └── for each article:
           process_article(article)
           │
           ├── md5(title + content) → content_hash
           ├── skip if hash exists in DB (deduplication)
           ├── INSERT News record
           └── redis_client.lpush("news_queue", json)
```

**`process_article(article)`**

Deduplicates by `content_hash` rather than URL to handle URL parameter variations. Returns `True` if article was new and inserted.

```python
# Returns False if duplicate; True if newly inserted and queued
was_new = fetcher.process_article({"title": "KCB profits...", "url": "...", ...})
```

---

### 4.13 `app/ingestion/price_scraper.py`

**Purpose:** Celery worker class that scrapes end-of-day price data for all tickers in the database. Currently runs in mock mode (`self.mock = True`) which generates synthetic OHLCV data — replace `mock_fetch_price` with real scraping logic when a data source is confirmed.

```python
# To activate real scraping:
# 1. Set self.mock = False
# 2. Implement the URL fetch + BeautifulSoup parse in scrape_eod_data()
scraper.scrape_eod_data()   # populates Price table via db.merge()
```

---

### 4.14 `app/processing/sentiment_analyzer.py`

**Purpose:** Celery worker that drains the `news_queue` Redis list in batches, sends each article to the LLM via `LLMClient`, and stores `SentimentScore` records in PostgreSQL.

#### Architecture

```
  SentimentAnalyzer.process_queue(batch_size=50)
    │
    └── while queue not empty (up to batch_size):
           redis_client.rpop("news_queue")   → {news_id, title, content}
           LLMClient.analyze_sentiment()     → {ticker, sentiment_score, ...}
           Ticker lookup by symbol (ilike)
           INSERT SentimentScore(news_id, ticker_id, sentiment_score, raw_json)
```

**`process_queue(batch_size)`**

Processes up to `batch_size` items per call. Designed to be invoked repeatedly by a Celery Beat schedule. Individual item failures do not abort the batch.

```python
analyzer.process_queue(batch_size=50)
# Any error on one article is caught; others continue
```

---

### 4.15 `app/prediction/features.py`

**Purpose:** Builds the feature matrix for the XGBoost model. Joins price OHLCV data (from TimescaleDB) with daily-aggregated sentiment scores to produce a time-indexed DataFrame.

#### Features Produced

| Feature | Formula |
|---|---|
| `ret_1d` | `close.pct_change(1)` |
| `ret_5d` | `close.pct_change(5)` |
| `ret_20d` | `close.pct_change(20)` |
| `vol_change` | `volume.pct_change(1)` |
| `sma_20` | 20-day simple moving average of close |
| `rsi_14` | 14-period EMA-based RSI |
| `score` | Mean daily sentiment score (0 where no news) |
| `score_trend_3d` | 3-day rolling mean of `score` |
| `target` | `1` if next-day close > today's close, else `0` |

```python
df = build_features_for_ticker(db, ticker_id=1,
    start_date=datetime(2024,1,1), end_date=datetime(2024,12,31))
# df.columns = ["ret_1d","ret_5d","ret_20d","vol_change","sma_20",
#               "rsi_14","score","score_trend_3d","target"]
```

---

### 4.16 `app/prediction/train.py`

**Purpose:** Weekly Celery task that retrains one XGBoost classifier per ticker using the last 365 days of feature data. Saves each model via `ModelRegistry`.

#### Training Configuration

| Parameter | Value |
|---|---|
| Lookback | 365 days |
| Min rows | 50 (skip ticker if fewer) |
| Test split | 20% (chronological, `shuffle=False`) |
| Model | `XGBClassifier(n_estimators=100, max_depth=3, lr=0.05)` |
| Features | `FEATURE_COLS` (8 columns from `features.py`) |
| Target | Next-day direction (binary: 1=up, 0=down) |

```python
# Triggered by Celery Beat every Sunday
trainer = ModelTrainer(db)
trainer.train_all_models()
```

---

### 4.17 `app/prediction/predict.py`

**Purpose:** Hourly Celery task. Loads the saved XGBoost model for each ticker, builds the feature vector from the latest 60 days of data, and stores a `Prediction` record with the probability and direction.

```
  Predictor.run_predictions()
    │
    └── for each ticker:
           model = ModelRegistry.load_model(ticker.symbol)
           df    = build_features_for_ticker(last 60 days)
           X     = df.iloc[-1:][FEATURE_COLS]
           prob  = model.predict_proba(X)[0][1]   # P(up)
           direction = 1 if prob > 0.5 else 0
           INSERT Prediction(ticker_id, direction, probability, features_snapshot)
```

```python
predictor = Predictor(db)
predictor.run_predictions()   # fires hourly via Celery Beat
```

---

### 4.18 `StocksKE_architecture/config/settings.py`

**Purpose:** Pydantic `BaseSettings` class. Reads from `.env` automatically. All infrastructure URLs and API keys are centralised here.

| Setting | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL + TimescaleDB connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for news queue + Celery broker |
| `NEWS_API_KEY` | `""` | newsapi.org key |
| `OPENAI_API_KEY` | `""` | OpenAI/compatible LLM key |
| `TELEGRAM_BOT_TOKEN` | `""` | Telegram bot for alerts |
| `TELEGRAM_CHAT_ID` | `""` | Telegram channel/chat |
| `PREDICTION_THRESHOLD` | `0.7` | Minimum probability to fire an alert |

```python
from config.settings import settings
print(settings.DATABASE_URL)   # read from .env
```

---

## 5. Test Suite

All tests use **pytest**. Run from `nse_predictor/`:

```bash
cd nse_predictor
python -m pytest tests/ -v
# 49 passed in 0.51s
```

---

### 5.1 `test_validator.py`

**Location:** `nse_predictor/tests/test_validator.py`

Tests the `validate()` and `filter_predictions()` functions.

#### Test Classes

**`TestValidate`** — 10 tests

| Test | What it verifies |
|---|---|
| `test_valid_prediction_returns_no_errors` | A well-formed prediction produces zero errors |
| `test_hallucinated_ticker_is_caught` | Ticker `"FAKE"` raises an invalid-ticker error |
| `test_invalid_direction_is_caught` | `"SIDEWAYS"` is rejected |
| `test_invalid_impact_type_is_caught` | `"rumour"` is rejected |
| `test_invalid_event_type_is_caught` | `"gossip"` is rejected |
| `test_confidence_out_of_range_is_caught` | Confidence `1.5` is rejected |
| `test_confidence_zero_is_valid` | Confidence `0.0` is accepted |
| `test_ticker_in_both_groups_is_caught` | Same ticker in both affected lists is rejected |
| `test_non_dict_prediction_returns_error` | Passing a string raises `"not a dict"` error |
| `test_multiple_errors_returned` | Multiple simultaneous violations all reported |

**`TestFilterPredictions`** — 4 tests

| Test | What it verifies |
|---|---|
| `test_valid_passes_through` | Valid prediction ends up in `valid` list |
| `test_invalid_is_separated` | Invalid prediction ends up in `invalid` list |
| `test_empty_input` | Empty input returns `([], [])` |
| `test_all_invalid` | All bad predictions return empty `valid` list |

---

### 5.2 `test_aligner.py`

**Location:** `nse_predictor/tests/test_aligner.py`

Uses `tempfile` to create real CSV files on disk (no mocking).

#### Test Classes

**`TestLoadPrices`** — 6 tests

| Test | What it verifies |
|---|---|
| `test_loads_correct_structure` | `(ticker, date)` keys with float values |
| `test_skips_non_numeric_price` | `"N/A"` is skipped gracefully |
| `test_handles_comma_formatted_numbers` | `"1,234.50"` → `1234.50` |
| `test_returns_empty_for_missing_file` | Non-existent path returns `{}` |
| `test_returns_empty_for_empty_path` | Empty string returns `{}` |
| `test_skips_rows_with_missing_fields` | Rows without a ticker are skipped |

**`TestGetPriceChange`** — 6 tests

| Test | What it verifies |
|---|---|
| `test_returns_correct_pct_change` | `(47.25 - 45.0) / 45.0 * 100` |
| `test_returns_none_when_start_price_missing` | Unknown date returns `(None,None,None)` |
| `test_returns_partial_when_lookahead_price_missing` | Returns `(t0, None, None)` |
| `test_skips_weekend_to_find_next_trading_day` | Saturday → searches forward to Monday |
| `test_returns_none_for_invalid_date_format` | `"not-a-date"` returns `(None,None,None)` |
| `test_returns_none_for_unknown_ticker` | Unknown ticker returns `None` |

**`TestGenerateLabel`** — 6 tests

| Test | Condition | Expected |
|---|---|---|
| `test_above_threshold_is_up` | `pct=2.0, threshold=1.5` | `"UP"` |
| `test_exactly_threshold_is_up` | `pct=1.5, threshold=1.5` | `"UP"` |
| `test_below_negative_threshold_is_down` | `pct=-2.0` | `"DOWN"` |
| `test_exactly_negative_threshold_is_down` | `pct=-1.5` | `"DOWN"` |
| `test_within_threshold_is_neutral` | `pct=0.5, -1.4, 0.0` | `"NEUTRAL"` |
| `test_none_pct_change_is_neutral` | `pct=None` | `"NEUTRAL"` |

**`TestAlign`** — 5 tests

| Test | What it verifies |
|---|---|
| `test_correct_prediction_marked_correct` | KCB UP prediction → price rises → `correct=True` |
| `test_wrong_prediction_marked_incorrect` | KCB DOWN prediction → price rises → `correct=False` |
| `test_returns_empty_for_empty_predictions` | `[]` input → `[]` output |
| `test_missing_price_data_yields_neutral_label` | Date with no price data → `"NEUTRAL"` label |
| `test_wrapper_format_is_handled` | `{article, prediction}` wrapper is unwrapped |

---

### 5.3 `test_extractor.py`

**Location:** `nse_predictor/tests/test_extractor.py`

**`TestBuildUserMessage`** — 8 tests

| Test | What it verifies |
|---|---|
| `test_returns_valid_json` | Output is parseable JSON with required keys |
| `test_prefers_content_over_description` | `content` wins when both present |
| `test_falls_back_to_description_when_no_content` | Empty `content` → uses `description` |
| `test_falls_back_to_title_when_no_content_or_description` | `None` content+desc → uses `title` |
| `test_truncates_long_content_to_2000_chars` | 5000-char string is clipped |
| `test_uses_published_at_as_article_date` | `published_at` key populated |
| `test_falls_back_to_publishedAt_key` | Camel-case `publishedAt` also accepted |
| `test_empty_article_does_not_raise` | `{}` produces `{"article_text": "", ...}` |

**`TestStripFences`** — 4 tests

| Test | Input | Expected |
|---|---|---|
| `test_strips_json_fence` | ```` ```json\n{...}\n``` ```` | `{...}` |
| `test_strips_plain_fence` | ```` ```\n{...}\n``` ```` | `{...}` |
| `test_no_fence_unchanged` | `{...}` | `{...}` |
| `test_strips_whitespace` | `"   {...}   "` | `{...}` |

---

### 5.4 `test_importer.py`

**Location:** `tests/test_importer.py`

Uses `pytest`'s `tmp_path` fixture and `unittest.mock.patch` to avoid real HTTP calls.

**`TestParseRow`** — 5 tests

| Test | What it verifies |
|---|---|
| `test_skips_non_string_first_column` | `None` and integers are skipped |
| `test_skips_row_without_ord_in_name` | `"Safaricom Plc"` (no "Ord") → `None` |
| `test_parses_row_without_isin` | Correct column offsets for non-ISIN layout |
| `test_parses_row_with_isin` | ISIN in col 1 shifts all data columns right by 1 |
| `test_isin_detection_requires_ke_prefix_and_12_chars` | `"US..."` and short strings are not treated as ISIN |

**`TestCompileSecurities`** — 4 tests

| Test | What it verifies |
|---|---|
| `test_compiles_single_security` | One "Ord" row → one output file |
| `test_output_file_is_valid_excel` | Output XLSX has expected columns |
| `test_raises_for_missing_input_folder` | `FileNotFoundError` on missing dir |
| `test_safe_name_sanitises_special_chars` | `/` in security name → `_` in filename |

**`TestDownloadPriceLists`** — 3 tests

| Test | What it verifies |
|---|---|
| `test_skips_existing_files` | Already-downloaded file not re-requested |
| `test_saves_file_on_200` | HTTP 200 content written to disk |
| `test_returns_empty_on_404` | 404 returns empty saved list |

---

## 6. Configuration Reference

### Environment Variables (`.env`)

Create a `.env` file in both the project root (for `importer.py`) and inside `nse_predictor/`.

```ini
# --- News API ---
NEWS_API_KEY=your_newsapi_org_key_here

# --- LLM (OpenAI or compatible) ---
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=1500

# --- Data paths ---
OUTPUT_DIR=nse_dataset
COMPILED_DIR=C:/data/compiled_securities
PRICE_CSV_PATH=                          # leave blank; pipeline builds it

# --- Prediction tuning ---
PRICE_CHANGE_THRESHOLD=1.5
LOOKAHEAD_DAYS=3
CONFIDENCE_THRESHOLD=0.3

# --- Production system (StocksKE_architecture) ---
DATABASE_URL=postgresql://user:pass@localhost:5432/stocks_db
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
PREDICTION_THRESHOLD=0.7
KNOWLEDGE_GRAPH_PATH=path/to/nse_knowledge_graph_data.json
NEWS_DB_PATH=nse_news_database.db
```

---

## 7. How to Run

### Prerequisites

```bash
# Install Python dependencies (research pipeline)
pip install requests pandas openpyxl python-dotenv beautifulsoup4 lxml pytest

# Install production system dependencies
pip install fastapi sqlalchemy alembic xgboost scikit-learn prometheus-client \
            structlog redis celery pydantic-settings python-dateutil
```

### Step-by-step: Research Pipeline

```bash
# 1. Download price lists (e.g., full year 2024)
python importer.py download --start 2024-01-01 --end 2024-12-31

# 2. Compile per-security XLSX files
python importer.py compile --in downloaded_price_lists --out compiled_securities

# 3. Build unified prices CSV
python importer.py build-csv --in compiled_securities --out prices.csv

# 4. Run the full prediction pipeline (fetches 30 days of news)
cd nse_predictor
python pipeline.py --import-from ../compiled_securities --days 30

# 5. Evaluate accuracy on the labeled dataset
python backtest.py

# 6. Demo with synthetic data (no keys needed)
python backtest.py --demo
```

### Or in one command

```bash
python importer.py all --start 2024-01-01 --end 2024-12-31 --prices-csv prices.csv
cd nse_predictor && python pipeline.py --import-from ../compiled_securities
```

### Running Tests

```bash
cd nse_predictor
python -m pytest tests/ -v        # all 49 tests
python -m pytest tests/ -k aligner  # just aligner tests
```

### Production System (Docker)

```bash
cd StocksKE_architecture

# Copy and fill in the .env file
cp .env.example .env

# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec app alembic upgrade head

# (Optional) Load the knowledge graph
docker-compose exec app python scripts/load_knowledge_graph.py

# API is live at http://localhost:8000
# Metrics at http://localhost:8000/metrics
```

---

*Generated: 2026-05-20*
