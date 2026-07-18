# StocksKE

**An event-exposure and news-analysis engine for the Nairobi Securities
Exchange (NSE).** Given a news event, it maps *which* NSE-listed names are
connected to it — directly, and through a knowledge graph of competitors,
sectors, shared products, and macro/commodity drivers — and flags high-severity
events for review.

> ### What it does — and what it doesn't (be honest with stakeholders)
> - ✅ **Reliable:** *exposure mapping* — "this event connects to KQ (shared
>   Boeing fleet), to TOTL vs KQ/cement (oil price), to the banking sector
>   (rates)…", plus LLM-extracted event type, severity and sentiment.
> - 🧪 **Experimental / unproven:** predicting the *direction and magnitude* of
>   the resulting price move. On a real point-in-time event study (abnormal
>   returns vs baselines) it currently shows **no demonstrated edge** and, before
>   hardening, systematically over-called moves. Directional output is therefore
>   **conservative by default** (only severe events get a call) and stays
>   experimental until `forward.py` accumulates a statistically meaningful
>   backtest. Do not present accuracy claims that the harness has not earned.

### The output: event alerts (`alert.py`)

Each news event becomes a ranked **alert**, not a price call:

```
[MEDIUM] EARNINGS (severity 0.8) — KCB Group   2026-07-01
  6 NSE names exposed | move-likelihood ranked (direction = context only, NOT a prediction):
    KCB    move~48% [MEDIUM] dir(info)=UP      via direct
    EQTY   move~41% [MEDIUM] dir(info)=NEUTRAL via competitor
    ...
```

The **move-likelihood** score (event-type prior × severity × coupling) ranks how
likely each exposed name is to have an *abnormal move* — validated to sort by
realised move rate. Direction is shown for context only. Ship *this*; do not ship
a direction predictor.

### Data source matters — use Kenyan news, not global NewsAPI

A rigorous check found **NewsAPI returns ~99.7% foreign noise for NSE-ticker
queries** (global "KCB"/"Equity"/"TCL" matches), and foreign entities collide
with NSE tickers ("TCL" the electronics brand vs TCL = TransCentury). So:
`collector.is_relevant` **trusts Kenyan RSS feeds** (Standard, Nation, Capital
FM, KBC, Business Daily…) on the signal alone, and **requires a Kenyan-context
term for any other source**. Prefer RSS; treat NewsAPI as low-value here.

It is built in three independent layers that feed into each other:

| Layer | Folder | What it does |
|---|---|---|
| **Importer** | `importer/` | Downloads daily price lists from Innova, compiles per-security XLSX files, exports a unified prices CSV |
| **Pipeline** | `pipeline/` | Collects NSE news, LLM-extracts the source event, propagates it through the knowledge graph to an **exposure map** (+ conservative directional read), and scores predictions against realised **abnormal** returns |
| **Production** | `production/` | Dockerised FastAPI + Celery service — ingests prices and news at scale, trains XGBoost models, sends Telegram alerts |

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                          StocksKE                               │
  │                                                                 │
  │  ┌───────────────┐     ┌────────────────────────────────────┐  │
  │  │   IMPORTER    │     │           PIPELINE                 │  │
  │  │               │     │                                    │  │
  │  │  Innova XLS   │     │  News API + Business Daily         │  │
  │  │  files        │     │       │                            │  │
  │  │      │        │     │  collector.py                      │  │
  │  │  compile      │     │       │                            │  │
  │  │      │        │     │  extractor.py  (LLM)               │  │
  │  │  prices.csv ──┼─────►       │                            │  │
  │  │               │     │  validator.py  (hallucination gate) │  │
  │  └───────────────┘     │       │                            │  │
  │                        │  aligner.py   (ground-truth label) │  │
  │                        │       │                            │  │
  │                        │  backtest.py  (accuracy report)    │  │
  │                        └────────────────────────────────────┘  │
  │                                                                 │
  │  ┌──────────────────────────────────────────────────────────┐  │
  │  │                     PRODUCTION                           │  │
  │  │                                                          │  │
  │  │  NewsFetcher ──► Redis ──► SentimentAnalyzer (LLM)      │  │
  │  │  PriceScraper ──────────► TimescaleDB                   │  │
  │  │  features.py ──► ModelTrainer (XGBoost, weekly)         │  │
  │  │                  Predictor (hourly) ──► Telegram alert   │  │
  │  │  FastAPI  /feedback  /health  /metrics                   │  │
  │  └──────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
StocksKE/
├── importer/                   # Price data ingestion
│   ├── importer.py             # Download → compile → export prices CSV
│   └── tests/
│       └── test_importer.py
│
├── pipeline/                   # News-driven LLM prediction pipeline
│   ├── collector.py            # Keyless Kenyan news RSS (Standard/Nation/…) + optional NewsAPI
│   ├── extractor.py            # LLM analysis → source event (entities, severity)
│   ├── validator.py            # Hallucination filter
│   ├── graph.py                # Knowledge graph + impact-propagation engine
│   ├── graph_data.json         # Curated graph structure (products/drivers/suppliers)
│   ├── graph_sources.py        # Data-derived edges (price co-movement, co-occurrence)
│   ├── graph_export.json       # Full graph export (consumed by production loader)
│   ├── alert.py                # Event-alert product: exposure map + move-likelihood score
│   ├── forward.py              # Daily forward-accumulation runner (grows the backtest)
│   ├── build_dataset.py        # Pull historical NewsAPI news for a labelled set
│   ├── aligner.py              # Join predictions with realised prices
│   ├── calibrate.py            # Fit propagation magnitude to realised moves
│   ├── pipeline.py             # Orchestrator (runs all steps)
│   ├── backtest.py             # Accuracy evaluation (raw, single-horizon)
│   ├── harness.py              # Honest event-study backtest (abnormal returns)
│   ├── companies.py            # NSE company registry (49 equities)
│   ├── config.py               # All settings via env vars
│   ├── requirements.txt
│   └── tests/
│       ├── test_validator.py
│       ├── test_aligner.py
│       └── test_extractor.py
│
├── production/                 # Dockerised production system
│   ├── app/
│   │   ├── ingestion/          # NewsFetcher, PriceScraper
│   │   ├── processing/         # SentimentAnalyzer (LLM via Redis queue)
│   │   ├── prediction/         # Feature engineering, XGBoost train/predict
│   │   ├── alerting/           # Telegram notifier
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── scheduler/          # Celery app + Beat tasks
│   │   └── main.py             # FastAPI server
│   ├── config/settings.py      # Pydantic settings
│   ├── scripts/                # DB init, knowledge graph loader
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── DOCUMENTATION.md            # Full module-by-module reference
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Quick Start

### 1 — Setup

```bash
git clone https://github.com/Omega-Makena/StocksKE.git
cd StocksKE
cp .env.example .env          # fill in your API keys
```

### 2 — Install dependencies

```bash
# Pipeline layer
pip install -r pipeline/requirements.txt

# Importer (uses pandas + requests)
pip install requests pandas openpyxl xlrd python-dotenv   # xlrd is required to read the .xls price lists
```

### 3 — Download and compile price data

```bash
# Download raw XLS files from Innova
python importer/importer.py download --start 2024-01-01 --end 2024-12-31

# Compile into one XLSX per security
python importer/importer.py compile --in downloaded_price_lists --out compiled_securities

# Export the unified prices CSV the pipeline needs
python importer/importer.py build-csv --in compiled_securities --out prices.csv

# Or do all three in one command
python importer/importer.py all --start 2024-01-01 --end 2024-12-31 --prices-csv prices.csv
```

### 4 — Run the prediction pipeline

```bash
cd pipeline

# Full run — fetches 30 days of news, builds predictions, labels them against prices
python pipeline.py --import-from ../compiled_securities --days 30

# Re-run analysis on already-downloaded news (skip the API calls)
python pipeline.py --import-from ../compiled_securities --skip-collect

# Or point at an already-built prices CSV directly
python pipeline.py --prices-csv ../prices.csv
```

### 5 — Evaluate accuracy

```bash
cd pipeline

# Run backtest on all labeled data
python backtest.py

# Demo with synthetic data (no API keys needed)
python backtest.py --demo

# Output raw JSON metrics
python backtest.py --json
```

### 5a — Honest event-study backtest

`backtest.py` labels on *raw* price change over one horizon — good for a smoke
test, misleading as evidence. `harness.py` is the trustworthy evaluation:

```bash
cd pipeline
python harness.py --predictions nse_dataset/extractions/extractions_XXXX.jsonl \
                  --prices ../prices.csv --horizons 1,3,5
```

It is **point-in-time** (betas, liquidity and momentum for an event at date D use
only data before D — no lookahead), labels on **abnormal returns**
(`AR = R_stock − β·R_market`, so index-wide moves are removed), **filters
illiquid names** (stale/absent prices), **flags corporate actions** (implausible
single-day jumps it can't adjust for), scores **multiple horizons**, and — most
importantly — reports the model against **baselines** (always-NEUTRAL, majority,
random, momentum). If it can't beat those, the "accuracy" is noise. It also
reports magnitude MAE and a confidence-calibration curve.

> Note: NSE prices are daily closes, so the spec's "1 hour" horizon isn't
> achievable without an intraday feed. Corporate-action handling is a heuristic
> jump-filter, not a true adjusted-close.

### 5a-ii — Forward accumulation (getting a real-sized backtest)

A single scored run is meaningless: RSS carries only the last few days of news,
whose prediction horizons fall in the *future* and can't be scored yet. The
honest way to build a statistically meaningful backtest is to **run daily** —
each day adds a few predictions, and each day more *old* predictions have their
horizons realise and become scoreable.

```bash
cd pipeline
python forward.py            # refresh prices · collect+extract NEW news · re-score everything
python forward.py --score-only   # just re-score accumulated predictions
```

`forward.py` reports the growing sample, e.g. `Accumulated N predictions ->
scoreable per horizon {1: 11, 3: 6}`. Schedule it daily (Windows Task Scheduler
or cron); over a few weeks the scoreable sample grows into the hundreds and the
harness scorecard becomes trustworthy. Cross-run dedup ensures only unseen
articles hit the LLM, so the cost stays incremental.

### 5b — Calibrate the propagation engine

Two levels of calibration, both persisted to `calibration.json` (auto-loaded by
`graph.py` on import):

```bash
cd pipeline

# Magnitude fit — MAGNITUDE_SCALE + DIRECTION_THRESHOLD from labeled rows
python calibrate.py --demo          # self-test: recovers a known scale
python calibrate.py --write         # fit from nse_dataset/labeled and persist

# Structural fit — HOP_DECAY + per-family channel gains (sector / competitor /
# product / supplier). Re-propagates each event under candidate coefficients.
python calibrate.py --demo-structural           # self-test on synthetic events
python calibrate.py --fit-structural --events events.jsonl --write
```

The structural fitter needs *event records* (`{event_type, sources, realised}`),
which `calibrate.build_event_records(predictions, prices_csv)` builds from real
extractor output + a prices CSV. It fits one gain per channel *family* (not per
event×edge cell) to avoid over-fitting; the hand-set sign structure in
`graph.CHANNEL` is preserved.

### 6 — Run tests

```bash
# Pipeline tests (49 tests)
cd pipeline && python -m pytest tests/ -v

# Importer tests
cd importer && python -m pytest tests/ -v
```

---

## Production System (Docker)

```bash
cd production
cp .env.example .env            # fill in DATABASE_URL, REDIS_URL, API keys

docker-compose up -d            # starts app, postgres, redis, celery worker

docker-compose exec app alembic upgrade head          # run DB migrations
docker-compose exec app python scripts/load_knowledge_graph.py  # seed tickers
```

Services:
- **API** → `http://localhost:8000`
- **Metrics** → `http://localhost:8000/metrics` (Prometheus)
- **Celery Beat** runs `fetch_news` hourly, `run_predictions` hourly, `retrain_model` weekly

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Key variables:

| Variable | Used by | Description |
|---|---|---|
| `NEWS_API_KEY` | pipeline, production | [newsapi.org](https://newsapi.org) key — **optional**; without it the collector uses keyless Kenyan RSS feeds (Standard, Nation, Capital FM, KBC, Kenyans.co.ke) |
| `OPENAI_API_KEY` | pipeline, production | LLM API key |
| `OPENAI_BASE_URL` | pipeline | Swap in any OpenAI-compatible endpoint |
| `MODEL_NAME` | pipeline | e.g. `gpt-4o-mini` |
| `MAX_TOKENS` | pipeline | LLM output cap (default `600` — lean schema) |
| `MAX_ARTICLE_CHARS` | pipeline | Article truncation before sending (default `1600`) |
| `PREFILTER_ARTICLES` | pipeline | Skip the LLM for articles with no NSE/macro hit (default `1`) |
| `COMPILED_DIR` | pipeline | Path to `compiled_securities/` from importer |
| `PRICE_CHANGE_THRESHOLD` | pipeline | % move to label as UP/DOWN (default `1.5`) |
| `LOOKAHEAD_DAYS` | pipeline | Days after article to check price (default `3`) |
| `DATABASE_URL` | production | PostgreSQL + TimescaleDB connection string |
| `REDIS_URL` | production | Redis connection string |
| `TELEGRAM_BOT_TOKEN` | production | Telegram bot for price alerts |
| `PREDICTION_THRESHOLD` | production | Min XGBoost probability to fire an alert (default `0.7`) |

See `.env.example` for the full list.

---

## How the Pipeline Works

```
  News article
       │
       ▼
  collector.py    →  keyless Kenyan news RSS (Standard/Nation/Capital FM/KBC/
       │              Kenyans.co.ke) + optional NewsAPI + Business Daily scrape
       │
  extractor.py    →  LLM reads article, outputs the SOURCE EVENT:
       │              { event_type, severity, source_entities (companies +
       │                products, NSE or not), directly_affected }
       │
  validator.py    →  drops hallucinated tickers, invalid directions, bad confidence
       │
  graph.py        →  propagates impact through the knowledge graph.
       │              Seeds from source_entities + directly_affected, spreads
       │              along typed edges (competitor / sector / shared-product /
       │              supplier) with per-hop decay, and computes each affected
       │              ticker's { direction, magnitude %, confidence } —
       │              rebuilding indirectly_affected.
       │
  aligner.py      →  looks up realised price change (from prices.csv)
       │              assigns ground-truth label, marks correct/wrong,
       │              records magnitude_error (predicted % vs realised %)
       │
  backtest.py     →  precision · recall · F1 per class
                     accuracy by confidence band, impact type, ticker
```

**Knowledge-graph propagation** — the source event drives a typed graph, so a
foreign/unlisted event reaches NSE names without being named directly:

```
  Ethiopian Airlines crash (disaster, severity 1.0, DOWN)
        │  seeds product node
        ▼
   product: Boeing 737 MAX
        │
   ┌────┴───────────────┐
   ▼                    ▼
 Boeing (made_by)   Kenya Airways / KQ  (operated_by → shared-fleet
 [non-NSE]           contagion → DOWN, magnitude & confidence decayed by hop)
```

The graph carries four node kinds and several typed edges:

| Node kind | Examples | Role |
|---|---|---|
| **company** | KCB, KQ, Boeing*, RwandAir* | tradeable NSE names + non-NSE anchors (*) |
| **sector** | `sector:Banking` | spillover hub linking peers |
| **product** | `product:Boeing 737 MAX` | shared physical asset (fleet contagion) |
| **driver** | `driver:CBK rate`, `driver:Oil price`, `driver:KES/USD` | macro / commodity / shared-input hub |

**Driver nodes** carry the macro & commodity channels. A driver fans out to
firms via two polarity-typed edges — `helps_when_up` and `hurts_when_up` — so a
single event moves winners and losers in opposite directions:

```
  CBK rate  UP  (macro)            Oil price  UP  (commodity)
     ├── banks           → UP         ├── TOTL (marketer)     → UP
     └── HFCK/KPLC/BAMB… → DOWN       └── KQ/cement/brewer…   → DOWN
```

Coefficients (`MAGNITUDE_SCALE`, `HOP_DECAY`, per-family `CHANNEL_GAINS`,
per-event channel weights) are module constants in `graph.py`, fitted against
realised prices by `calibrate.py` and persisted to `calibration.json`
(auto-loaded on import).

### Where the graph comes from (not hardcoded)

The graph is composed from layered sources, so it can grow from data rather than
code edits:

| Source | Where | What it contributes |
|---|---|---|
| **Company registry** | `companies.py` | 49 NSE equities, sectors, competitor pairs |
| **Curated structure** | `graph_data.json` | products, non-NSE aliases, supplier chains, macro/commodity drivers — edit this file, no code change |
| **Price co-movement** | `graph_sources.py` | peer edges *discovered* from correlated returns in `prices.csv` |
| **Article co-occurrence** | `graph_sources.py` | association edges accrued from LLM `source_entities`, so novel entities wire themselves in |

`graph.build_graph(prices_csv=…, extraction_paths=…)` composes them;
`build_default_graph()` is the curated seed alone. `graph.export_graph()` writes
`graph_export.json`, the single artifact the production DB loader
(`production/scripts/load_knowledge_graph.py`) consumes to populate its
`sectors`, `tickers`, and `graph_edges` tables — one source of truth across both
processes.

---

## Prediction Schema

Each LLM prediction looks like this:

```json
{
  "event_type": "earnings",
  "primary_sector": "Banking",
  "directly_affected": [
    {
      "ticker": "KCB",
      "direction": "UP",
      "confidence": 0.91,
      "impact_type": "direct",
      "reasoning": "Strong earnings beat drives buying pressure."
    }
  ],
  "indirectly_affected": [
    {
      "ticker": "EQTY",
      "direction": "NEUTRAL",
      "confidence": 0.42,
      "impact_type": "competitor"
    }
  ]
}
```

Valid tickers are the 49 NSE-listed equities in `pipeline/companies.py`. Any ticker outside that list is rejected by the validator.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Price ingestion | Python, `requests`, `pandas`, `openpyxl` |
| News collection | `requests`, `BeautifulSoup` |
| LLM analysis | OpenAI-compatible chat API (`gpt-4o-mini` default) |
| Backtesting | Pure Python, `pytest` |
| Production API | FastAPI, SQLAlchemy, Alembic |
| Task queue | Celery + Redis |
| Time-series DB | PostgreSQL + TimescaleDB |
| ML model | XGBoost |
| Observability | Prometheus, Structlog |
| Deployment | Docker, Docker Compose |

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system architecture with diagrams: data flow, the knowledge graph, graph sourcing, and production notes/limitations.
- **[DOCUMENTATION.md](DOCUMENTATION.md)** — full module-by-module reference, function signatures, and test documentation.
