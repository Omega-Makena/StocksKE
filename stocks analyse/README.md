# StocksKE

A system for predicting price movements on the **Nairobi Securities Exchange (NSE)**.

It is built in three independent layers that feed into each other:

| Layer | Folder | What it does |
|---|---|---|
| **Importer** | `importer/` | Downloads daily price lists from Innova, compiles per-security XLSX files, exports a unified prices CSV |
| **Pipeline** | `pipeline/` | Collects NSE news, runs LLM analysis on each article, validates predictions, aligns them against realised prices, measures accuracy |
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
│   ├── collector.py            # News API + Business Daily scraper
│   ├── extractor.py            # LLM analysis → structured predictions
│   ├── validator.py            # Hallucination filter
│   ├── aligner.py              # Join predictions with realised prices
│   ├── pipeline.py             # Orchestrator (runs all steps)
│   ├── backtest.py             # Accuracy evaluation
│   ├── companies.py            # NSE company registry (46 equities)
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
pip install requests pandas openpyxl python-dotenv
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
| `NEWS_API_KEY` | pipeline, production | [newsapi.org](https://newsapi.org) key |
| `OPENAI_API_KEY` | pipeline, production | LLM API key |
| `OPENAI_BASE_URL` | pipeline | Swap in any OpenAI-compatible endpoint |
| `MODEL_NAME` | pipeline | e.g. `gpt-4o-mini` |
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
  collector.py    →  fetches from newsapi.org and businessdailyafrica.com
       │
  extractor.py    →  LLM reads article, outputs:
       │              { ticker, direction: UP/DOWN/NEUTRAL, confidence, impact_type }
       │
  validator.py    →  drops hallucinated tickers, invalid directions, bad confidence
       │
  aligner.py      →  looks up realised price change (from prices.csv)
       │              assigns ground-truth label, marks prediction correct/wrong
       │
  backtest.py     →  precision · recall · F1 per class
                     accuracy by confidence band, impact type, ticker
```

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

Valid tickers are the 46 NSE-listed equities in `pipeline/companies.py`. Any ticker outside that list is rejected by the validator.

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

Full module-by-module reference including architecture diagrams, function signatures, and test documentation is in [DOCUMENTATION.md](DOCUMENTATION.md).
