# Kenyan Stock Market Prediction and Alert System

A production-ready system for ingesting Kenyan stock market data, performing sentiment analysis, predicting market movements using XGBoost, and sending alerts via Telegram.

## Architecture & Technology Stack

- **Python 3.10+**: Core programming language.
- **PostgreSQL 14+ with TimescaleDB**: Time-series database for financial quotes and relational data.
- **Redis & Celery**: Distributed task queue for asynchronous ingestion and periodic machine learning jobs.
- **FastAPI**: Synchronous feedback API block.
- **SQLAlchemy & Alembic**: ORM and migrations.
- **BeautifulSoup**: Price and specific market data scraping.
- **OpenAI API**: Intelligent LLM-driven sentiment analysis on Kenyan news flow.
- **XGBoost**: Tree-booster logic for the core predictive module on closing variations.
- **Prometheus & Structlog**: Deep system observability and transparent error tracking.
- **Docker & Docker Compose**: Unified local and remote deployment setup.

## Project Structure
Detailed in the codebase (follows standard package hierarchies).

## Local Development & Running

### Requirements
- Docker and Docker Compose installed.

### Setup Steps
1. Copy `.env.example` to `.env` and fill in the dummy values with your API keys:
   ```bash
   cp .env.example .env
   ```

2. Start the services via Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Initialize the database and schemas (Almembic migrations):
   ```bash
   docker-compose exec app alembic upgrade head
   ```

4. (Optional) Load initial knowledge graph data:
   ```bash
   docker-compose exec app python scripts/load_knowledge_graph.py
   ```

5. The system is live!
   - API: http://localhost:8000
   - Metrics: http://localhost:8000/metrics

### Notes on Services
- **fetch_news**: Polled continuously at your definition interval (hourly via Celery Beat setup).
- **scrape_prices**: EOD metrics captured natively out of targeted hosts.
- **retrain_model**: Sunday weekly calibration tasks.
- **run_predictions**: Hourly updates evaluated into thresholds yielding alerting vectors.