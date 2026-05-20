import logging
import os
import json
import sqlite3
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
DB_PATH = os.getenv("NEWS_DB_PATH", "nse_news_database.db")
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

# Knowledge Graph path — override via env var to avoid hardcoding
KNOWLEDGE_GRAPH_PATH = os.getenv(
    "KNOWLEDGE_GRAPH_PATH",
    r"C:\Users\omegam\.gemini\antigravity\brain\2676afa7-7048-4960-96a7-a70be160f744\nse_knowledge_graph_data.json",
)


def setup_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialise the SQLite database and return an open connection."""
    logger.info("Setting up SQLite database at: %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            title         TEXT,
            description   TEXT,
            content       TEXT,
            url           TEXT UNIQUE,
            published_at  TEXT,
            source_name   TEXT,
            matched_tickers TEXT,
            ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def load_knowledge_graph(path: str = KNOWLEDGE_GRAPH_PATH) -> list[dict]:
    """Load listed companies from the Knowledge Graph JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge Graph not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("companies", [])


def _chunk(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _strip_suffixes(name: str) -> str:
    for suffix in (" Limited", " PLC", " Plc", " Ltd"):
        name = name.replace(suffix, "")
    return name.strip()


def _fetch_with_retry(params: dict) -> dict | None:
    """
    Call the News API with exponential-backoff retry.
    Returns parsed JSON dict or None on unrecoverable failure.
    """
    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY is not set — cannot fetch news")
        return None

    headers = {"X-Api-Key": NEWS_API_KEY, "User-Agent": "StocksKE-Bot/1.0"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                NEWS_API_ENDPOINT, headers=headers, params=params, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 429:
                logger.warning("Rate limited (429); sleeping 60s before retry")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
            time.sleep(2 ** attempt)
    logger.error("All %d attempts failed", MAX_RETRIES)
    return None


def fetch_and_store_news(conn: sqlite3.Connection, companies: list[dict]) -> int:
    """
    Fetch news from the last 24 hours for all Knowledge Graph companies
    and insert new articles into the database.
    Returns the number of articles inserted.
    """
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Fetching news published on or after: %s", yesterday_str)

    cursor = conn.cursor()
    articles_inserted = 0

    # Query at most 15 companies per request to stay within URL length limits
    chunks = list(_chunk(companies, 15))

    for i, chunk in enumerate(chunks):
        logger.info("Processing batch %d/%d …", i + 1, len(chunks))

        keywords = [f'"{_strip_suffixes(c["name"])}"' for c in chunk]
        query_str = "(" + " OR ".join(keywords) + ") AND Kenya"

        params = {
            "q": query_str,
            "from": yesterday_str,
            "language": "en",
            "sortBy": "publishedAt",
        }

        data = _fetch_with_retry(params)
        if data is None:
            continue

        if data.get("status") != "ok":
            logger.warning("API error for batch %d: %s", i + 1, data.get("message"))
            continue

        articles = data.get("articles", [])
        logger.info("  → %d articles returned for batch %d", len(articles), i + 1)

        for article in articles:
            url = article.get("url") or ""
            if not url:
                continue

            title       = article.get("title") or ""
            description = article.get("description") or ""
            content     = article.get("content") or ""
            published_at = article.get("publishedAt") or ""
            source_name = (article.get("source") or {}).get("name", "")

            text_lower = f"{title} {description} {content}".lower()

            # Determine which tickers from this batch matched the article text.
            # The API sometimes returns broad matches, so we do a secondary text check.
            matched = [
                c["ticker"]
                for c in chunk
                if _strip_suffixes(c["name"]).lower() in text_lower
            ]
            matched_str = ",".join(matched) if matched else "MULTIPLE"

            cursor.execute(
                """
                INSERT OR IGNORE INTO news_articles
                    (title, description, content, url, published_at, source_name, matched_tickers)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (title, description, content, url, published_at, source_name, matched_str),
            )
            if cursor.rowcount > 0:
                articles_inserted += 1

    conn.commit()
    logger.info("Inserted %d new unique articles into the database.", articles_inserted)
    return articles_inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not NEWS_API_KEY:
        raise SystemExit("ERROR: NEWS_API_KEY is not set. Add it to your .env file.")

    print("=" * 50)
    print("   StocksKE — News Ingestion Pipeline")
    print("=" * 50)

    db_conn = setup_database()
    try:
        companies_list = load_knowledge_graph()
        logger.info("Loaded %d companies from the Knowledge Graph.", len(companies_list))
        fetch_and_store_news(db_conn, companies_list)
    finally:
        db_conn.close()
        logger.info("Database connection closed.")
