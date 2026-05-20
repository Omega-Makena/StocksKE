import json
import time
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin
import urllib.robotparser as robotparser

import requests

from config import NEWS_API_KEY, OUTPUT_DIR
from companies import NSE_COMPANIES

logger = logging.getLogger(__name__)
USER_AGENT = "NSE-Research-Bot/1.0"


def fetch_news_api(
    company_name: str,
    ticker: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
) -> list[dict]:
    """
    Calls https://newsapi.org/v2/everything
    Query: '"{company_name}" OR "{ticker}"'
    Handles: HTTP errors, rate limits (429 → sleep 60s + retry),
             empty responses, missing fields.
    Returns empty list on any unrecoverable error. Never raises.
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not configured; skipping fetch for %s", company_name)
        return []

    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": NEWS_API_KEY, "User-Agent": USER_AGENT}
    q = f'"{company_name}" OR "{ticker}"'
    params = {
        "q": q,
        "from": from_date,
        "to": to_date,
        "pageSize": min(page_size, 100),
        "language": "en",
        "sortBy": "publishedAt",
    }

    attempts = 0
    while attempts < 3:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited by News API; sleeping 60s")
                time.sleep(60)
                attempts += 1
                continue
            resp.raise_for_status()
            data = resp.json()
            items = []
            for a in data.get("articles", []):
                items.append(
                    {
                        "ticker": ticker,
                        "company": company_name,
                        "title": a.get("title") or "",
                        "description": a.get("description") or "",
                        "content": a.get("content") or "",
                        "source": (a.get("source") or {}).get("name", "") if isinstance(a.get("source"), dict) else a.get("source", ""),
                        "published_at": a.get("publishedAt") or "",
                        "url": a.get("url") or "",
                    }
                )
            return items
        except requests.RequestException as e:
            logger.exception("Network error fetching news for %s (%s)", company_name, e)
            attempts += 1
            time.sleep(2 ** attempts)
        except ValueError as e:
            logger.exception("JSON decode error for %s: %s", company_name, e)
            return []
    logger.error("Failed to fetch news for %s after retries", company_name)
    return []


def scrape_business_daily(max_pages: int = 5) -> list[dict]:
    """
    Scrapes https://www.businessdailyafrica.com/bd/markets/companies
    Uses BeautifulSoup to extract article titles, URLs, and dates.
    Sets ticker="UNKNOWN" and company="UNKNOWN" — these get resolved
    in the extractor module.
    Respects robots.txt. Sleeps 2s between pages.
    Returns empty list on failure. Never raises.
    """
    from bs4 import BeautifulSoup

    base = "https://www.businessdailyafrica.com"
    start_path = "/bd/markets/companies"
    robot_url = urljoin(base, "/robots.txt")

    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robot_url)
        rp.read()
        if not rp.can_fetch("NSE-Research-Bot", urljoin(base, start_path)) and not rp.can_fetch("*", urljoin(base, start_path)):
            logger.warning("Scrape disallowed by robots.txt: %s", robot_url)
            return []
    except Exception:
        logger.info("Could not read robots.txt; proceeding with caution")

    results = []
    headers = {"User-Agent": USER_AGENT}
    for p in range(1, max_pages + 1):
        try:
            page_url = urljoin(base, start_path) if p == 1 else f"{urljoin(base, start_path)}?page={p}"
            resp = requests.get(page_url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            anchors = soup.find_all("a", href=True)
            seen = set()
            for a in anchors:
                href = a.get("href")
                if not href:
                    continue
                full = urljoin(base, href)
                if start_path in href and full not in seen:
                    title = (a.get_text() or "").strip()
                    # try to find nearby time tag
                    dt = ""
                    parent = a.parent
                    time_tag = None
                    if parent is not None:
                        time_tag = parent.find("time") if hasattr(parent, "find") else None
                    if time_tag:
                        dt = time_tag.get("datetime") or (time_tag.get_text() or "")
                    results.append(
                        {
                            "ticker": "UNKNOWN",
                            "company": "UNKNOWN",
                            "title": title,
                            "description": "",
                            "content": "",
                            "source": "Business Daily",
                            "published_at": dt,
                            "url": full,
                        }
                    )
                    seen.add(full)
            time.sleep(2)
        except requests.RequestException as e:
            logger.exception("Failed to scrape Business Daily page %s: %s", p, e)
            return []
        except Exception as e:
            logger.exception("Unexpected parsing error on page %s: %s", p, e)
            return []
    return results


def collect_all(days_back: int = 30) -> list[dict]:
    """
    Runs fetch_news_api for every company in NSE_COMPANIES,
    then runs scrape_business_daily.
    Deduplicates by URL.
    Saves to OUTPUT_DIR/news/news_{from_date}_to_{to_date}.jsonl
    Returns combined list.
    """
    total = len(NSE_COMPANIES)
    today = datetime.utcnow().date()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    out_dir = Path(OUTPUT_DIR) / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"news_{from_date}_to_{to_date}.jsonl"

    aggregated = []
    seen_urls = set()
    for i, comp in enumerate(NSE_COMPANIES, start=1):
        name = comp.get("name")
        ticker = comp.get("ticker")
        logger.info("[%d/%d] Fetching %s...", i, total, name)
        try:
            items = fetch_news_api(name, ticker, from_date, to_date)
            for it in items:
                url = it.get("url") or ""
                if url in seen_urls:
                    continue
                aggregated.append(it)
                seen_urls.add(url)
        except Exception as e:
            logger.exception("Error fetching for %s: %s", name, e)
        time.sleep(0.5)

    # add scraped items
    try:
        scraped = scrape_business_daily(max_pages=5)
        for it in scraped:
            url = it.get("url") or ""
            if url and url not in seen_urls:
                aggregated.append(it)
                seen_urls.add(url)
    except Exception:
        logger.exception("Error scraping Business Daily")

    # write file
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for a in aggregated:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        logger.info("Saved %d articles to %s", len(aggregated), out_file)
    except Exception as e:
        logger.exception("Failed to save news jsonl: %s", e)

    return aggregated
