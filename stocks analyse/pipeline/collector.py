import json
import time
import logging
import os
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin
import urllib.robotparser as robotparser

import requests

from config import NEWS_API_KEY, OUTPUT_DIR, ROBOTS_FAIL_OPEN, DEDUP_STORE
from companies import NSE_COMPANIES

logger = logging.getLogger(__name__)
USER_AGENT = "NSE-Research-Bot/1.0"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NSE-Research-Bot/1.0"

# Keyless Kenyan news RSS feeds, verified reachable and parseable. Business
# sections first (highest NSE signal); general feeds are filtered downstream by
# the extractor's is_relevant() prefilter. Tuko is intentionally excluded — it
# hard-blocks automated requests (HTTP 403).
RSS_FEEDS: list[tuple[str, str]] = [
    ("Standard", "https://www.standardmedia.co.ke/rss/business.php"),
    ("Standard", "https://www.standardmedia.co.ke/rss/headlines.php"),
    ("Nation", "https://nation.africa/kenya/rss.xml"),
    ("Capital FM", "https://www.capitalfm.co.ke/business/feed/"),
    ("KBC", "https://www.kbc.co.ke/category/business/feed/"),
    ("Kenyans.co.ke", "https://www.kenyans.co.ke/feeds/news"),
]


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


def _rss_date_to_iso(raw: str) -> str:
    """RFC-822 pubDate ('Fri, 10 Jul 2026 12:54:18 +0300') -> 'YYYY-MM-DD'.
    Returns '' if unparseable (downstream can still use the article)."""
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError, IndexError):
        # some feeds already emit ISO; take the leading date if present
        raw = raw.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw[:len(datetime.now().strftime(fmt))], fmt).date().isoformat()
            except ValueError:
                pass
    return ""


def parse_rss(xml_bytes: bytes, source: str) -> list[dict]:
    """Parse RSS/Atom bytes into article dicts. Handles both <item> (RSS) and
    <entry> (Atom). Never raises — returns [] on malformed XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        logger.warning("Malformed feed XML from %s", source)
        return []

    def _text(el, *tags):
        for t in tags:
            found = el.find(t)
            if found is not None and (found.text or "").strip():
                return found.text.strip()
        return ""

    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    out = []
    for it in items:
        title = _text(it, "title", "{http://www.w3.org/2005/Atom}title")
        desc = _text(it, "description", "summary", "{http://www.w3.org/2005/Atom}summary")
        link = _text(it, "link", "guid")
        if not link:  # Atom link is an attribute
            le = it.find("{http://www.w3.org/2005/Atom}link")
            link = le.get("href", "") if le is not None else ""
        pub = _text(it, "pubDate", "{http://purl.org/dc/elements/1.1/}date",
                    "published", "{http://www.w3.org/2005/Atom}updated")
        if not (title or link):
            continue
        out.append({
            "ticker": "UNKNOWN",
            "company": "UNKNOWN",
            "title": title,
            "description": desc,
            "content": "",           # RSS gives a summary, not full body
            "source": source,
            "published_at": _rss_date_to_iso(pub),
            "url": link,
        })
    return out


def scrape_rss_feeds(feeds: list[tuple[str, str]] = None, sleep_s: float = 1.0) -> list[dict]:
    """Fetch and parse the configured Kenyan news RSS feeds (keyless). Per-feed
    errors are logged and skipped; the function never raises."""
    feeds = feeds if feeds is not None else RSS_FEEDS
    results: list[dict] = []
    headers = {"User-Agent": BROWSER_UA}
    for source, url in feeds:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                logger.warning("Feed %s returned HTTP %s", url, resp.status_code)
                continue
            items = parse_rss(resp.content, source)
            logger.info("RSS %s (%s): %d items", source, url, len(items))
            results.extend(items)
        except requests.RequestException as e:
            logger.warning("Failed to fetch feed %s: %s", url, e)
        time.sleep(sleep_s)
    return results


_ROBOTS_CACHE: dict[str, robotparser.RobotFileParser] = {}


def _robots_allowed(url: str) -> bool:
    """Politeness check, cached per host. On any error, allow (fail-open)."""
    try:
        from urllib.parse import urlparse
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        rp = _ROBOTS_CACHE.get(host)
        if rp is None:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(host, "/robots.txt"))
            rp.read()
            _ROBOTS_CACHE[host] = rp
        return rp.can_fetch(USER_AGENT, url) or rp.can_fetch("*", url)
    except Exception:
        # couldn't read robots.txt: allow (availability) or block (compliance)
        return ROBOTS_FAIL_OPEN


def fetch_article_body(url: str, timeout: int = 15, max_chars: int = 4000) -> str:
    """Fetch an article page and extract its main text (paragraph content).

    RSS feeds only carry a headline + short summary; this pulls the body so the
    LLM has the full event to reason over. Uses a simple <article>/<p> heuristic
    (no heavy readability dependency). Respects robots.txt; never raises."""
    if not url or not _robots_allowed(url):
        return ""
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=timeout)
        if resp.status_code != 200 or "html" not in resp.headers.get("Content-Type", ""):
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup
        paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
        text = " ".join(t for t in paras if len(t) > 30)  # drop nav/boilerplate scraps
        return text[:max_chars]
    except Exception:
        logger.debug("Body fetch failed for %s", url, exc_info=True)
        return ""


def enrich_articles_with_body(articles: list[dict], should_fetch=None,
                              limit: int = 60, sleep_s: float = 0.4) -> int:
    """Fill each article's empty ``content`` with its fetched body. Only articles
    passing ``should_fetch`` (default: all) are fetched, capped at ``limit`` to
    bound requests/cost. Returns how many bodies were filled."""
    filled = 0
    for a in articles:
        if filled >= limit:
            break
        if a.get("content"):
            continue
        if should_fetch is not None and not should_fetch(a):
            continue
        body = fetch_article_body(a.get("url", ""))
        if body:
            a["content"] = body
            filled += 1
            time.sleep(sleep_s)
    logger.info("Enriched %d article bodies (limit %d)", filled, limit)
    return filled


def _seen_store_path() -> Path | None:
    if not DEDUP_STORE:
        return None
    return Path(OUTPUT_DIR) / DEDUP_STORE


def _load_seen() -> set[str]:
    p = _seen_store_path()
    if not p or not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("Could not read dedup store %s; starting fresh", p)
        return set()


def _save_seen(seen: set[str], keep: int = 20000) -> None:
    p = _seen_store_path()
    if not p:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # bound growth: keep the most recent `keep` keys (dict preserves order)
        p.write_text(json.dumps(list(seen)[-keep:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Could not persist dedup store %s", p)


def collect_all(days_back: int = 30, fetch_bodies: bool = True,
                body_limit: int = 60, persist_dedup: bool = True) -> list[dict]:
    """
    Runs fetch_news_api for every company in NSE_COMPANIES,
    then runs scrape_business_daily.
    Deduplicates by URL.
    Saves to OUTPUT_DIR/news/news_{from_date}_to_{to_date}.jsonl
    Returns combined list.
    """
    total = len(NSE_COMPANIES)
    # NSE trades in East Africa Time (UTC+3); using UTC would pick the wrong
    # calendar day for a few hours around midnight.
    from datetime import timezone
    EAT = timezone(timedelta(hours=3))
    today = datetime.now(EAT).date()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    out_dir = Path(OUTPUT_DIR) / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"news_{from_date}_to_{to_date}.jsonl"

    aggregated = []
    # start from URLs seen in previous runs so we don't re-emit old news
    persisted = _load_seen() if persist_dedup else set()
    seen_urls = set(persisted)
    new_keys: set[str] = set()

    def _add(items):
        for it in items:
            url = it.get("url") or ""
            # dedup by URL; fall back to title for URL-less items
            key = url or ("title:" + (it.get("title") or ""))
            if key and key not in seen_urls:
                aggregated.append(it)
                seen_urls.add(key)
                new_keys.add(key)

    # 1) NewsAPI per company — only when a key is configured (else skip the 46
    #    warning-generating no-op calls entirely).
    if NEWS_API_KEY:
        for i, comp in enumerate(NSE_COMPANIES, start=1):
            name, ticker = comp.get("name"), comp.get("ticker")
            logger.info("[%d/%d] NewsAPI %s...", i, total, name)
            try:
                _add(fetch_news_api(name, ticker, from_date, to_date))
            except Exception as e:
                logger.exception("Error fetching for %s: %s", name, e)
            time.sleep(0.5)
    else:
        logger.info("No NEWS_API_KEY — using keyless RSS + scrapers only.")

    # 2) Keyless Kenyan news RSS feeds (primary source without an API key).
    try:
        _add(scrape_rss_feeds())
    except Exception:
        logger.exception("Error collecting RSS feeds")

    # 3) Business Daily listing scrape (best-effort; markup-fragile).
    try:
        _add(scrape_business_daily(max_pages=5))
    except Exception:
        logger.exception("Error scraping Business Daily")

    # 4) Fetch full article bodies for the NSE/macro-relevant items (RSS only
    #    gives a summary). Gated by is_relevant so we don't fetch every headline.
    if fetch_bodies:
        try:
            from extractor import is_relevant
            enrich_articles_with_body(aggregated, should_fetch=is_relevant, limit=body_limit)
        except Exception:
            logger.exception("Error enriching article bodies")

    # write file
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for a in aggregated:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        logger.info("Saved %d NEW articles (%d already seen in prior runs) to %s",
                    len(aggregated), len(persisted), out_file)
    except Exception as e:
        logger.exception("Failed to save news jsonl: %s", e)

    # persist the updated seen-set so the next run skips these
    if persist_dedup and new_keys:
        _save_seen(persisted | new_keys)

    return aggregated
