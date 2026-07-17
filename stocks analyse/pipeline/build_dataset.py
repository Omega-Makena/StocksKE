"""
build_dataset.py — assemble a real-sized labeled set for evaluating accuracy.

RSS is current-only (horizons in the future = unscoreable), which is why we can
only ever score a handful of events. NewsAPI returns ~3-4 weeks of HISTORY, so
news from that window has fully-realised price horizons. This pulls per-company
historical news over a window whose horizons land inside the price history,
extracts with the configured LLM, and saves extraction batches for scoring.

    python build_dataset.py --from 2026-06-20 --to 2026-07-08

Score afterwards with forward.py --score-only (or the signal analysis).
"""
from __future__ import annotations

import argparse
import logging

from companies import NSE_COMPANIES
import collector
import extractor

logger = logging.getLogger(__name__)


def build(from_date: str, to_date: str, max_per_company: int = 100) -> int:
    seen = set()
    articles = []
    for i, c in enumerate(NSE_COMPANIES, 1):
        name, ticker = c["name"], c["ticker"]
        logger.info("[%d/%d] NewsAPI history for %s (%s)", i, len(NSE_COMPANIES), name, ticker)
        try:
            items = collector.fetch_news_api(name, ticker, from_date, to_date, page_size=max_per_company)
        except Exception:
            logger.exception("fetch failed for %s", name)
            continue
        for it in items:
            url = it.get("url") or it.get("title") or ""
            if url and url not in seen:
                seen.add(url)
                articles.append(it)
    logger.info("Collected %d unique historical articles", len(articles))

    # enrich bodies for relevant ones (NewsAPI 'content' is truncated)
    try:
        collector.enrich_articles_with_body(articles, should_fetch=extractor.is_relevant, limit=400)
    except Exception:
        logger.exception("body enrich failed")

    extracted = extractor.extract_all(articles)   # prefilters + saves a batch
    n = sum(1 for e in extracted if e.get("prediction"))
    logger.info("Extracted %d predictions from the historical window", n)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--max-per-company", type=int, default=100)
    args = ap.parse_args()
    build(args.from_date, args.to_date, args.max_per_company)


if __name__ == "__main__":
    main()
