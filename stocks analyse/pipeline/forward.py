"""
forward.py — forward-accumulation runner for a real, growing backtest.

A single scored run is meaningless on NSE news, because RSS only carries the last
few days and their prediction horizons fall in the future (unscoreable). The fix
is to run this DAILY: each day collects a few new predictions, and each day more
of the OLD predictions have their horizons realise and become scoreable. Over a
few weeks this accumulates a statistically meaningful sample.

Each run does three things:
  1. refresh prices  — incrementally download recent Innova files (skips existing),
                       compile, rebuild prices.csv.
  2. collect + extract NEW news — cross-run dedup means only unseen articles hit
                       the LLM; enriched predictions are persisted (extraction batches).
  3. re-score EVERYTHING — enrich every accumulated prediction and run the honest
                       harness against the refreshed prices, so newly-realised
                       horizons get scored. Reports how the scoreable sample grows.

Schedule it daily (Windows Task Scheduler / cron):
    python forward.py

Flags: --no-prices (skip price refresh), --no-collect (skip LLM), --score-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import OUTPUT_DIR, LOOKAHEAD_DAYS
import collector
import extractor
import validator
import graph as G
import harness

logger = logging.getLogger(__name__)
EAT = timezone(timedelta(hours=3))

DATA = Path(OUTPUT_DIR)
PRICES_CSV = DATA / "prices.csv"
PRICE_XLS_DIR = DATA / "prices"
COMPILED_DIR = DATA / "prices_compiled"
EXTRACTIONS_GLOB = str(DATA / "extractions" / "extractions_*.jsonl")


def refresh_prices(days_history: int = 90) -> None:
    """Incrementally pull recent price lists and rebuild prices.csv."""
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from importer.importer import (
        download_price_lists, compile_securities, build_prices_csv, _load_ticker_map,
    )
    today = datetime.now(EAT).date()
    start = datetime.combine(today - timedelta(days=days_history), datetime.min.time())
    end = datetime.combine(today, datetime.min.time())
    logger.info("Refreshing prices %s .. %s (incremental)", start.date(), end.date())
    download_price_lists(start, end, str(PRICE_XLS_DIR))   # skips files already present
    compile_securities(str(PRICE_XLS_DIR), str(COMPILED_DIR))
    build_prices_csv(str(COMPILED_DIR), str(PRICES_CSV), _load_ticker_map(None))


def collect_and_extract(days_back: int = 7) -> int:
    """Collect new news (deduped across runs) and extract predictions. Returns
    the number of new predictions produced."""
    articles = collector.collect_all(days_back)
    if not articles:
        logger.info("No new articles this run.")
        return 0
    extracted = extractor.extract_all(articles)
    return sum(1 for e in extracted if e.get("prediction"))


def _load_all_predictions() -> list[dict]:
    """Every raw prediction ever extracted (across all batches)."""
    preds = []
    for fp in sorted(glob.glob(EXTRACTIONS_GLOB)):
        try:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    p = item.get("prediction") if isinstance(item, dict) and "prediction" in item else item
                    if isinstance(p, dict):
                        preds.append(p)
        except Exception:
            logger.exception("failed reading %s", fp)
    return preds


def score_all() -> dict:
    """Enrich every accumulated prediction and score against the latest prices."""
    if not PRICES_CSV.exists():
        logger.warning("No prices.csv yet — run a price refresh first.")
        return {}
    g = G.build_default_graph()
    preds = _load_all_predictions()
    enriched = [G.enrich_prediction(p, g) for p in preds]
    events = harness.predictions_to_events(enriched)
    metrics = harness.evaluate(events, str(PRICES_CSV), horizons=(1, 3, 5))
    metrics["_accumulated_predictions"] = len(preds)
    metrics["_events"] = len(events)
    harness.print_report(metrics)
    logger.info("Accumulated %d predictions -> %d events; scoreable per horizon: %s",
                len(preds), len(events),
                {h: r.get("n", 0) for h, r in metrics.get("horizons", {}).items()})
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Forward-accumulation backtest runner")
    ap.add_argument("--no-prices", action="store_true", help="skip price refresh")
    ap.add_argument("--no-collect", action="store_true", help="skip news+LLM")
    ap.add_argument("--score-only", action="store_true", help="only re-score accumulated predictions")
    ap.add_argument("--days-history", type=int, default=90)
    ap.add_argument("--days-back", type=int, default=7)
    args = ap.parse_args()

    if args.score_only:
        score_all()
        return
    if not args.no_prices:
        try:
            refresh_prices(args.days_history)
        except Exception:
            logger.exception("Price refresh failed (continuing with existing prices)")
    if not args.no_collect:
        n = collect_and_extract(args.days_back)
        logger.info("New predictions this run: %d", n)
    score_all()


if __name__ == "__main__":
    main()
