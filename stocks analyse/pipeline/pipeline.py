"""
pipeline.py — End-to-end NSE prediction pipeline.

Data sources:
  • Prices  : compiled_securities/ produced by importer.py (via COMPILED_DIR or --import-from)
  • News    : News API + Business Daily scraper (via collector.py)

Steps:
  0. Build/locate the prices CSV from importer output
  1. Collect news articles
  2. Extract LLM predictions from articles
  3. Validate / filter hallucinations
  4. Align predictions against realised prices → labeled dataset

Usage:
    # Full run — builds prices CSV on-the-fly from compiled_securities/
    python pipeline.py --import-from compiled_securities --prices-csv prices.csv

    # Skip re-downloading news (use last saved batch)
    python pipeline.py --import-from compiled_securities --skip-collect

    # Point at an already-built prices CSV
    python pipeline.py --prices-csv prices.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import (
    COMPILED_DIR,
    LOOKAHEAD_DAYS,
    OUTPUT_DIR,
    PRICE_CHANGE_THRESHOLD,
    PRICE_CSV_PATH,
)
from collector import collect_all
from extractor import extract_all
import validator
from aligner import align
from companies import NAME_MAP
from graph import build_graph, enrich_prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latest_news_jsonl(news_dir: Path) -> list[dict]:
    files = list(news_dir.glob("news_*.jsonl"))
    if not files:
        return []
    latest = max(files, key=lambda p: p.stat().st_mtime)
    articles = []
    with latest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                articles.append(json.loads(line))
            except Exception:
                logger.exception("Failed to parse news line in %s", latest)
    logger.info("Loaded %d articles from %s", len(articles), latest)
    return articles


def _build_ticker_map() -> dict[str, str]:
    """Return {normalised_company_name_lower: ticker} from the local companies module."""
    return {name.lower(): ticker for ticker, name in NAME_MAP.items()}


def _ensure_prices_csv(compiled_dir: str, prices_csv: str) -> str:
    """
    Build (or verify) the unified prices CSV from importer's compiled_securities dir.
    Returns the path to the CSV to use.
    Raises RuntimeError if neither source yields a usable CSV.
    """
    # Add the project root to path so we can import the importer package
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from importer.importer import build_prices_csv
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import importer/importer.py — make sure it is in the project root."
        ) from exc

    ticker_map = _build_ticker_map()
    out_path = build_prices_csv(compiled_dir, prices_csv, ticker_map)
    return str(out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    days_back: int = 30,
    skip_collection: bool = False,
    import_from: str = "",
    prices_csv_override: str = "",
) -> list[dict]:
    """
    Execute the full pipeline and return the labeled dataset.

    Parameters
    ----------
    days_back          : how many days of news to collect
    skip_collection    : if True, load news from the last saved JSONL instead of fetching
    import_from        : path to compiled_securities dir from importer.py (overrides COMPILED_DIR)
    prices_csv_override: explicit path to prices CSV (skips the importer build step)
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Fail loudly on obvious misconfiguration rather than silently producing nothing.
    import config as _cfg
    for problem in _cfg.validate_config():
        logger.warning("CONFIG: %s", problem)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: resolve prices CSV
    # ------------------------------------------------------------------
    compiled_source = import_from or COMPILED_DIR
    prices_csv = prices_csv_override or PRICE_CSV_PATH

    if compiled_source:
        auto_csv = str(out_dir / "prices.csv")
        logger.info("Step 0: building prices CSV from %s → %s", compiled_source, auto_csv)
        try:
            prices_csv = _ensure_prices_csv(compiled_source, auto_csv)
        except Exception:
            logger.exception("Failed to build prices CSV; will attempt alignment without prices")
    elif not prices_csv:
        logger.warning(
            "No price data source configured. "
            "Set COMPILED_DIR env var or pass --import-from / --prices-csv."
        )

    # ------------------------------------------------------------------
    # Step 1: collect news
    # ------------------------------------------------------------------
    if skip_collection:
        news_dir = out_dir / "news"
        if not news_dir.exists():
            logger.error("No news directory found and --skip-collect requested")
            return []
        articles = _load_latest_news_jsonl(news_dir)
    else:
        logger.info("Step 1: collecting news (last %d days)", days_back)
        articles = collect_all(days_back)

    logger.info("Articles collected: %d", len(articles))
    if not articles:
        logger.warning("No articles collected — nothing to predict")
        return []

    # ------------------------------------------------------------------
    # Step 2: extract predictions via LLM
    # ------------------------------------------------------------------
    logger.info("Step 2: extracting predictions from %d articles", len(articles))
    extracted = extract_all(articles)
    logger.info("Predictions generated: %d", len(extracted))

    # ------------------------------------------------------------------
    # Step 3: validate / filter hallucinations
    # ------------------------------------------------------------------
    predictions = [e.get("prediction") for e in extracted if e.get("prediction")]
    valid, invalid = validator.filter_predictions(predictions)
    logger.info(
        "Step 3: valid=%d  invalid/hallucinations=%d",
        len(valid), len(invalid),
    )

    # ------------------------------------------------------------------
    # Step 3.5: propagate impact through the knowledge graph
    #   Replaces the LLM's guessed indirectly_affected with graph-derived
    #   impacts (direction + magnitude % + confidence) seeded from the source
    #   event (directly_affected + source_entities). The graph is the curated
    #   seed PLUS data-derived edges (price co-movement + article co-occurrence).
    # ------------------------------------------------------------------
    extraction_glob = str(out_dir / "extractions" / "extractions_*.jsonl")
    graph = build_graph(
        prices_csv=prices_csv or None,
        extraction_paths=[extraction_glob],
    )
    # Be explicit about calibration status — default coefficients are NOT fitted
    # to NSE data, so magnitudes are indicative only until calibrate.py has run.
    import graph as _g
    if not _g.CALIBRATION_FILE.exists():
        logger.warning(
            "Graph is running on DEFAULT (uncalibrated) coefficients — magnitudes "
            "are indicative only. Run calibrate.py on real labeled data to fit them."
        )
    reached_total = 0
    for i, pred in enumerate(valid):
        enriched = enrich_prediction(pred, graph)
        valid[i] = enriched
        reached_total += enriched.get("propagation", {}).get("reached", 0)
    logger.info(
        "Step 3.5: graph propagation enriched %d predictions (%d downstream impacts)",
        len(valid), reached_total,
    )

    # ------------------------------------------------------------------
    # Step 3.6: build event alerts — the product the evidence supports
    #   (exposure map + validated move-likelihood ranking; direction is
    #   informational only, not a prediction). Saves a ranked analyst feed.
    # ------------------------------------------------------------------
    try:
        import alert as _alert
        alerts = [_alert.build_alert(p) for p in valid]
        alerts = [a for a in alerts if a["exposed_count"]]
        alerts.sort(key=lambda a: (a["names"][0]["move_likelihood"] if a["names"] else 0), reverse=True)
        adir = out_dir / "alerts"
        adir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        with (adir / f"alerts_{ts}.jsonl").open("w", encoding="utf-8") as fh:
            for a in alerts:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        top = [a for a in alerts if a["top_tier"] in ("HIGH", "MEDIUM")][:10]
        if top:
            logger.info("Step 3.6: %d alerts (%d HIGH/MEDIUM). Top of the feed:", len(alerts), len(top))
            for a in top:
                print(_alert.render_alert(a, top=5))
        else:
            logger.info("Step 3.6: %d alerts built (none reached HIGH/MEDIUM tier)", len(alerts))
    except Exception:
        logger.exception("Alert generation failed (pipeline result unaffected)")

    # ------------------------------------------------------------------
    # Step 4: align predictions against realised prices
    # ------------------------------------------------------------------
    logger.info("Step 4: aligning %d predictions against prices CSV: %s", len(valid), prices_csv)
    labeled = align(
        valid,
        prices_csv,
        lookahead_days=LOOKAHEAD_DAYS,
        threshold=PRICE_CHANGE_THRESHOLD,
    )
    logger.info("Labeled examples created: %d", len(labeled))

    # summary
    dist: dict[str, int] = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    for p in valid:
        for g in (p.get("directly_affected") or []) + (p.get("indirectly_affected") or []):
            d = g.get("direction")
            if d in dist:
                dist[d] += 1
    logger.info("Predicted direction distribution: %s", dist)

    if labeled:
        correct = sum(1 for r in labeled if r.get("correct"))
        logger.info(
            "Accuracy on labeled set: %.1f%%  (%d / %d)  [naive: raw change, single horizon]",
            100 * correct / len(labeled), correct, len(labeled),
        )

    # ------------------------------------------------------------------
    # Step 5: honest event-study scorecard (abnormal returns vs baselines)
    #   This is the trustworthy number. The naive accuracy above is a smoke
    #   test; harness.evaluate is point-in-time, market-adjusted, liquidity-
    #   filtered, and compared against baselines.
    # ------------------------------------------------------------------
    if prices_csv and Path(prices_csv).exists():
        try:
            import harness
            events = harness.predictions_to_events(valid)
            metrics = harness.evaluate(events, prices_csv, horizons=(1, 3, 5))
            harness.print_report(metrics)
        except Exception:
            logger.exception("Event-study scorecard failed (pipeline result is unaffected)")
    else:
        logger.info("Step 5: skipped event-study scorecard (no prices CSV available)")

    return labeled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NSE prediction pipeline")

    parser.add_argument(
        "--import-from",
        default="",
        metavar="DIR",
        help=(
            "Path to the compiled_securities/ folder produced by importer.py. "
            "The pipeline will auto-build a prices CSV from it. "
            "Overrides the COMPILED_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--prices-csv",
        default="",
        metavar="PATH",
        help=(
            "Explicit path to an already-built prices CSV. "
            "Skips the importer build step entirely."
        ),
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="How many days of news to collect (default: 30)",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Load news from the last saved JSONL instead of fetching from APIs",
    )

    args = parser.parse_args()
    run(
        days_back=args.days,
        skip_collection=args.skip_collect,
        import_from=args.import_from,
        prices_csv_override=args.prices_csv,
    )


if __name__ == "__main__":
    main()
