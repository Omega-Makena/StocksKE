"""
backtest.py — Evaluate model prediction accuracy against realised price moves.

Usage:
    # Run on real labeled data produced by aligner.align():
    python backtest.py --labeled nse_dataset/labeled/labeled_20240301T120000.jsonl

    # Run on ALL labeled files in the default output dir:
    python backtest.py

    # Generate synthetic data and demo the report (no real data needed):
    python backtest.py --demo
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DIRECTIONS = ("UP", "DOWN", "NEUTRAL")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                logger.warning("Skipping malformed line in %s", path.name)


def load_labeled_rows(path: Path) -> list[dict]:
    rows = []
    for item in _iter_jsonl(path):
        # Each item is one labeled row from aligner.align()
        if isinstance(item, dict) and "predicted_direction" in item:
            rows.append(item)
    return rows


def load_all_labeled(output_dir: Path) -> list[dict]:
    labeled_dir = output_dir / "labeled"
    if not labeled_dir.exists():
        logger.error("No 'labeled' directory found under %s", output_dir)
        return []
    rows = []
    for f in sorted(labeled_dir.glob("labeled_*.jsonl")):
        batch = load_labeled_rows(f)
        logger.info("Loaded %d rows from %s", len(batch), f.name)
        rows.extend(batch)
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def compute_metrics(rows: list[dict]) -> dict:
    """
    Returns a nested dict with:
      - overall accuracy
      - per-class precision / recall / f1
      - accuracy broken down by confidence band, impact_type, ticker
    """
    total = len(rows)
    if total == 0:
        return {"error": "no rows"}

    # --- overall ---
    correct = sum(1 for r in rows if r.get("correct"))
    overall_accuracy = _safe_div(correct, total)

    # --- per-class confusion matrix ---
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for r in rows:
        pred = r.get("predicted_direction")
        actual = r.get("price_label")
        if pred not in DIRECTIONS or actual not in DIRECTIONS:
            continue
        for cls in DIRECTIONS:
            if pred == cls and actual == cls:
                tp[cls] += 1
            elif pred == cls and actual != cls:
                fp[cls] += 1
            elif pred != cls and actual == cls:
                fn[cls] += 1

    per_class: dict[str, dict] = {}
    for cls in DIRECTIONS:
        precision = _safe_div(tp[cls], tp[cls] + fp[cls])
        recall    = _safe_div(tp[cls], tp[cls] + fn[cls])
        f1        = _safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   tp[cls] + fn[cls],
        }

    # --- by confidence band ---
    bands = {"low (<0.50)": [], "medium (0.50-0.70)": [], "high (>0.70)": []}
    for r in rows:
        conf = r.get("predicted_confidence")
        if not isinstance(conf, (int, float)):
            continue
        if conf < 0.50:
            bands["low (<0.50)"].append(r)
        elif conf <= 0.70:
            bands["medium (0.50-0.70)"].append(r)
        else:
            bands["high (>0.70)"].append(r)

    by_confidence = {
        band: {
            "accuracy": round(_safe_div(sum(r["correct"] for r in rs), len(rs)), 4),
            "n": len(rs),
        }
        for band, rs in bands.items()
    }

    # --- by impact_type ---
    by_impact: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        it = r.get("impact_type") or "unknown"
        by_impact[it]["total"] += 1
        if r.get("correct"):
            by_impact[it]["correct"] += 1

    by_impact_summary = {
        k: {
            "accuracy": round(_safe_div(v["correct"], v["total"]), 4),
            "n": v["total"],
        }
        for k, v in sorted(by_impact.items())
    }

    # --- by ticker (top 15 by volume) ---
    by_ticker: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        t = r.get("ticker") or "unknown"
        by_ticker[t]["total"] += 1
        if r.get("correct"):
            by_ticker[t]["correct"] += 1

    by_ticker_summary = {
        k: {
            "accuracy": round(_safe_div(v["correct"], v["total"]), 4),
            "n": v["total"],
        }
        for k, v in sorted(by_ticker.items(), key=lambda x: -x[1]["total"])[:15]
    }

    return {
        "total_predictions": total,
        "overall_accuracy": round(overall_accuracy, 4),
        "per_class": per_class,
        "by_confidence_band": by_confidence,
        "by_impact_type": by_impact_summary,
        "by_ticker": by_ticker_summary,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    if "error" in metrics:
        print(f"[ERROR] {metrics['error']}")
        return

    sep = "-" * 58
    print("\n" + "=" * 58)
    print("  NSE PREDICTION BACKTEST REPORT")
    print("=" * 58)
    print(f"  Total predictions : {metrics['total_predictions']}")
    print(f"  Overall accuracy  : {metrics['overall_accuracy']:.1%}")
    print(sep)

    print("\n  Per-class metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 50)
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<10} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>9}")

    print(f"\n{sep}")
    print("  Accuracy by confidence band:")
    for band, m in metrics["by_confidence_band"].items():
        bar = "#" * int(m["accuracy"] * 20)
        print(f"  {band:<22}  {m['accuracy']:>6.1%}  n={m['n']:<5}  |{bar:<20}|")

    print(f"\n{sep}")
    print("  Accuracy by impact type:")
    for it, m in metrics["by_impact_type"].items():
        print(f"  {it:<20}  {m['accuracy']:>6.1%}  n={m['n']}")

    print(f"\n{sep}")
    print("  Accuracy by ticker (top 15):")
    for ticker, m in metrics["by_ticker"].items():
        bar = "#" * int(m["accuracy"] * 20)
        print(f"  {ticker:<8}  {m['accuracy']:>6.1%}  n={m['n']:<5}  |{bar:<20}|")

    print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def _generate_synthetic_rows(n: int = 500, seed: int = 42) -> list[dict]:
    """
    Produces synthetic labeled rows that mimic the aligner output.
    Simulates a model with ~58% overall accuracy, higher for high-confidence,
    lower for NEUTRAL, and mixed by ticker.
    """
    rng = random.Random(seed)
    tickers = ["KCB", "EQTY", "SCOM", "ABSA", "NCBA", "BAMB", "EABL", "KQ", "NMG", "CTUM"]
    impact_types = ["direct", "competitor", "sector_spillover", "supplier_chain", "regulatory"]
    rows = []

    for _ in range(n):
        ticker = rng.choice(tickers)
        predicted = rng.choices(DIRECTIONS, weights=[40, 35, 25])[0]
        confidence = rng.uniform(0.30, 0.95)
        impact = rng.choice(impact_types)

        # Simulate accuracy: high-conf correct more often, NEUTRAL harder
        base_correct_prob = 0.55
        if confidence > 0.70:
            base_correct_prob = 0.68
        elif confidence > 0.50:
            base_correct_prob = 0.60
        if predicted == "NEUTRAL":
            base_correct_prob -= 0.10

        if rng.random() < base_correct_prob:
            actual = predicted
        else:
            others = [d for d in DIRECTIONS if d != predicted]
            actual = rng.choice(others)

        rows.append({
            "ticker": ticker,
            "article_date": f"2024-0{rng.randint(1,6)}-{rng.randint(1,28):02d}",
            "predicted_direction": predicted,
            "predicted_confidence": round(confidence, 3),
            "price_label": actual,
            "correct": predicted == actual,
            "impact_type": impact,
            "pct_change": round(rng.uniform(-5, 5), 2),
        })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backtest NSE prediction accuracy")
    parser.add_argument("--labeled", help="Path to a specific labeled JSONL file")
    parser.add_argument(
        "--output-dir",
        default="nse_dataset",
        help="Root output directory (default: nse_dataset). Scans labeled/ subdirectory.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic data (no real data needed).",
    )
    parser.add_argument("--json", action="store_true", help="Print raw metrics JSON instead of formatted report")
    args = parser.parse_args()

    if args.demo:
        logger.info("Generating synthetic demo data...")
        rows = _generate_synthetic_rows(500)
    elif args.labeled:
        rows = load_labeled_rows(Path(args.labeled))
    else:
        rows = load_all_labeled(Path(args.output_dir))

    if not rows:
        print("No labeled data found. Use --demo to test with synthetic data.")
        sys.exit(1)

    metrics = compute_metrics(rows)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics)


if __name__ == "__main__":
    main()
