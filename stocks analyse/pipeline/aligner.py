import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def load_prices(csv_path: str) -> dict[tuple[str, str], float]:
    """
    Returns dict: {(ticker, "YYYY-MM-DD"): closing_price}
    Uses "Stock Code" as ticker and "Day's Final Price" as close price.
    Strips whitespace from all values.
    Skips rows where price is missing or non-numeric.
    """
    prices = {}
    if not csv_path:
        logger.error("No CSV path provided to load_prices")
        return prices
    p = Path(csv_path)
    if not p.exists():
        logger.error("Prices CSV not found: %s", csv_path)
        return prices

    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = (row.get("Stock Code") or "").strip()
            date = (row.get("Date") or "").strip()
            price_str = (row.get("Day's Final Price") or "").strip()
            if not (ticker and date and price_str):
                continue
            try:
                price = float(price_str.replace(',', ''))
            except ValueError:
                continue
            prices[(ticker, date)] = price
    return prices


def _date_to_str(d: datetime.date) -> str:
    return d.strftime("%Y-%m-%d")


def parse_date(raw) -> "datetime.date | None":
    """Tolerant date parsing. Accepts 'YYYY-MM-DD', ISO datetimes
    ('2024-06-04T10:00:00Z'), and common variants. Returns None if unusable.
    Production news dates arrive in several shapes; parsing only one silently
    dropped everything else."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # ISO date or datetime: the leading 10 chars are YYYY-MM-DD
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        pass
    for fmt in ("%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def get_price_change(
    prices: dict,
    ticker: str,
    from_date: str,
    lookahead_days: int = 3,
) -> Tuple[float | None, float | None, float | None]:
    """
    Returns (price_t0, price_tn, pct_change).
    price_t0: closing price on from_date (or nearest trading day after)
    price_tn: closing price lookahead_days trading days later
    pct_change: ((price_tn - price_t0) / price_t0) * 100
    Returns (None, None, None) if either price is not found.
    Searches up to 5 calendar days forward from each target date
    to handle weekends and holidays.
    """
    base_date = parse_date(from_date)
    if base_date is None:
        logger.warning("Unparseable article date, skipping: %r", from_date)
        return (None, None, None)

    # find price_t0 searching up to 5 calendar days forward
    price_t0 = None
    t0_date = None
    for delta in range(0, 6):
        d = base_date + timedelta(days=delta)
        key = (ticker, _date_to_str(d))
        if key in prices:
            price_t0 = prices[key]
            t0_date = d
            break
    if price_t0 is None:
        return (None, None, None)

    # desired lookahead target date (calendar days)
    target = t0_date + timedelta(days=lookahead_days)
    price_tn = None
    tn_date = None
    for delta in range(0, 6):
        d = target + timedelta(days=delta)
        key = (ticker, _date_to_str(d))
        if key in prices:
            price_tn = prices[key]
            tn_date = d
            break
    if price_tn is None:
        return (price_t0, None, None)

    try:
        pct = ((price_tn - price_t0) / price_t0) * 100
    except Exception:
        pct = None
    return (price_t0, price_tn, pct)


def generate_label(pct_change: float, threshold: float = 1.5) -> str:
    """
    Returns "UP" if pct_change >= threshold
    Returns "DOWN" if pct_change <= -threshold
    Returns "NEUTRAL" otherwise
    """
    if pct_change is None:
        return "NEUTRAL"
    if pct_change >= threshold:
        return "UP"
    if pct_change <= -threshold:
        return "DOWN"
    return "NEUTRAL"


def align(
    predictions_jsonl,
    prices_csv: str,
    lookahead_days: int = 3,
    threshold: float = 1.5,
) -> list[dict]:
    """
    For each prediction in predictions_jsonl:
      For each ticker in directly_affected + indirectly_affected:
        Look up price change using get_price_change
        Assign price_label using generate_label
        Compare price_label to model's predicted direction
        Set "correct": True/False

    Saves to OUTPUT_DIR/labeled/labeled_{timestamp}.jsonl
    Prints accuracy summary at the end.
    """
    # predictions_jsonl can be a path to a jsonl file or a list of prediction dicts
    preds = []
    if isinstance(predictions_jsonl, (str, Path)):
        p = Path(predictions_jsonl)
        if not p.exists():
            logger.error("Predictions file not found: %s", str(p))
            return []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    preds.append(json.loads(line))
                except Exception:
                    logger.exception("Failed to parse prediction line")
    else:
        preds = list(predictions_jsonl)

    prices = load_prices(prices_csv)
    labeled = []
    total_checks = 0
    correct = 0

    for item in preds:
        # support both formats: raw prediction object or wrapper {"article":..., "prediction":...}
        pred = item.get("prediction") if isinstance(item, dict) and "prediction" in item else item
        if not pred:
            continue
        # lean schema puts article_date at top level; older data nests it in
        # data_quality — support both.
        article_date = pred.get("article_date")
        if not article_date:
            article_date = (pred.get("data_quality") or {}).get("article_date")
        article_date = article_date or None
        for group_name in ("directly_affected", "indirectly_affected"):
            for ent in pred.get(group_name, []) or []:
                t = ent.get("ticker")
                if not t:
                    continue
                direction = ent.get("direction")
                confidence = ent.get("confidence")
                # graph-propagated entities carry a signed magnitude estimate
                predicted_magnitude = ent.get("magnitude_pct")
                price_t0, price_tn, pct = get_price_change(prices, t, article_date or "", lookahead_days=lookahead_days)
                price_label = generate_label(pct, threshold)
                is_correct = (direction == price_label) if (direction in ("UP", "DOWN", "NEUTRAL")) else False
                # absolute error between predicted and realised move (when both known)
                magnitude_error = None
                if isinstance(predicted_magnitude, (int, float)) and isinstance(pct, (int, float)):
                    magnitude_error = abs(float(predicted_magnitude) - float(pct))
                total_checks += 1
                if is_correct:
                    correct += 1
                labeled.append(
                    {
                        "article_date": article_date,
                        "ticker": t,
                        "company": ent.get("company"),
                        "impact_type": ent.get("impact_type"),
                        "predicted_direction": direction,
                        "predicted_confidence": confidence,
                        "predicted_magnitude": predicted_magnitude,
                        "price_t0": price_t0,
                        "price_tn": price_tn,
                        "pct_change": pct,
                        "magnitude_error": magnitude_error,
                        "price_label": price_label,
                        "correct": is_correct,
                    }
                )

    out_dir = Path(OUTPUT_DIR) / "labeled"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_file = out_dir / f"labeled_{ts}.jsonl"
    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for r in labeled:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write labeled outputs")

    accuracy = (correct / total_checks) if total_checks else 0.0
    logger.info("Alignment complete. Total checks=%d, Correct=%d, Accuracy=%.3f", total_checks, correct, accuracy)
    return labeled
