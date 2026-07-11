import logging
from typing import List

from companies import VALID_TICKERS

logger = logging.getLogger(__name__)
VALID_DIRECTIONS = {"UP", "DOWN", "NEUTRAL"}
VALID_IMPACT_TYPES = {"direct", "competitor", "sector_spillover", "supplier_chain", "regulatory"}
VALID_EVENT_TYPES = {
    "earnings",
    "regulation",
    "product_launch",
    "disaster",
    "merger_acquisition",
    "macro",
    "commodity",
    "legal",
    "management_change",
    "other",
}


def validate(prediction: dict) -> List[str]:
    """
    Returns list of error strings. Empty list = valid.
    Checks:
    - All tickers in directly_affected and indirectly_affected are in VALID_TICKERS
    - All direction values are in VALID_DIRECTIONS
    - All impact_type values are in VALID_IMPACT_TYPES
    - event_type is in VALID_EVENT_TYPES
    - All confidence values are floats in [0.0, 1.0]
    - No ticker appears in both directly_affected and indirectly_affected
    """
    errors = []
    if not isinstance(prediction, dict):
        errors.append("prediction is not a dict")
        return errors

    event_type = prediction.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        errors.append(f"invalid event_type: {event_type}")

    dirs = []
    for group in ("directly_affected", "indirectly_affected"):
        entries = prediction.get(group) or []
        if not isinstance(entries, list):
            errors.append(f"{group} is not a list")
            continue
        for ent in entries:
            t = ent.get("ticker")
            if not t or t not in VALID_TICKERS:
                errors.append(f"invalid ticker in {group}: {t}")
            d = ent.get("direction")
            if d not in VALID_DIRECTIONS:
                errors.append(f"invalid direction for {t}: {d}")
            it = ent.get("impact_type")
            if it not in VALID_IMPACT_TYPES:
                errors.append(f"invalid impact_type for {t}: {it}")
            conf = ent.get("confidence")
            if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
                errors.append(f"invalid confidence for {t}: {conf}")
            dirs.append(t)

    # check duplicates across groups
    direct = {e.get("ticker") for e in (prediction.get("directly_affected") or [])}
    indirect = {e.get("ticker") for e in (prediction.get("indirectly_affected") or [])}
    overlap = direct.intersection(indirect)
    if overlap:
        errors.append(f"ticker(s) appear in both direct and indirect: {sorted(list(overlap))}")

    return errors


def filter_predictions(predictions: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Splits predictions into (valid, invalid) using validate().
    Logs invalid predictions with their error messages.
    """
    valid = []
    invalid = []
    for p in predictions:
        errs = validate(p)
        if not errs:
            valid.append(p)
        else:
            invalid.append({"prediction": p, "errors": errs})
            logger.error("Invalid prediction: %s; errors: %s", p.get("article_summary", "<no summary>"), errs)
    return valid, invalid


if __name__ == "__main__":
    # simple unit test to ensure hallucinated ticker is caught
    import unittest


    class ValidatorTests(unittest.TestCase):
        def test_hallucinated_ticker_is_caught(self):
            fake = {
                "article_summary": "Fake news",
                "event_type": "earnings",
                "primary_sector": "Banking",
                "directly_affected": [
                    {
                        "ticker": "FAKE",
                        "company": "Fake Co",
                        "impact_type": "direct",
                        "direction": "UP",
                        "confidence": 0.9,
                        "reasoning": "made up",
                    }
                ],
                "indirectly_affected": [],
                "not_nse_listed": [],
                "macro_flags": {"currency_risk": False, "interest_rate_sensitive": False, "commodity_price_sensitive": False, "regulatory_change": False},
                "data_quality": {"article_date": "2024-01-01", "source_credibility": "low", "ambiguity_notes": ""},
            }
            errs = validate(fake)
            self.assertTrue(any("invalid ticker" in e or "invalid ticker in" in e for e in errs), f"Expected invalid ticker error, got: {errs}")


    unittest.main()
