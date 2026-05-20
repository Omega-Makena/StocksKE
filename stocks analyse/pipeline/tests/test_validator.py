"""Tests for nse_predictor/validator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from validator import validate, filter_predictions


def _valid_prediction(**overrides):
    base = {
        "article_summary": "KCB posts strong earnings.",
        "event_type": "earnings",
        "primary_sector": "Banking",
        "directly_affected": [
            {
                "ticker": "KCB",
                "company": "KCB Group Plc",
                "impact_type": "direct",
                "direction": "UP",
                "confidence": 0.85,
                "reasoning": "Strong earnings beat.",
            }
        ],
        "indirectly_affected": [
            {
                "ticker": "EQTY",
                "company": "Equity Group Holdings Plc",
                "impact_type": "competitor",
                "direction": "NEUTRAL",
                "confidence": 0.40,
                "reasoning": "Competitor effect.",
            }
        ],
        "not_nse_listed": [],
        "macro_flags": {
            "currency_risk": False,
            "interest_rate_sensitive": False,
            "commodity_price_sensitive": False,
            "regulatory_change": False,
        },
        "data_quality": {
            "article_date": "2024-03-01",
            "source_credibility": "high",
            "ambiguity_notes": "",
        },
    }
    base.update(overrides)
    return base


class TestValidate:
    def test_valid_prediction_returns_no_errors(self):
        assert validate(_valid_prediction()) == []

    def test_hallucinated_ticker_is_caught(self):
        pred = _valid_prediction(
            directly_affected=[
                {
                    "ticker": "FAKE",
                    "company": "Fake Co",
                    "impact_type": "direct",
                    "direction": "UP",
                    "confidence": 0.9,
                    "reasoning": "made up",
                }
            ]
        )
        errs = validate(pred)
        assert any("FAKE" in e for e in errs)

    def test_invalid_direction_is_caught(self):
        pred = _valid_prediction(
            directly_affected=[
                {
                    "ticker": "KCB",
                    "company": "KCB Group Plc",
                    "impact_type": "direct",
                    "direction": "SIDEWAYS",
                    "confidence": 0.7,
                    "reasoning": "bad direction",
                }
            ]
        )
        errs = validate(pred)
        assert any("direction" in e for e in errs)

    def test_invalid_impact_type_is_caught(self):
        pred = _valid_prediction(
            directly_affected=[
                {
                    "ticker": "KCB",
                    "company": "KCB Group Plc",
                    "impact_type": "rumour",
                    "direction": "UP",
                    "confidence": 0.7,
                    "reasoning": "bad impact_type",
                }
            ]
        )
        errs = validate(pred)
        assert any("impact_type" in e for e in errs)

    def test_invalid_event_type_is_caught(self):
        pred = _valid_prediction(event_type="gossip")
        errs = validate(pred)
        assert any("event_type" in e for e in errs)

    def test_confidence_out_of_range_is_caught(self):
        pred = _valid_prediction(
            directly_affected=[
                {
                    "ticker": "KCB",
                    "company": "KCB Group Plc",
                    "impact_type": "direct",
                    "direction": "UP",
                    "confidence": 1.5,
                    "reasoning": "over-confident",
                }
            ]
        )
        errs = validate(pred)
        assert any("confidence" in e for e in errs)

    def test_confidence_zero_is_valid(self):
        pred = _valid_prediction(
            directly_affected=[
                {
                    "ticker": "KCB",
                    "company": "KCB Group Plc",
                    "impact_type": "direct",
                    "direction": "UP",
                    "confidence": 0.0,
                    "reasoning": "unsure",
                }
            ]
        )
        errs = validate(pred)
        assert errs == []

    def test_ticker_in_both_groups_is_caught(self):
        pred = _valid_prediction(
            directly_affected=[
                {"ticker": "KCB", "company": "KCB", "impact_type": "direct", "direction": "UP", "confidence": 0.8, "reasoning": ""}
            ],
            indirectly_affected=[
                {"ticker": "KCB", "company": "KCB", "impact_type": "competitor", "direction": "NEUTRAL", "confidence": 0.4, "reasoning": ""}
            ],
        )
        errs = validate(pred)
        assert any("both" in e for e in errs)

    def test_non_dict_prediction_returns_error(self):
        errs = validate("not a dict")
        assert any("not a dict" in e for e in errs)

    def test_multiple_errors_returned(self):
        pred = _valid_prediction(
            event_type="bad_type",
            directly_affected=[
                {"ticker": "NOTREAL", "company": "X", "impact_type": "rumour", "direction": "SIDEWAYS", "confidence": 99, "reasoning": ""}
            ],
        )
        errs = validate(pred)
        assert len(errs) >= 3


class TestFilterPredictions:
    def test_valid_passes_through(self):
        valid, invalid = filter_predictions([_valid_prediction()])
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_invalid_is_separated(self):
        bad = _valid_prediction(event_type="nonsense")
        valid, invalid = filter_predictions([_valid_prediction(), bad])
        assert len(valid) == 1
        assert len(invalid) == 1

    def test_empty_input(self):
        valid, invalid = filter_predictions([])
        assert valid == []
        assert invalid == []

    def test_all_invalid(self):
        preds = [_valid_prediction(event_type="bad") for _ in range(3)]
        valid, invalid = filter_predictions(preds)
        assert valid == []
        assert len(invalid) == 3
