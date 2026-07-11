"""Tests for nse_predictor/aligner.py"""
import csv
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import date
from aligner import load_prices, get_price_change, generate_label, align, parse_date


class TestParseDate:
    def test_iso_date(self):
        assert parse_date("2024-06-04") == date(2024, 6, 4)

    def test_iso_datetime_is_accepted(self):
        # NewsAPI-style timestamps must not be silently dropped
        assert parse_date("2024-06-04T10:30:00Z") == date(2024, 6, 4)
        assert parse_date("2024-06-04T10:30:00+03:00") == date(2024, 6, 4)

    def test_day_month_year(self):
        assert parse_date("04-June-2024") == date(2024, 6, 4)

    def test_garbage_and_empty_return_none(self):
        assert parse_date("not a date") is None
        assert parse_date("") is None
        assert parse_date(None) is None


def _make_prices_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["Stock Code", "Date", "Day's Final Price"])
    writer.writeheader()
    writer.writerows(rows)
    f.close()
    return f.name


SAMPLE_CSV_ROWS = [
    {"Stock Code": "KCB",  "Date": "2024-03-01", "Day's Final Price": "45.00"},
    {"Stock Code": "KCB",  "Date": "2024-03-04", "Day's Final Price": "47.25"},
    {"Stock Code": "KCB",  "Date": "2024-03-05", "Day's Final Price": "43.00"},
    {"Stock Code": "EQTY", "Date": "2024-03-01", "Day's Final Price": "52.50"},
    {"Stock Code": "EQTY", "Date": "2024-03-04", "Day's Final Price": "52.50"},
]


class TestLoadPrices:
    def test_loads_correct_structure(self):
        path = _make_prices_csv(SAMPLE_CSV_ROWS)
        prices = load_prices(path)
        assert ("KCB", "2024-03-01") in prices
        assert prices[("KCB", "2024-03-01")] == 45.0

    def test_skips_non_numeric_price(self):
        path = _make_prices_csv([
            {"Stock Code": "KCB", "Date": "2024-03-01", "Day's Final Price": "N/A"},
        ])
        prices = load_prices(path)
        assert ("KCB", "2024-03-01") not in prices

    def test_handles_comma_formatted_numbers(self):
        path = _make_prices_csv([
            {"Stock Code": "SCOM", "Date": "2024-03-01", "Day's Final Price": "1,234.50"},
        ])
        prices = load_prices(path)
        assert prices[("SCOM", "2024-03-01")] == pytest.approx(1234.50)

    def test_returns_empty_for_missing_file(self):
        prices = load_prices("/nonexistent/path.csv")
        assert prices == {}

    def test_returns_empty_for_empty_path(self):
        prices = load_prices("")
        assert prices == {}

    def test_skips_rows_with_missing_fields(self):
        path = _make_prices_csv([
            {"Stock Code": "", "Date": "2024-03-01", "Day's Final Price": "45.0"},
        ])
        prices = load_prices(path)
        assert len(prices) == 0


class TestGetPriceChange:
    def setup_method(self):
        path = _make_prices_csv(SAMPLE_CSV_ROWS)
        self.prices = load_prices(path)

    def test_returns_correct_pct_change(self):
        t0, tn, pct = get_price_change(self.prices, "KCB", "2024-03-01", lookahead_days=3)
        assert t0 == pytest.approx(45.0)
        assert tn == pytest.approx(47.25)
        assert pct == pytest.approx((47.25 - 45.0) / 45.0 * 100)

    def test_returns_none_when_start_price_missing(self):
        t0, tn, pct = get_price_change(self.prices, "KCB", "2024-01-01", lookahead_days=3)
        assert t0 is None
        assert tn is None
        assert pct is None

    def test_returns_partial_when_lookahead_price_missing(self):
        t0, tn, pct = get_price_change(self.prices, "EQTY", "2024-03-01", lookahead_days=3)
        # EQTY has no data after 2024-03-04, so searching +3 calendar days = 2024-03-04 which IS present
        assert t0 == pytest.approx(52.50)

    def test_skips_weekend_to_find_next_trading_day(self):
        # 2024-03-02 is a Saturday; should find 2024-03-04 (Monday)
        t0, tn, pct = get_price_change(self.prices, "KCB", "2024-03-02", lookahead_days=3)
        assert t0 == pytest.approx(47.25)

    def test_returns_none_for_invalid_date_format(self):
        t0, tn, pct = get_price_change(self.prices, "KCB", "not-a-date", lookahead_days=3)
        assert (t0, tn, pct) == (None, None, None)

    def test_returns_none_for_unknown_ticker(self):
        t0, tn, pct = get_price_change(self.prices, "FAKE", "2024-03-01", lookahead_days=3)
        assert t0 is None


class TestGenerateLabel:
    def test_above_threshold_is_up(self):
        assert generate_label(2.0, threshold=1.5) == "UP"

    def test_exactly_threshold_is_up(self):
        assert generate_label(1.5, threshold=1.5) == "UP"

    def test_below_negative_threshold_is_down(self):
        assert generate_label(-2.0, threshold=1.5) == "DOWN"

    def test_exactly_negative_threshold_is_down(self):
        assert generate_label(-1.5, threshold=1.5) == "DOWN"

    def test_within_threshold_is_neutral(self):
        assert generate_label(0.5, threshold=1.5) == "NEUTRAL"
        assert generate_label(-1.4, threshold=1.5) == "NEUTRAL"
        assert generate_label(0.0, threshold=1.5) == "NEUTRAL"

    def test_none_pct_change_is_neutral(self):
        assert generate_label(None, threshold=1.5) == "NEUTRAL"


class TestAlign:
    def _make_prediction(self, ticker, direction, article_date="2024-03-01"):
        return {
            "event_type": "earnings",
            "data_quality": {"article_date": article_date, "source_credibility": "high", "ambiguity_notes": ""},
            "directly_affected": [
                {
                    "ticker": ticker,
                    "company": "Test Co",
                    "impact_type": "direct",
                    "direction": direction,
                    "confidence": 0.8,
                }
            ],
            "indirectly_affected": [],
        }

    def test_correct_prediction_marked_correct(self):
        csv_path = _make_prices_csv(SAMPLE_CSV_ROWS)
        pred = self._make_prediction("KCB", "UP", "2024-03-01")
        labeled = align([pred], csv_path, lookahead_days=3, threshold=1.5)
        kcb_row = next(r for r in labeled if r["ticker"] == "KCB")
        assert kcb_row["correct"] is True

    def test_wrong_prediction_marked_incorrect(self):
        csv_path = _make_prices_csv(SAMPLE_CSV_ROWS)
        pred = self._make_prediction("KCB", "DOWN", "2024-03-01")
        labeled = align([pred], csv_path, lookahead_days=3, threshold=1.5)
        kcb_row = next(r for r in labeled if r["ticker"] == "KCB")
        assert kcb_row["correct"] is False

    def test_returns_empty_for_empty_predictions(self):
        csv_path = _make_prices_csv(SAMPLE_CSV_ROWS)
        labeled = align([], csv_path)
        assert labeled == []

    def test_missing_price_data_yields_neutral_label(self):
        csv_path = _make_prices_csv(SAMPLE_CSV_ROWS)
        pred = self._make_prediction("KCB", "NEUTRAL", "2020-01-01")
        labeled = align([pred], csv_path)
        row = next(r for r in labeled if r["ticker"] == "KCB")
        assert row["price_label"] == "NEUTRAL"

    def test_wrapper_format_is_handled(self):
        csv_path = _make_prices_csv(SAMPLE_CSV_ROWS)
        pred = self._make_prediction("KCB", "UP", "2024-03-01")
        wrapped = {"article": {"title": "test"}, "prediction": pred}
        labeled = align([wrapped], csv_path, lookahead_days=3, threshold=1.5)
        assert any(r["ticker"] == "KCB" for r in labeled)
