"""Tests for the forward-accumulation runner (pure-logic parts)."""
import os
import sys
import json
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import forward


def _write_batch(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({"article": {}, "prediction": p}) + "\n")


def test_load_all_predictions_reads_every_batch(tmp_path):
    ext = tmp_path / "extractions"
    ext.mkdir()
    _write_batch(ext / "extractions_1.jsonl", [{"event_type": "earnings", "directly_affected": [{"ticker": "KCB"}]}])
    _write_batch(ext / "extractions_2.jsonl", [{"event_type": "macro"}, {"event_type": "disaster"}])
    with patch.object(forward, "EXTRACTIONS_GLOB", str(ext / "extractions_*.jsonl")):
        preds = forward._load_all_predictions()
    assert len(preds) == 3                      # accumulates ACROSS batches
    assert {p["event_type"] for p in preds} == {"earnings", "macro", "disaster"}


def test_load_all_predictions_empty_when_none(tmp_path):
    with patch.object(forward, "EXTRACTIONS_GLOB", str(tmp_path / "none_*.jsonl")):
        assert forward._load_all_predictions() == []


def test_score_all_without_prices_is_safe(tmp_path):
    # no prices.csv -> returns empty, does not raise
    with patch.object(forward, "PRICES_CSV", tmp_path / "missing.csv"):
        assert forward.score_all() == {}
