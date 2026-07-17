"""Tests for the event-alert / move-likelihood product."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import alert


class TestMoveLikelihood:
    def test_eventful_beats_macro(self):
        # a severe M&A is more move-likely than routine macro news
        hi = alert.move_likelihood("merger_acquisition", severity=1.0, confidence=0.3)
        lo = alert.move_likelihood("macro", severity=0.3, confidence=0.05)
        assert hi > lo

    def test_macro_sits_near_base(self):
        # macro prior == base rate, so score never lifts above base
        s = alert.move_likelihood("macro", severity=1.0, confidence=0.3)
        assert abs(s - alert.BASE_MOVE_RATE) < 1e-9

    def test_severity_and_confidence_increase_score(self):
        base = alert.move_likelihood("earnings", severity=0.2, confidence=0.05)
        more_sev = alert.move_likelihood("earnings", severity=0.9, confidence=0.05)
        more_conf = alert.move_likelihood("earnings", severity=0.2, confidence=0.3)
        assert more_sev > base and more_conf > base

    def test_bounded_between_base_and_prior(self):
        for et in ("earnings", "disaster", "regulation"):
            prior = alert.EVENT_MOVE_PRIOR[et]
            s = alert.move_likelihood(et, 1.0, 1.0)
            assert alert.BASE_MOVE_RATE <= s <= prior + 1e-9


class TestBuildAlert:
    def _enriched(self):
        return {
            "event_type": "disaster", "severity": 1.0, "article_date": "2026-07-01",
            "source_entities": [{"name": "Boeing 737 MAX"}],
            "exposed": [
                {"ticker": "KQ", "direction": "DOWN", "confidence": 0.3, "impact_type": "supplier_chain", "reasoning": "via product"},
                {"ticker": "TOTL", "direction": "NEUTRAL", "confidence": 0.05, "impact_type": "supplier_chain", "reasoning": "via x"},
            ],
        }

    def test_alert_structure_and_ranking(self):
        a = alert.build_alert(self._enriched())
        assert a["event_type"] == "disaster"
        assert a["exposed_count"] == 2
        # ranked by move_likelihood descending
        scores = [n["move_likelihood"] for n in a["names"]]
        assert scores == sorted(scores, reverse=True)
        # direction is present but labelled informational (not a prediction field)
        assert "direction_informational" in a["names"][0]
        assert all("tier" in n for n in a["names"])

    def test_empty_exposed(self):
        a = alert.build_alert({"event_type": "macro", "severity": 0.3, "exposed": []})
        assert a["names"] == [] and a["top_tier"] == "LOW"

    def test_render_is_text_and_marks_direction_informational(self):
        text = alert.render_alert(alert.build_alert(self._enriched()))
        assert "NOT a prediction" in text
        assert "KQ" in text
