"""Tests for the propagation calibration harness."""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import graph as G
import calibrate as C


def test_fit_scale_recovers_exact_line():
    # realised = 3.0 * shock, no noise -> fitted scale == 3.0
    pairs = [(s, 3.0 * s) for s in (-1.0, -0.5, 0.2, 0.7, 1.0)]
    assert abs(C.fit_scale(pairs) - 3.0) < 1e-9


def test_fit_scale_none_when_degenerate():
    assert C.fit_scale([(0.0, 1.0), (0.0, -2.0)]) is None
    assert C.fit_scale([]) is None


def test_direction_accuracy_perfect_when_aligned():
    pairs = [(1.0, 5.0), (-1.0, -5.0), (0.05, 0.1)]
    # scale 5 -> predicted 5, -5, 0.25 ; threshold 0.5 -> UP, DOWN, NEUTRAL
    assert C.direction_accuracy(pairs, scale=5.0, threshold=0.5) == 1.0


def test_extract_pairs_recovers_raw_shock():
    # predicted_magnitude was written with the current scale; recovery divides
    # it back out to a raw shock.
    scale = G.MAGNITUDE_SCALE
    rows = [{"predicted_magnitude": 0.5 * scale, "pct_change": 2.0}]
    pairs = C.extract_pairs(rows, scale)
    assert len(pairs) == 1
    assert abs(pairs[0][0] - 0.5) < 1e-9
    assert pairs[0][1] == 2.0


def test_extract_pairs_skips_missing_fields():
    rows = [
        {"predicted_magnitude": 1.0},               # no realised
        {"pct_change": 1.0},                         # no prediction
        {"predicted_magnitude": None, "pct_change": 1.0},
    ]
    assert C.extract_pairs(rows, G.MAGNITUDE_SCALE) == []


def test_demo_fitter_recovers_known_scale():
    rows = C._demo_rows(n=800, true_scale=4.2, noise=0.5, seed=1)
    pairs = C.extract_pairs(rows, G.MAGNITUDE_SCALE)
    report = C.build_report(pairs)
    assert "error" not in report
    # fitted scale should land near the true 4.2 despite noise
    assert abs(report["fitted"]["MAGNITUDE_SCALE"] - 4.2) < 0.4
    # fitting should not make magnitude error worse than the current guess
    assert report["fitted"]["mae"] <= report["current"]["mae"] + 1e-6


def test_write_and_load_calibration_roundtrip(tmp_path):
    saved = {k: getattr(G, k) for k in G._CALIBRATABLE}
    cal_path = tmp_path / "calibration.json"
    try:
        C.write_calibration({"MAGNITUDE_SCALE": 9.99, "DIRECTION_THRESHOLD": 0.75}, path=cal_path)
        data = json.loads(cal_path.read_text())
        assert data["MAGNITUDE_SCALE"] == 9.99
        applied = G.load_calibration(cal_path)
        assert applied["MAGNITUDE_SCALE"] == 9.99
        assert G.MAGNITUDE_SCALE == 9.99
        assert G.DIRECTION_THRESHOLD == 0.75
    finally:
        # restore module globals so other tests are unaffected
        for k, v in saved.items():
            setattr(G, k, v)


def test_load_calibration_absent_file_is_noop(tmp_path):
    missing = tmp_path / "nope.json"
    assert G.load_calibration(missing) == {}


# ---------------------------------------------------------------------------
# Structural calibration (HOP_DECAY + channel gains)
# ---------------------------------------------------------------------------

FULL_GAINS = {"sector": 1.0, "competitor": 1.0, "product": 1.0, "supplier": 1.0}


def test_propagate_event_respects_family_gain():
    g = G.build_default_graph()
    ev = {"event_type": "earnings", "sources": {"KCB": 1.0}}
    off = C.propagate_event(g, ev, hop_decay=0.75,
                            gains={"sector": 0, "competitor": 0, "product": 0, "supplier": 0})
    on = C.propagate_event(g, ev, hop_decay=0.75, gains=FULL_GAINS)
    assert "EQTY" not in off          # all channels dead -> nothing reached
    assert "EQTY" in on               # competitor channel reaches a rival bank
    strong = C.propagate_event(g, ev, hop_decay=0.75,
                               gains={"sector": 1, "competitor": 2, "product": 1, "supplier": 1})
    assert abs(strong["EQTY"]) > abs(on["EQTY"])  # bigger gain -> bigger shock


def test_propagate_event_uses_unit_scale():
    """propagate_event returns raw shock (magnitude_scale=1), not a scaled %."""
    g = G.build_default_graph()
    ev = {"event_type": "regulation", "sources": {"KCB": 1.0}}
    shocks = C.propagate_event(g, ev, hop_decay=0.75, gains=FULL_GAINS)
    assert shocks and all(abs(s) <= 1.0 for s in shocks.values())


def test_structural_pairs_penalises_unreached_tickers():
    g = G.build_default_graph()
    # KQ (Transport) is not reachable from a KCB earnings event
    ev = {"event_type": "earnings", "sources": {"KCB": 1.0}, "realised": {"KQ": 3.0}}
    pairs = C.structural_pairs(g, [ev], 0.75, FULL_GAINS)
    assert any(shock == 0.0 for shock, _ in pairs)


def test_demo_structural_reduces_error_and_recovers_signal():
    events, graph, true_gains = C._demo_events(n=240, seed=5)
    report = C.fit_structural(graph, events, rounds=4)
    # fitting must not worsen either metric, and should strictly improve MAE
    # because the generating params differ from the engine defaults
    assert report["fitted"]["mae"] < report["current"]["mae"]
    assert report["fitted"]["direction_accuracy"] >= report["current"]["direction_accuracy"] - 1e-9
    # competitor channel was generated 4x stronger than sector -> fitted gains
    # should preserve that ordering
    fg = report["fitted"]["CHANNEL_GAINS"]
    assert fg["competitor"] >= fg["sector"]


def test_fit_structural_write_load_roundtrip(tmp_path):
    saved = {k: getattr(G, k) for k in G._CALIBRATABLE}
    cal_path = tmp_path / "calibration.json"
    try:
        C.write_calibration(
            {"HOP_DECAY": 0.55, "CHANNEL_GAINS": {"product": 2.5}}, path=cal_path
        )
        applied = G.load_calibration(cal_path)
        assert G.HOP_DECAY == 0.55
        # partial gains dict merges: product overridden, others keep default 1.0
        assert G.CHANNEL_GAINS["product"] == 2.5
        assert G.CHANNEL_GAINS["competitor"] == 1.0
    finally:
        for k, v in saved.items():
            setattr(G, k, v)
