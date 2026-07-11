"""Tests for the honest event-study backtest harness."""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import harness as H


# ---------------------------------------------------------------------------
# Synthetic panel builders
# ---------------------------------------------------------------------------

def _dates(n, start="2024-01-01"):
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _market_levels(dates):
    """Deterministic market index level path."""
    lvl = 1.0
    out = {}
    for i, d in enumerate(dates):
        if i:
            mr = 0.001 * ((i % 5) - 2)      # small varying return
            lvl *= (1 + mr)
        out[d] = lvl
    return out


def _panel(series):
    cal = sorted({d for s in series.values() for d in s})
    return H.Panel(series=series, calendar=cal).build()


def _index_tickers(dates, mlevels, k=10):
    """k tickers that track the market exactly (define the index)."""
    return {f"IDX{j}": {d: 100 * mlevels[d] for d in dates} for j in range(k)}


# ---------------------------------------------------------------------------
# Abnormal return removes the market component
# ---------------------------------------------------------------------------

def test_abnormal_return_zero_for_pure_market_mover():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    # PURE market mover: tracks the index exactly -> abnormal ~ 0 even though the
    # raw return over the window is clearly positive.
    series["PMM"] = {d: 50 * ml[d] for d in dates}
    panel = _panel(series)
    sc = H.score_event(panel, "PMM", dates[40], horizon=5, threshold=1.5)
    assert sc is not None
    assert abs(sc.abnormal_return) < 1.5          # NEUTRAL band
    assert sc.label == "NEUTRAL"
    # sanity: the raw return really was non-trivial and beta ~ 1
    assert abs(sc.beta - 1.0) < 0.05


def test_idiosyncratic_move_shows_up_as_abnormal():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    # tracks market, then jumps +6% idiosyncratically at the exit day
    px = {d: 50 * ml[d] for d in dates}
    entry_i = 40
    exit_i = 43
    for j in range(exit_i, len(dates)):
        px[dates[j]] *= 1.06
    series["JMP"] = px
    panel = _panel(series)
    sc = H.score_event(panel, "JMP", dates[entry_i], horizon=3, threshold=1.5)
    assert sc.label == "UP"
    assert sc.abnormal_return > 4.0


# ---------------------------------------------------------------------------
# Point-in-time: no lookahead leakage
# ---------------------------------------------------------------------------

def test_future_prices_do_not_change_past_event_score():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    series["ABC"] = {d: 30 * ml[d] * (1.02 if i >= 43 else 1.0)
                     for i, d in enumerate(dates)}
    panel = _panel(series)
    before = H.score_event(panel, "ABC", dates[40], horizon=3)

    # Mutate prices AFTER the event window (indices 60+) for the stock and the
    # index. A point-in-time score must be identical.
    series2 = {t: dict(s) for t, s in series.items()}
    for i in range(60, len(dates)):
        series2["ABC"][dates[i]] *= 5.0
        for j in range(10):
            series2[f"IDX{j}"][dates[i]] *= 0.3
    panel2 = _panel(series2)
    after = H.score_event(panel2, "ABC", dates[40], horizon=3)

    assert abs(before.abnormal_return - after.abnormal_return) < 1e-9
    assert abs(before.beta - after.beta) < 1e-9
    assert before.label == after.label


# ---------------------------------------------------------------------------
# Liquidity filter
# ---------------------------------------------------------------------------

def test_liquidity_filter_rejects_stale_and_absent():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    # STALE name: same price every day (no trading)
    series["STALE"] = {d: 12.0 for d in dates}
    # THIN name: present only every 10th day
    series["THIN"] = {d: 20 * ml[d] for i, d in enumerate(dates) if i % 10 == 0}
    # GOOD name: trades and moves daily
    series["GOOD"] = {d: 40 * ml[d] * (1 + 0.001 * (i % 3)) for i, d in enumerate(dates)}
    panel = _panel(series)
    ei = H._find_event_index(panel, "GOOD", dates[40])
    assert H.liquidity_ok(panel, "GOOD", ei)
    assert not H.liquidity_ok(panel, "STALE", ei)
    ei_thin = H._find_event_index(panel, "THIN", dates[40])
    assert not H.liquidity_ok(panel, "THIN", ei_thin)


# ---------------------------------------------------------------------------
# Corporate-action guard
# ---------------------------------------------------------------------------

def test_corporate_action_jump_is_flagged():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    px = {d: 25 * ml[d] for d in dates}
    px[dates[42]] *= 0.7      # -30% single-day drop (e.g. ex-dividend / split)
    for j in range(43, len(dates)):
        px[dates[j]] *= 0.7
    series["DIV"] = px
    panel = _panel(series)
    sc = H.score_event(panel, "DIV", dates[40], horizon=5, jump_thresh=0.20)
    assert sc.corp_action is True


# ---------------------------------------------------------------------------
# Model vs baselines
# ---------------------------------------------------------------------------

def test_model_aligned_with_abnormal_beats_baselines():
    import random as _r
    rng = _r.Random(7)
    dates = _dates(90)
    # A real market factor with genuine variance, built by MANY index names so
    # the test ticker doesn't dominate the index (else its idio becomes "market"
    # and beta absorbs it — as the harness correctly does).
    mf = [0.0] + [rng.gauss(0, 0.015) for _ in dates[1:]]
    mlvl, lvl = {}, 1.0
    for i, d in enumerate(dates):
        lvl *= (1 + mf[i])
        mlvl[d] = lvl
    series = {f"IDX{j}": {d: (80 + j) * mlvl[d] for d in dates} for j in range(30)}

    # SIG = market + INDEPENDENT idiosyncratic bump (fixed ±3% > threshold)
    idio = {i: (0.03 if rng.random() < 0.5 else -0.03) for i in range(len(dates))}
    px = {dates[0]: 60.0}
    for i in range(1, len(dates)):
        px[dates[i]] = px[dates[i - 1]] * (1 + mf[i] + idio[i])
    series["SIG"] = px
    panel = _panel(series)

    # model knows the sign of the exit-day idiosyncratic move (h=1 -> exit=entry+1)
    events = [{"ticker": "SIG",
               "direction": "UP" if idio[entry_i + 1] > 0 else "DOWN",
               "confidence": 0.8, "article_date": dates[entry_i]}
              for entry_i in range(40, 85)]

    metrics = H.evaluate(events, prices_csv="", horizons=(1,), threshold=1.5, panel=panel)
    r = metrics["horizons"]["1"]
    assert r["n"] >= 20
    assert r["model_accuracy"] > 0.8
    assert r["edge_over_best_baseline"] > 0
    assert r["baselines"]["always_neutral"] < 0.2   # labels are all UP/DOWN


def test_predictions_to_events_flattens_direct_and_indirect():
    preds = [{
        "article_date": "2024-03-01",
        "directly_affected": [{"ticker": "KCB", "direction": "UP", "confidence": 0.9}],
        "indirectly_affected": [{"ticker": "EQTY", "direction": "DOWN",
                                 "confidence": 0.4, "magnitude_pct": -1.2}],
    }]
    evs = H.predictions_to_events(preds)
    assert {e["ticker"] for e in evs} == {"KCB", "EQTY"}
    assert all(e["article_date"] == "2024-03-01" for e in evs)


def test_evaluate_counts_exclusions():
    dates = _dates(80)
    ml = _market_levels(dates)
    series = _index_tickers(dates, ml)
    series["STALE"] = {d: 5.0 for d in dates}
    panel = _panel(series)
    events = [
        {"ticker": "STALE", "direction": "UP", "article_date": dates[40]},
        {"ticker": "NOPE", "direction": "UP", "article_date": dates[40]},   # no price
        {"ticker": "IDX0", "direction": "UP", "article_date": ""},          # no date
    ]
    metrics = H.evaluate(events, prices_csv="", horizons=(3,), panel=panel)
    excl = metrics["excluded"]
    assert excl.get("illiquid", 0) >= 1
    assert excl.get("no_price", 0) >= 1 or excl.get("no_date", 0) >= 1
