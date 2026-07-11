"""
calibrate.py — Fit the propagation engine's magnitude coefficients to realised
price moves.

The graph engine emits a *signed shock* per affected ticker; ``MAGNITUDE_SCALE``
maps that unit shock to a percentage price move, and ``DIRECTION_THRESHOLD``
decides the UP/DOWN/NEUTRAL cutoff. Both are just starting guesses until fitted
against data.

This harness reads labeled rows produced by ``aligner.align()`` — each row
carries ``predicted_magnitude`` (the engine's estimate) and ``pct_change`` (the
realised move) — recovers the underlying raw shock, and fits:

  * ``MAGNITUDE_SCALE``   — least-squares-through-origin of realised ≈ scale·shock
  * ``DIRECTION_THRESHOLD`` — grid-searched to maximise direction accuracy

Results are written to ``calibration.json`` next to ``graph.py``, which loads
them automatically on import (no source edits needed).

Usage
-----
    # Fit from all labeled data and write calibration.json
    python calibrate.py --write

    # Fit from one file, print the report only
    python calibrate.py --labeled nse_dataset/labeled/labeled_XXXX.jsonl

    # Self-test: generate data from a known scale and confirm recovery
    python calibrate.py --demo
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

try:
    import graph as G
except ImportError:  # pragma: no cover
    from . import graph as G

logger = logging.getLogger(__name__)

DIRECTIONS = ("UP", "DOWN", "NEUTRAL")


# ---------------------------------------------------------------------------
# Loading (predicted raw shock, realised %) pairs
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                logger.warning("Skipping malformed line in %s", path.name)


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if not p.exists():
            logger.error("Not found: %s", p)
            continue
        rows.extend(list(_iter_jsonl(p)))
    return rows


def extract_pairs(rows: list[dict], current_scale: float) -> list[tuple[float, float]]:
    """
    Return (raw_shock, realised_pct) pairs. The raw shock is recovered by
    dividing the stored ``predicted_magnitude`` by the scale that produced it
    (the engine's current ``MAGNITUDE_SCALE``), so a new scale can be fitted
    independent of whatever scale was in force at label time.
    """
    pairs = []
    for r in rows:
        pm = r.get("predicted_magnitude")
        realised = r.get("pct_change")
        if not isinstance(pm, (int, float)) or not isinstance(realised, (int, float)):
            continue
        if current_scale == 0:
            continue
        shock = float(pm) / current_scale
        if shock == 0.0:
            continue
        pairs.append((shock, float(realised)))
    return pairs


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_scale(pairs: list[tuple[float, float]]) -> float | None:
    """Least-squares through the origin: scale = Σ(shock·realised) / Σ(shock²)."""
    num = sum(s * r for s, r in pairs)
    den = sum(s * s for s, r in pairs)
    if den == 0:
        return None
    return num / den


def mae(pairs: list[tuple[float, float]], scale: float) -> float:
    if not pairs:
        return 0.0
    return sum(abs(scale * s - r) for s, r in pairs) / len(pairs)


def direction_accuracy(pairs: list[tuple[float, float]], scale: float, threshold: float) -> float:
    """Fraction of pairs whose predicted direction (from scale·shock vs
    ±threshold) matches the realised direction (from realised vs ±threshold)."""
    if not pairs:
        return 0.0

    def label(x: float) -> str:
        if x >= threshold:
            return "UP"
        if x <= -threshold:
            return "DOWN"
        return "NEUTRAL"

    correct = sum(1 for s, r in pairs if label(scale * s) == label(r))
    return correct / len(pairs)


def fit_threshold(pairs: list[tuple[float, float]], scale: float,
                  grid: list[float] | None = None) -> tuple[float, float]:
    """Grid-search the direction threshold that maximises accuracy.
    Returns (best_threshold, best_accuracy)."""
    grid = grid or [round(0.25 * i, 2) for i in range(1, 13)]  # 0.25 .. 3.0
    best_t, best_a = grid[0], -1.0
    for t in grid:
        a = direction_accuracy(pairs, scale, t)
        if a > best_a:
            best_t, best_a = t, a
    return best_t, best_a


# ---------------------------------------------------------------------------
# Structural fitting: HOP_DECAY + per-family CHANNEL_GAINS
#
# Unlike the magnitude fit above (which needs only stored predicted/realised
# numbers), fitting decay and channel gains requires *re-propagating* each event
# under candidate coefficients. The input is therefore richer — an "event
# record" — rather than a flat labeled row:
#
#   {"event_type": "disaster",
#    "sources":  {"product:Boeing 737 MAX": -1.0, "Ethiopian Airlines": -1.0},
#    "realised": {"KQ": -3.0, "EQTY": 0.1}}     # ticker -> realised % move
#
# For each candidate (hop_decay, gains) we propagate every event to raw shocks,
# pair them with realised moves, re-fit MAGNITUDE_SCALE by least squares, and
# score magnitude MAE. Coordinate descent walks decay then each family gain.
# ---------------------------------------------------------------------------

FAMILIES = ("sector", "competitor", "product", "supplier")
DECAY_GRID = [round(0.35 + 0.05 * i, 2) for i in range(12)]      # 0.35 .. 0.90
GAIN_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def propagate_event(graph, event: dict, hop_decay: float,
                    gains: dict[str, float]) -> dict[str, float]:
    """Propagate one event to {ticker: raw_shock} under given coefficients.
    magnitude_scale is held at 1.0 so we recover the raw shock and fit scale
    separately."""
    sources = {k: float(v) for k, v in (event.get("sources") or {}).items()}
    impacts = G.propagate(
        graph, sources, event.get("event_type", ""),
        hop_decay=hop_decay, gains=gains, magnitude_scale=1.0,
    )
    return {t: imp.shock for t, imp in impacts.items()}


def structural_pairs(graph, events: list[dict], hop_decay: float,
                     gains: dict[str, float]) -> list[tuple[float, float]]:
    """(raw_shock, realised_pct) over every (event, realised-ticker). Tickers
    that the propagation fails to reach contribute a 0-shock pair, so under-reach
    is penalised."""
    pairs = []
    for ev in events:
        pred = propagate_event(graph, ev, hop_decay, gains)
        for ticker, realised in (ev.get("realised") or {}).items():
            if not isinstance(realised, (int, float)):
                continue
            pairs.append((pred.get(ticker, 0.0), float(realised)))
    return pairs


def score_structural(graph, events: list[dict], hop_decay: float,
                     gains: dict[str, float]) -> dict:
    pairs = structural_pairs(graph, events, hop_decay, gains)
    scale = fit_scale(pairs)
    if scale is None:
        return {"mae": float("inf"), "scale": 0.0, "direction_accuracy": 0.0,
                "threshold": G.DIRECTION_THRESHOLD, "n": len(pairs)}
    thr, acc = fit_threshold(pairs, scale)
    return {
        "mae": mae(pairs, scale),
        "scale": scale,
        "direction_accuracy": acc,
        "threshold": thr,
        "n": len(pairs),
    }


def fit_structural(graph, events: list[dict], rounds: int = 4) -> dict:
    """Coordinate-descent fit of HOP_DECAY + per-family CHANNEL_GAINS.

    Returns the fitted coefficients plus baseline (current) vs fitted metrics.
    """
    decay = float(G.HOP_DECAY)
    gains = {f: float(G.CHANNEL_GAINS.get(f, 1.0)) for f in FAMILIES}

    baseline = score_structural(graph, events, decay, gains)
    best = dict(baseline)

    for _ in range(rounds):
        improved = False

        # --- hop_decay ---
        for cand in DECAY_GRID:
            s = score_structural(graph, events, cand, gains)
            if s["mae"] < best["mae"] - 1e-9:
                best, decay, improved = s, cand, True

        # --- each family gain ---
        for fam in FAMILIES:
            base_gain = gains[fam]
            for cand in GAIN_GRID:
                trial = dict(gains)
                trial[fam] = cand
                s = score_structural(graph, events, decay, trial)
                if s["mae"] < best["mae"] - 1e-9:
                    best, base_gain, improved = s, cand, True
            gains[fam] = base_gain

        if not improved:
            break

    return {
        "n_pairs": best["n"],
        "current": {
            "HOP_DECAY": round(float(G.HOP_DECAY), 4),
            "CHANNEL_GAINS": {f: round(float(G.CHANNEL_GAINS.get(f, 1.0)), 4) for f in FAMILIES},
            "MAGNITUDE_SCALE": round(baseline["scale"], 4),
            "mae": round(baseline["mae"], 4),
            "direction_accuracy": round(baseline["direction_accuracy"], 4),
        },
        "fitted": {
            "HOP_DECAY": round(decay, 4),
            "CHANNEL_GAINS": {f: round(gains[f], 4) for f in FAMILIES},
            "MAGNITUDE_SCALE": round(best["scale"], 4),
            "DIRECTION_THRESHOLD": round(best["threshold"], 4),
            "mae": round(best["mae"], 4),
            "direction_accuracy": round(best["direction_accuracy"], 4),
        },
    }


def build_event_records(predictions: list[dict], prices_csv: str,
                        lookahead_days: int = 3, graph=None) -> list[dict]:
    """
    Turn real extractor predictions + a prices CSV into event records for
    structural calibration. Seeds are collected exactly as the pipeline does
    (``graph.collect_sources``); realised moves come from the aligner's price
    lookup for every ticker the propagation reaches.
    """
    from aligner import load_prices, get_price_change  # lazy: pulls in config

    graph = graph or G.build_default_graph()
    prices = load_prices(prices_csv)
    events = []
    for pred in predictions:
        sources, _conf = G.collect_sources(pred, graph)
        if not sources:
            continue
        article_date = (pred.get("article_date")
                        or (pred.get("data_quality") or {}).get("article_date") or "")
        impacts = G.propagate(graph, sources, pred.get("event_type", ""))
        realised = {}
        for ticker in impacts:
            _t0, _tn, pct = get_price_change(prices, ticker, article_date,
                                             lookahead_days=lookahead_days)
            if isinstance(pct, (int, float)):
                realised[ticker] = round(pct, 4)
        if realised:
            events.append({
                "event_type": pred.get("event_type", ""),
                "sources": sources,
                "realised": realised,
            })
    return events


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def write_calibration(updates: dict, path: Path = G.CALIBRATION_FILE) -> None:
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote calibration to %s: %s", path, updates)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(pairs: list[tuple[float, float]]) -> dict:
    current_scale = G.MAGNITUDE_SCALE
    current_thr = G.DIRECTION_THRESHOLD
    fitted_scale = fit_scale(pairs)
    if fitted_scale is None:
        return {"error": "no usable (predicted_magnitude, pct_change) pairs"}

    fitted_thr, fitted_acc = fit_threshold(pairs, fitted_scale)
    return {
        "n_pairs": len(pairs),
        "current": {
            "MAGNITUDE_SCALE": round(current_scale, 4),
            "DIRECTION_THRESHOLD": round(current_thr, 4),
            "mae": round(mae(pairs, current_scale), 4),
            "direction_accuracy": round(direction_accuracy(pairs, current_scale, current_thr), 4),
        },
        "fitted": {
            "MAGNITUDE_SCALE": round(fitted_scale, 4),
            "DIRECTION_THRESHOLD": round(fitted_thr, 4),
            "mae": round(mae(pairs, fitted_scale), 4),
            "direction_accuracy": round(fitted_acc, 4),
        },
    }


def print_report(report: dict) -> None:
    if "error" in report:
        print(f"[calibrate] {report['error']}")
        return
    c, f = report["current"], report["fitted"]
    print("\n" + "=" * 56)
    print("  PROPAGATION MAGNITUDE CALIBRATION")
    print("=" * 56)
    print(f"  pairs used            : {report['n_pairs']}")
    print(f"  {'':22}{'current':>12}{'fitted':>12}")
    print(f"  {'MAGNITUDE_SCALE':22}{c['MAGNITUDE_SCALE']:>12}{f['MAGNITUDE_SCALE']:>12}")
    print(f"  {'DIRECTION_THRESHOLD':22}{c['DIRECTION_THRESHOLD']:>12}{f['DIRECTION_THRESHOLD']:>12}")
    print(f"  {'magnitude MAE':22}{c['mae']:>12}{f['mae']:>12}")
    print(f"  {'direction accuracy':22}{c['direction_accuracy']:>12}{f['direction_accuracy']:>12}")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# Demo (self-test the fitter recovers a known scale)
# ---------------------------------------------------------------------------

def _demo_events(n: int = 240, true_decay: float = 0.6, true_scale: float = 4.0,
                 noise: float = 0.4, seed: int = 3):
    """Generate event records over the real graph from KNOWN structural params.

    realised ≈ true_scale · shock(true_decay, true_gains) + noise. The default
    coefficients differ from the truth, so a correct fitter must reduce MAE.
    Returns (events, graph, true_gains).
    """
    rng = random.Random(seed)
    graph = G.build_default_graph()
    true_gains = {"sector": 0.5, "competitor": 2.0, "product": 1.5, "supplier": 1.0}

    # (event_type, seed node) templates exercising different channel families
    templates = [
        ("disaster", "product:Boeing 737 MAX"),
        ("disaster", "product:Boeing 787 Dreamliner"),
        ("earnings", "KCB"),
        ("earnings", "EQTY"),
        ("regulation", "KCB"),
        ("regulation", "BAMB"),
    ]
    events = []
    for i in range(n):
        etype, seed_node = templates[i % len(templates)]
        sign = rng.choice((-1.0, 1.0))
        sources = {seed_node: sign}
        shocks = propagate_event(graph, {"event_type": etype, "sources": sources},
                                 true_decay, true_gains)
        realised = {t: round(true_scale * s + rng.gauss(0, noise), 4)
                    for t, s in shocks.items()}
        if realised:
            events.append({"event_type": etype, "sources": sources, "realised": realised})
    return events, graph, true_gains


def print_structural_report(report: dict) -> None:
    c, f = report["current"], report["fitted"]
    print("\n" + "=" * 60)
    print("  STRUCTURAL CALIBRATION (HOP_DECAY + CHANNEL_GAINS)")
    print("=" * 60)
    print(f"  pairs used            : {report['n_pairs']}")
    print(f"  {'':22}{'current':>12}{'fitted':>12}")
    print(f"  {'HOP_DECAY':22}{c['HOP_DECAY']:>12}{f['HOP_DECAY']:>12}")
    for fam in FAMILIES:
        print(f"  gain[{fam:<10}]     {c['CHANNEL_GAINS'][fam]:>12}{f['CHANNEL_GAINS'][fam]:>12}")
    print(f"  {'MAGNITUDE_SCALE':22}{c['MAGNITUDE_SCALE']:>12}{f['MAGNITUDE_SCALE']:>12}")
    print(f"  {'magnitude MAE':22}{c['mae']:>12}{f['mae']:>12}")
    print(f"  {'direction accuracy':22}{c['direction_accuracy']:>12}{f['direction_accuracy']:>12}")
    print("=" * 60 + "\n")


def _demo_rows(n: int = 400, true_scale: float = 4.2, noise: float = 0.6,
               seed: int = 7) -> list[dict]:
    """Rows where realised ≈ true_scale·shock + noise, with predicted_magnitude
    written using the engine's *current* scale (so recovery must divide it out)."""
    rng = random.Random(seed)
    cur = G.MAGNITUDE_SCALE
    rows = []
    for _ in range(n):
        shock = rng.uniform(-1.0, 1.0)
        realised = true_scale * shock + rng.gauss(0, noise)
        rows.append({
            "ticker": "DEMO",
            "predicted_magnitude": round(shock * cur, 4),  # engine output today
            "pct_change": round(realised, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Calibrate propagation magnitude coefficients")
    parser.add_argument("--labeled", help="Path to a specific labeled JSONL file")
    parser.add_argument("--output-dir", default="nse_dataset",
                        help="Root dir; scans labeled/ (default: nse_dataset)")
    parser.add_argument("--demo", action="store_true",
                        help="Self-test the magnitude fit on synthetic data")
    parser.add_argument("--events",
                        help="JSONL of event records for structural fitting "
                             "(HOP_DECAY + channel gains)")
    parser.add_argument("--fit-structural", action="store_true",
                        help="Fit HOP_DECAY + channel gains (needs --events)")
    parser.add_argument("--demo-structural", action="store_true",
                        help="Self-test the structural fit on synthetic events")
    parser.add_argument("--write", action="store_true",
                        help="Write fitted coefficients to calibration.json")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Structural branch: fit HOP_DECAY + per-family channel gains
    # ------------------------------------------------------------------
    if args.demo_structural or args.fit_structural or args.events:
        if args.demo_structural:
            events, graph, true_gains = _demo_events()
            print(f"[calibrate] demo-structural: true HOP_DECAY=0.6, "
                  f"true gains={true_gains} (defaults differ)")
        else:
            if not args.events:
                print("[calibrate] --fit-structural requires --events FILE")
                sys.exit(2)
            events = load_rows([Path(args.events)])
            graph = G.build_default_graph()

        if not events:
            print("[calibrate] no usable event records")
            sys.exit(1)

        report = fit_structural(graph, events)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_structural_report(report)

        if args.write:
            write_calibration({
                "HOP_DECAY": report["fitted"]["HOP_DECAY"],
                "CHANNEL_GAINS": report["fitted"]["CHANNEL_GAINS"],
                "MAGNITUDE_SCALE": report["fitted"]["MAGNITUDE_SCALE"],
                "DIRECTION_THRESHOLD": report["fitted"]["DIRECTION_THRESHOLD"],
            })
            print("[calibrate] calibration.json updated.")
        return

    # ------------------------------------------------------------------
    # Magnitude branch: fit MAGNITUDE_SCALE + DIRECTION_THRESHOLD
    # ------------------------------------------------------------------
    if args.demo:
        rows = _demo_rows()
        print(f"[calibrate] demo: true_scale=4.2, current MAGNITUDE_SCALE={G.MAGNITUDE_SCALE}")
    elif args.labeled:
        rows = load_rows([Path(args.labeled)])
    else:
        labeled_dir = Path(args.output_dir) / "labeled"
        files = sorted(labeled_dir.glob("labeled_*.jsonl")) if labeled_dir.exists() else []
        rows = load_rows(files)

    pairs = extract_pairs(rows, G.MAGNITUDE_SCALE)
    report = build_report(pairs)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if "error" in report:
        sys.exit(1)

    if args.write:
        write_calibration({
            "MAGNITUDE_SCALE": report["fitted"]["MAGNITUDE_SCALE"],
            "DIRECTION_THRESHOLD": report["fitted"]["DIRECTION_THRESHOLD"],
        })
        print("[calibrate] calibration.json updated — graph.py will use it on next import.")


if __name__ == "__main__":
    main()
