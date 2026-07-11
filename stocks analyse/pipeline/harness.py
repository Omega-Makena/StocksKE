"""
harness.py — an honest event-study backtest for news-driven predictions.

The pipeline's own accuracy print is naive: it labels on *raw* price change over a
single horizon. That conflates the signal with market-wide moves, corporate
actions, and thin-trade noise, and it has no baseline — so a number like "55%"
means nothing. This harness fixes that:

  * point-in-time      — every estimate for an event at date D uses ONLY data
                         strictly before D (betas, liquidity, momentum). No
                         lookahead. Mutating future prices cannot change a past
                         event's score (see tests).
  * abnormal returns   — a market-model event study: label on
                         AR = R_stock - beta * R_market, with beta from a trailing
                         estimation window, so index-wide moves are removed.
  * liquidity filter   — drop events on names that barely trade (stale/absent
                         prices in the trailing window) where "moves" are noise.
  * corporate-action   — flag/exclude event windows containing implausible single
       guard             day jumps (dividends/splits/rights) we can't adjust for.
  * multi-horizon      — score at several trading-day horizons at once.
  * baselines          — compare the model against always-NEUTRAL, majority-class,
                         random, and momentum. If it can't beat these, it's noise.

Pure standard library. Runs on the importer's prices CSV
(columns: "Stock Code", "Date", "Day's Final Price") plus a list of predictions
(the pipeline's enriched output) or flat event rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DIRECTIONS = ("UP", "DOWN", "NEUTRAL")


# ---------------------------------------------------------------------------
# Price panel
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    series: dict[str, dict[str, float]]          # ticker -> {date: close}
    calendar: list[str]                          # sorted unique trading dates
    cal_index: dict[str, int] = field(default_factory=dict)
    market_ret: dict[str, float] = field(default_factory=dict)   # date -> eq-wt daily return
    market_level: dict[str, float] = field(default_factory=dict) # date -> index level

    def build(self) -> "Panel":
        self.cal_index = {d: i for i, d in enumerate(self.calendar)}
        # equal-weighted daily market return across names present on both days
        for i in range(1, len(self.calendar)):
            d, prev = self.calendar[i], self.calendar[i - 1]
            rs = []
            for t, s in self.series.items():
                p1, p0 = s.get(d), s.get(prev)
                if p0 and p1 and p0 > 0:
                    rs.append(p1 / p0 - 1.0)
            self.market_ret[d] = sum(rs) / len(rs) if rs else 0.0
        # cumulative index level for O(1) horizon market returns
        lvl = 1.0
        for i, d in enumerate(self.calendar):
            if i > 0:
                lvl *= (1.0 + self.market_ret.get(d, 0.0))
            self.market_level[d] = lvl
        return self

    def price(self, ticker: str, date: str) -> float | None:
        return self.series.get(ticker, {}).get(date)

    def daily_return(self, ticker: str, date_idx: int) -> float | None:
        """Return over the *calendar* step ending at calendar[date_idx]."""
        if date_idx <= 0 or date_idx >= len(self.calendar):
            return None
        d, prev = self.calendar[date_idx], self.calendar[date_idx - 1]
        p1, p0 = self.price(ticker, d), self.price(ticker, prev)
        if p0 and p1 and p0 > 0:
            return p1 / p0 - 1.0
        return None


def load_panel(prices_csv: str) -> Panel:
    series: dict[str, dict[str, float]] = defaultdict(dict)
    dates: set[str] = set()
    p = Path(prices_csv)
    if p.exists():
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = (row.get("Stock Code") or "").strip()
                d = (row.get("Date") or "").strip()
                v = (row.get("Day's Final Price") or "").strip().replace(",", "")
                if not (t and d and v):
                    continue
                try:
                    series[t][d] = float(v)
                except ValueError:
                    continue
                dates.add(d)
    else:
        logger.error("prices CSV not found: %s", prices_csv)
    return Panel(series=dict(series), calendar=sorted(dates)).build()


# ---------------------------------------------------------------------------
# Point-in-time estimators (all use data strictly before the event)
# ---------------------------------------------------------------------------

def estimate_beta(panel: Panel, ticker: str, event_idx: int,
                  est_window: int = 60, min_obs: int = 20) -> float:
    """OLS slope of stock daily returns on market daily returns over the trailing
    window ending the day BEFORE the event. Falls back to beta = 1.0."""
    lo = max(1, event_idx - est_window)
    xs, ys = [], []
    for i in range(lo, event_idx):          # strictly < event_idx  => point-in-time
        rm = panel.market_ret.get(panel.calendar[i])
        rs = panel.daily_return(ticker, i)
        if rm is not None and rs is not None:
            xs.append(rm)
            ys.append(rs)
    n = len(xs)
    if n < min_obs:
        return 1.0
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    if vx <= 0:
        return 1.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / vx


def liquidity_ok(panel: Panel, ticker: str, event_idx: int,
                 window: int = 30, min_present: int = 15,
                 max_stale_frac: float = 0.5) -> bool:
    """True if the name traded enough in the trailing window: present on enough
    days and not mostly stale (unchanged) prices."""
    lo = max(1, event_idx - window)
    present = 0
    stale = 0
    for i in range(lo, event_idx):
        d, prev = panel.calendar[i], panel.calendar[i - 1]
        p1 = panel.price(ticker, d)
        if p1 is None:
            continue
        present += 1
        p0 = panel.price(ticker, prev)
        if p0 is not None and p0 == p1:
            stale += 1
    if present < min_present:
        return False
    return (stale / present) <= max_stale_frac if present else False


def _find_event_index(panel: Panel, ticker: str, article_date: str,
                      fwd_tol: int = 4) -> int | None:
    """First calendar index at/after the article date where the ticker has a
    price (the entry point of the event window)."""
    # locate first calendar date >= article_date
    start = None
    for i, d in enumerate(panel.calendar):
        if d >= article_date:
            start = i
            break
    if start is None:
        return None
    for i in range(start, min(start + fwd_tol + 1, len(panel.calendar))):
        if panel.price(ticker, panel.calendar[i]) is not None:
            return i
    return None


@dataclass
class EventScore:
    ticker: str
    horizon: int
    abnormal_return: float     # AR in %  (already *100)
    raw_return: float          # %
    market_return: float       # %
    beta: float
    label: str                 # UP/DOWN/NEUTRAL from abnormal return
    corp_action: bool          # implausible jump in window


def score_event(panel: Panel, ticker: str, article_date: str, horizon: int,
                threshold: float = 1.5, est_window: int = 60, min_obs: int = 20,
                jump_thresh: float = 0.20, fwd_tol: int = 4) -> EventScore | None:
    """Abnormal return of `ticker` over `horizon` trading days from the first
    tradeable day at/after `article_date`. Returns None if it can't be scored
    (no entry/exit price)."""
    ei = _find_event_index(panel, ticker, article_date, fwd_tol)
    if ei is None:
        return None
    exit_idx = ei + horizon
    if exit_idx >= len(panel.calendar):
        return None
    d0, d1 = panel.calendar[ei], panel.calendar[exit_idx]
    p0, p1 = panel.price(ticker, d0), panel.price(ticker, d1)
    if not (p0 and p1 and p0 > 0):
        return None

    raw = p1 / p0 - 1.0
    mkt = panel.market_level[d1] / panel.market_level[d0] - 1.0
    beta = estimate_beta(panel, ticker, ei, est_window, min_obs)
    ar = raw - beta * mkt

    # corporate-action guard: any implausible single-day jump in the window
    corp = False
    for i in range(ei + 1, exit_idx + 1):
        r = panel.daily_return(ticker, i)
        if r is not None and abs(r) > jump_thresh:
            corp = True
            break

    ar_pct = ar * 100.0
    if ar_pct >= threshold:
        lab = "UP"
    elif ar_pct <= -threshold:
        lab = "DOWN"
    else:
        lab = "NEUTRAL"

    return EventScore(ticker=ticker, horizon=horizon, abnormal_return=ar_pct,
                      raw_return=raw * 100.0, market_return=mkt * 100.0,
                      beta=beta, label=lab, corp_action=corp)


def momentum_direction(panel: Panel, ticker: str, article_date: str,
                       horizon: int, threshold: float = 1.5, **kw) -> str:
    """Baseline: sign of the ticker's abnormal return over the `horizon` days
    BEFORE the event (point-in-time momentum)."""
    ei = _find_event_index(panel, ticker, article_date, kw.get("fwd_tol", 4))
    if ei is None or ei - horizon < 1:
        return "NEUTRAL"
    d0, d1 = panel.calendar[ei - horizon], panel.calendar[ei]
    p0, p1 = panel.price(ticker, d0), panel.price(ticker, d1)
    if not (p0 and p1 and p0 > 0):
        return "NEUTRAL"
    raw = p1 / p0 - 1.0
    mkt = panel.market_level[d1] / panel.market_level[d0] - 1.0
    beta = estimate_beta(panel, ticker, ei - horizon, kw.get("est_window", 60), kw.get("min_obs", 20))
    ar = (raw - beta * mkt) * 100.0
    return "UP" if ar >= threshold else "DOWN" if ar <= -threshold else "NEUTRAL"


# ---------------------------------------------------------------------------
# Events adapter
# ---------------------------------------------------------------------------

def predictions_to_events(preds: list[dict]) -> list[dict]:
    """Flatten enriched predictions into scoreable (ticker, direction, …) rows."""
    events = []
    for p in preds:
        d = p.get("article_date") or (p.get("data_quality") or {}).get("article_date")
        for grp in ("directly_affected", "indirectly_affected"):
            for e in p.get(grp) or []:
                if e.get("ticker") and e.get("direction"):
                    events.append({
                        "ticker": e["ticker"],
                        "direction": e["direction"],
                        "confidence": e.get("confidence"),
                        "predicted_magnitude": e.get("magnitude_pct"),
                        "impact_type": e.get("impact_type"),
                        "article_date": d,
                    })
    return events


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _accuracy(pairs: list[tuple[str, str]]) -> float:
    return sum(1 for pr, ac in pairs if pr == ac) / len(pairs) if pairs else 0.0


def _majority_label(labels: list[str]) -> str:
    c = defaultdict(int)
    for l in labels:
        c[l] += 1
    return max(DIRECTIONS, key=lambda k: c[k])


def evaluate(events: list[dict], prices_csv: str, horizons=(1, 3, 5),
             threshold: float = 1.5, liquidity: bool = True,
             exclude_corp_actions: bool = True, seed: int = 42,
             panel: Panel | None = None) -> dict:
    """Score predictions against abnormal returns at each horizon, with baselines
    and exclusion accounting."""
    panel = panel or load_panel(prices_csv)
    rng = random.Random(seed)
    out: dict = {"n_events": len(events), "horizons": {}}
    excl = defaultdict(int)

    for h in horizons:
        model_pairs: list[tuple[str, str]] = []
        labels: list[str] = []
        confidences: list[tuple[float, bool]] = []
        mag_errors: list[float] = []
        mom_pairs: list[tuple[str, str]] = []

        for ev in events:
            t, ad = ev.get("ticker"), ev.get("article_date")
            if not t or not ad:
                excl["no_date"] += 1
                continue
            ei = _find_event_index(panel, t, ad)
            if ei is None:
                excl["no_price"] += 1
                continue
            if liquidity and not liquidity_ok(panel, t, ei):
                excl["illiquid"] += 1
                continue
            sc = score_event(panel, t, ad, h, threshold=threshold)
            if sc is None:
                excl["unscoreable"] += 1
                continue
            if exclude_corp_actions and sc.corp_action:
                excl["corp_action"] += 1
                continue

            pred = ev["direction"]
            model_pairs.append((pred, sc.label))
            labels.append(sc.label)
            mom_pairs.append((momentum_direction(panel, t, ad, h, threshold), sc.label))
            if isinstance(ev.get("confidence"), (int, float)):
                confidences.append((float(ev["confidence"]), pred == sc.label))
            if isinstance(ev.get("predicted_magnitude"), (int, float)):
                mag_errors.append(abs(float(ev["predicted_magnitude"]) - sc.abnormal_return))

        n = len(model_pairs)
        if n == 0:
            out["horizons"][str(h)] = {"n": 0}
            continue

        maj = _majority_label(labels)
        baselines = {
            "always_neutral": _accuracy([("NEUTRAL", ac) for _, ac in model_pairs]),
            "majority": _accuracy([(maj, ac) for _, ac in model_pairs]),
            "random": _accuracy([(rng.choice(DIRECTIONS), ac) for _, ac in model_pairs]),
            "momentum": _accuracy(mom_pairs),
        }
        model_acc = _accuracy(model_pairs)
        best_base = max(baselines.values())

        # per-class precision/recall
        per_class = {}
        for cls in DIRECTIONS:
            tp = sum(1 for pr, ac in model_pairs if pr == cls and ac == cls)
            fp = sum(1 for pr, ac in model_pairs if pr == cls and ac != cls)
            fn = sum(1 for pr, ac in model_pairs if pr != cls and ac == cls)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                              "support": tp + fn}

        res = {
            "n": n,
            "model_accuracy": round(model_acc, 4),
            "baselines": {k: round(v, 4) for k, v in baselines.items()},
            "edge_over_best_baseline": round(model_acc - best_base, 4),
            "label_distribution": {c: labels.count(c) for c in DIRECTIONS},
            "per_class": per_class,
        }
        if mag_errors:
            res["magnitude_mae"] = round(sum(mag_errors) / len(mag_errors), 4)
        if confidences:
            res["confidence_calibration"] = _calibration_bins(confidences)
        out["horizons"][str(h)] = res

    out["excluded"] = dict(excl)
    return out


def _calibration_bins(confidences: list[tuple[float, bool]], nbins: int = 5) -> list[dict]:
    """Reliability bins: is a 0.7-confidence call right ~70% of the time?"""
    bins = []
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        sel = [correct for c, correct in confidences if (lo <= c < hi or (b == nbins - 1 and c == hi))]
        if sel:
            bins.append({"range": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                         "empirical_accuracy": round(sum(sel) / len(sel), 4)})
    return bins


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 64)
    print("  EVENT-STUDY BACKTEST (abnormal returns, point-in-time)")
    print("=" * 64)
    print(f"  events in       : {metrics['n_events']}")
    print(f"  excluded        : {metrics.get('excluded', {})}")
    for h, r in metrics["horizons"].items():
        print("-" * 64)
        if not r.get("n"):
            print(f"  horizon {h}d: no scoreable events")
            continue
        print(f"  horizon {h}d  (n={r['n']}, labels={r['label_distribution']})")
        print(f"    model accuracy        : {r['model_accuracy']:.1%}")
        for k, v in r["baselines"].items():
            print(f"    baseline {k:<14}: {v:.1%}")
        edge = r["edge_over_best_baseline"]
        verdict = "BEATS baselines" if edge > 0 else "NO EDGE over baselines"
        print(f"    edge over best base   : {edge:+.1%}   -> {verdict}")
        if "magnitude_mae" in r:
            print(f"    magnitude MAE (AR)    : {r['magnitude_mae']:.2f} pp")
        if "confidence_calibration" in r:
            print(f"    confidence calibration: {r['confidence_calibration']}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_predictions(path: Path) -> list[dict]:
    preds = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds.append(item.get("prediction") if isinstance(item, dict) and "prediction" in item else item)
    return preds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Honest event-study backtest")
    ap.add_argument("--predictions", help="JSONL of enriched predictions")
    ap.add_argument("--events", help="JSONL of flat event rows {ticker,direction,article_date}")
    ap.add_argument("--prices", required=True, help="prices CSV from importer")
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--no-liquidity-filter", action="store_true")
    ap.add_argument("--keep-corp-actions", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.predictions:
        events = predictions_to_events(_load_predictions(Path(args.predictions)))
    elif args.events:
        events = _load_predictions(Path(args.events))
    else:
        ap.error("provide --predictions or --events")

    horizons = tuple(int(x) for x in args.horizons.split(","))
    metrics = evaluate(events, args.prices, horizons=horizons, threshold=args.threshold,
                       liquidity=not args.no_liquidity_filter,
                       exclude_corp_actions=not args.keep_corp_actions)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics)


if __name__ == "__main__":
    main()
