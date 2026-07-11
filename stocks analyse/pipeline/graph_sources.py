"""
graph_sources.py — data-derived edge providers for the knowledge graph.

The curated structure (companies.py + graph_data.json) is a hand-verified seed.
These providers let the graph *grow from data* instead of being fully hardcoded:

  * price co-movement — peer edges discovered from correlated daily returns in
    the importer's prices CSV (statistical peers, not asserted competitors)
  * article co-occurrence — association edges accrued from entities the LLM
    extracts together across articles, so NOVEL entities (a new airline, a new
    product) wire themselves in over time rather than hitting a wall.

Both feed :func:`graph.build_graph`. They are pure-stdlib and add edges in the
same typed vocabulary the propagation engine already understands
(``comovement`` / ``association`` — the calibratable "cohort" family).
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

try:
    import graph as G
except ImportError:  # pragma: no cover
    from . import graph as G

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price co-movement
# ---------------------------------------------------------------------------

def _load_price_series(prices_csv: str) -> dict[str, dict[str, float]]:
    """{ticker: {date: close}} from the importer's prices CSV (stdlib only)."""
    series: dict[str, dict[str, float]] = defaultdict(dict)
    p = Path(prices_csv)
    if not p.exists():
        logger.warning("prices CSV not found for co-movement: %s", prices_csv)
        return series
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
    return series


def _returns(dates_prices: dict[str, float]) -> dict[str, float]:
    """Daily simple returns keyed by the later date."""
    items = sorted(dates_prices.items())
    out = {}
    for (_, p0), (d1, p1) in zip(items, items[1:]):
        if p0:
            out[d1] = (p1 - p0) / p0
    return out


def _pearson(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    """Correlation of two return series over their common dates; (corr, n)."""
    common = a.keys() & b.keys()
    n = len(common)
    if n < 2:
        return 0.0, n
    xs = [a[d] for d in common]
    ys = [b[d] for d in common]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0, n
    return cov / math.sqrt(vx * vy), n


def comovement_edges(prices_csv: str, min_abs_corr: float = 0.5,
                     min_overlap: int = 30, max_per_node: int = 6,
                     as_of: str | None = None) -> list[tuple[str, str, float]]:
    """
    Discover peer edges from correlated returns. Returns (a, b, weight) with
    weight = |corr| for pairs whose positive return-correlation clears
    ``min_abs_corr`` over at least ``min_overlap`` shared days. Each ticker keeps
    its strongest ``max_per_node`` links to bound density.

    ``as_of`` (YYYY-MM-DD) makes the discovery POINT-IN-TIME: only prices strictly
    before that date are used. Pass the event date when building a graph to score
    a past event, so co-movement carries no lookahead leakage.
    """
    series = _load_price_series(prices_csv)
    if as_of:
        series = {t: {d: p for d, p in dp.items() if d < as_of} for t, dp in series.items()}
    rets = {t: _returns(dp) for t, dp in series.items() if len(dp) > min_overlap}
    tickers = sorted(rets)
    scored: list[tuple[float, str, str]] = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            corr, n = _pearson(rets[a], rets[b])
            if n >= min_overlap and corr >= min_abs_corr:
                scored.append((corr, a, b))

    # keep each node's strongest few
    kept: dict[str, int] = defaultdict(int)
    edges: list[tuple[str, str, float]] = []
    for corr, a, b in sorted(scored, reverse=True):
        if kept[a] >= max_per_node or kept[b] >= max_per_node:
            continue
        kept[a] += 1
        kept[b] += 1
        edges.append((a, b, round(corr, 4)))
    logger.info("co-movement: %d peer edges from %d tickers", len(edges), len(tickers))
    return edges


def add_comovement_edges(graph, prices_csv: str, **kwargs) -> int:
    """Add discovered co-movement edges (bidirectional, type ``comovement``) to a
    graph. Returns the number of edges added."""
    edges = comovement_edges(prices_csv, **kwargs)
    for a, b, w in edges:
        if a in graph.kind and b in graph.kind:
            graph.add_edge(a, b, "comovement", w)
            graph.add_edge(b, a, "comovement", w)
    return len(edges)


# ---------------------------------------------------------------------------
# Article co-occurrence (self-population)
# ---------------------------------------------------------------------------

def _iter_predictions(paths: list[str]):
    for pattern in paths:
        for fp in sorted(glob.glob(pattern)):
            try:
                with open(fp, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        pred = item.get("prediction") if isinstance(item, dict) and "prediction" in item else item
                        if isinstance(pred, dict):
                            yield pred
            except Exception:
                logger.exception("failed reading extractions %s", fp)


def _date_prefix(s) -> "datetime.date | None":
    from datetime import datetime
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def cooccurrence_edges(paths: list[str], min_count: float = 2,
                       weight_scale: float = 3.0, max_weight: float = 0.8,
                       as_of: str | None = None, max_age_days: int | None = None,
                       half_life_days: float | None = None
                       ) -> list[tuple[str, str, float, dict]]:
    """
    Accrue association edges from entities the LLM names together. For every
    prediction, resolved source_entities (+ directly_affected tickers) are
    pairwise linked; weighted counts accumulate across articles.

    AGING (so stale associations decay, not persist forever):
      * ``max_age_days`` — ignore predictions older than this (relative to
        ``as_of`` or today).
      * ``half_life_days`` — weight each co-occurrence by 0.5**(age/half_life),
        so recent news dominates. ``min_count`` then applies to the summed weight.
    """
    from datetime import date, timedelta
    ref = _date_prefix(as_of) or date.today()

    counts: dict[tuple[str, str], float] = defaultdict(float)
    node_kind: dict[str, str] = {}

    for pred in _iter_predictions(paths):
        adate = _date_prefix(pred.get("article_date")
                             or (pred.get("data_quality") or {}).get("article_date"))
        age = (ref - adate).days if adate else 0
        if max_age_days is not None and adate is not None and age > max_age_days:
            continue
        weight = 1.0
        if half_life_days and adate is not None and age > 0:
            weight = 0.5 ** (age / half_life_days)

        nodes = []
        for ent in pred.get("source_entities", []) or []:
            name = ent.get("name")
            if not name:
                continue
            kind = (ent.get("kind") or "company").lower()
            node = G.resolve_entity_node(name, kind)
            node_kind[node] = "product" if node.startswith("product:") else \
                              "driver" if node.startswith("driver:") else "company"
            nodes.append(node)
        for ent in pred.get("directly_affected", []) or []:
            t = ent.get("ticker")
            if t:
                node_kind[t] = "company"
                nodes.append(t)
        uniq = sorted(set(nodes))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1:]:
                counts[(a, b)] += weight

    edges = []
    for (a, b), c in counts.items():
        if c >= min_count:
            w = min(c / weight_scale, max_weight)
            edges.append((a, b, round(w, 4), {"kind_a": node_kind.get(a, "company"),
                                              "kind_b": node_kind.get(b, "company"),
                                              "count": round(c, 3)}))
    logger.info("co-occurrence: %d association edges from extractions", len(edges))
    return edges


def validate_driver_exposures(prices_csv: str, min_overlap: int = 20) -> list[dict]:
    """
    Data-driven sanity check on the HAND-ASSIGNED driver exposures in
    graph_data.json. For each driver, the ``helps_when_up`` firms should co-move
    as a group and the ``hurts_when_up`` firms as another, with the two groups
    LESS correlated with each other (since the driver pushes them oppositely).

    Returns one report per driver with mean within-group vs cross-group return
    correlation and a ``suspect`` flag when the grouping doesn't separate. This
    does NOT prove the signs are right (that needs a driver time series) — it just
    flags exposures that the price data contradicts. Honest guardrail, not truth.
    """
    series = _load_price_series(prices_csv)
    rets = {t: _returns(dp) for t, dp in series.items() if len(dp) > min_overlap}

    def mean_corr(group_a, group_b, same):
        vals = []
        for i, a in enumerate(group_a):
            for j, b in enumerate(group_b):
                if same and j <= i:
                    continue
                if a in rets and b in rets and a != b:
                    c, n = _pearson(rets[a], rets[b])
                    if n >= min_overlap:
                        vals.append(c)
        return sum(vals) / len(vals) if vals else None

    reports = []
    for d in G.DRIVERS:
        helps = [G._canonical(x) for x in d.get("helps_when_up", [])]
        hurts = [G._canonical(x) for x in d.get("hurts_when_up", [])]
        within = [c for c in (mean_corr(helps, helps, True), mean_corr(hurts, hurts, True)) if c is not None]
        within_avg = sum(within) / len(within) if within else None
        cross = mean_corr(helps, hurts, False) if (helps and hurts) else None
        suspect = (within_avg is not None and cross is not None and cross >= within_avg)
        reports.append({
            "driver": d["driver"],
            "within_group_corr": round(within_avg, 3) if within_avg is not None else None,
            "cross_group_corr": round(cross, 3) if cross is not None else None,
            "suspect": suspect,
            "note": "groups don't separate — check the sign assignment" if suspect else "ok",
        })
    return reports


def add_cooccurrence_edges(graph, paths: list[str], tradeable_lookup=None, **kwargs) -> int:
    """Add association edges from extractions, registering any NEW entity nodes
    (so novel companies/products wire themselves in). Returns edges added."""
    tradeable_lookup = tradeable_lookup or (lambda n: n in G.VALID_TICKERS)
    edges = cooccurrence_edges(paths, **kwargs)
    for a, b, w, meta in edges:
        if a not in graph.kind:
            graph.add_node(a, meta["kind_a"], tradeable=tradeable_lookup(a))
        if b not in graph.kind:
            graph.add_node(b, meta["kind_b"], tradeable=tradeable_lookup(b))
        graph.add_edge(a, b, "association", w)
        graph.add_edge(b, a, "association", w)
    return len(edges)
