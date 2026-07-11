"""
graph.py — Knowledge graph + impact-propagation engine.

This is the "reasoning" half of the system. The extractor/LLM is the
"perception" half: it reads an article and says *what happened, to whom, and
how badly* (the source event). This module knows *who else is affected* and
propagates the shock outward through a typed knowledge graph, attenuating
direction / magnitude / confidence at every hop.

Worked example (the motivating case from the spec)::

    "Ethiopian Airlines Boeing 737 MAX crashes"
              │  LLM extracts: source = Ethiopian Airlines, event = disaster,
              │                severity = 1.0, direction = DOWN
              ▼
        product: Boeing 737 MAX   (Ethiopian *operates* it)
              │
      ┌───────┴──────────────┐
      ▼                       ▼
   Boeing (made_by)      Kenya Airways / KQ  (also *operates* it → shared-fleet
                          contagion → perceived riskier → DOWN)

Nothing in the LLM prompt needs to know KQ flies Boeing. The *graph* knows,
through the shared ``product:Boeing 737 MAX`` node.

Design
------
* Nodes are typed: companies (NSE-listed or not), sectors, products.
* Edges are typed and carry a base coupling ``weight`` in (0, 1]. The *sign* of
  an impact is not stored on the edge — it comes from the event channel
  (``CHANNEL``), because the same structural relationship transmits impact
  differently depending on the event (a rival's earnings *beat* is mildly bad
  for you; a rival's factory *fire* is mildly good — or, via a shared product,
  bad). Keeping sign in one place (per event × edge-type) makes the model
  auditable.
* Propagation is a signed breadth-first spread with per-hop decay and a
  minimum-shock cutoff, so it always terminates.

The module is pure standard library so it can be unit-tested in isolation.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Structural data for NSE-listed names lives in companies.py (the audited source
# of truth). We layer the non-NSE / product / supplier edges on top here.
try:  # allow both `import graph` (from pipeline/) and `pipeline.graph`
    from companies import NSE_COMPANIES, COMPETITOR_RELATIONSHIPS, VALID_TICKERS
except ImportError:  # pragma: no cover - packaging convenience
    from .companies import NSE_COMPANIES, COMPETITOR_RELATIONSHIPS, VALID_TICKERS


# ---------------------------------------------------------------------------
# Tunable coefficients (calibratable later against realised prices)
# ---------------------------------------------------------------------------

#: Per-hop multiplicative decay applied on top of edge weights.
HOP_DECAY = 0.75

#: Stop spreading once |accumulated shock| falls below this.
MIN_SHOCK = 0.02

#: Hard cap on path length from a source node.
MAX_HOPS = 4

#: A full unit shock (|shock| == 1.0) maps to this percentage price move.
MAGNITUDE_SCALE = 6.0

#: |predicted move %| below this is reported as NEUTRAL.
DIRECTION_THRESHOLD = 0.5

#: calibrate.py writes fitted coefficients here; they override the defaults
#: above at import time so tuning does not require editing source.
CALIBRATION_FILE = Path(__file__).with_name("calibration.json")

# Keys in calibration.json that may override the module constants above.
# CHANNEL_GAINS is applied last because it is defined further down the module.
_CALIBRATABLE = (
    "HOP_DECAY", "MIN_SHOCK", "MAX_HOPS", "MAGNITUDE_SCALE",
    "DIRECTION_THRESHOLD", "CHANNEL_GAINS",
)


def load_calibration(path: Path = CALIBRATION_FILE) -> dict:
    """Load fitted coefficients from calibration.json and apply them as module
    globals. Returns the applied overrides (empty dict if the file is absent).

    Scalar keys replace the global; ``CHANNEL_GAINS`` is *merged* into the
    existing defaults so a partial gains dict keeps unspecified families at 1.0.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    applied = {}
    g = globals()
    for key in _CALIBRATABLE:
        if key not in data:
            continue
        if key == "CHANNEL_GAINS" and isinstance(data[key], dict):
            merged = dict(g.get("CHANNEL_GAINS", {}))
            merged.update(data[key])
            g["CHANNEL_GAINS"] = merged
            applied[key] = merged
        else:
            g[key] = data[key]
            applied[key] = data[key]
    return applied


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    dst: str
    etype: str
    weight: float  # base coupling strength in (0, 1]


@dataclass
class KnowledgeGraph:
    # adjacency: node -> list[Edge]
    adj: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))
    # node -> kind ("company" | "sector" | "product")
    kind: dict[str, str] = field(default_factory=dict)
    # subset of company nodes that are tradeable on the NSE (get predictions)
    tradeable: set[str] = field(default_factory=set)

    def add_node(self, node: str, kind: str, tradeable: bool = False) -> None:
        self.kind.setdefault(node, kind)
        if tradeable:
            self.tradeable.add(node)

    def add_edge(self, src: str, dst: str, etype: str, weight: float) -> None:
        self.adj[src].append(Edge(dst=dst, etype=etype, weight=weight))

    def add_biedge(
        self, a: str, b: str, etype_ab: str, etype_ba: str, weight: float
    ) -> None:
        """Add a directed edge each way (types may differ, e.g. operates/operated_by)."""
        self.add_edge(a, b, etype_ab, weight)
        self.add_edge(b, a, etype_ba, weight)

    def neighbors(self, node: str) -> list[Edge]:
        return self.adj.get(node, [])

    def nodes(self) -> Iterable[str]:
        return self.kind.keys()


# ---------------------------------------------------------------------------
# Event channels: how each event type transmits along each edge type.
#
# Value is a SIGNED multiplier. Positive = same direction as the source shock,
# negative = opposite direction, 0 / absent = channel inactive for this event.
# ``in_sector`` / ``operates`` / ``made_by`` etc. are routing edges kept active
# so the spread can reach the nodes that matter.
# ---------------------------------------------------------------------------

# Applied to every event unless the event overrides the key. Routing edges live
# here so we never forget to keep a structural hop open.
DEFAULT_CHANNEL: dict[str, float] = {
    "in_sector": 1.0,       # company -> its sector hub (pure routing)
    "has_member": 0.20,     # sector hub -> peer company (spillover, weak)
    "competitor": 0.20,
    "operates": 0.30,       # company -> product it uses
    "operated_by": 0.60,    # product -> a company that uses it (shared-fleet)
    "manufactures": 0.40,   # maker -> product it builds
    "made_by": 0.70,        # product -> its maker
    "supplies": 0.35,       # supplier -> customer
    "supplied_by": 0.25,    # customer -> supplier
    # driver edges are inactive unless the event type explicitly enables them,
    # so an ordinary company event does not fan out through macro/commodity hubs.
    "helps_when_up": 0.0,
    "hurts_when_up": 0.0,
    "moved_by": 0.0,
    # DATA-DERIVED edges (not curated): price co-movement and article co-occurrence.
    # Same-direction, weak — statistical association, always mildly active.
    "comovement": 0.15,
    "association": 0.10,
}

CHANNEL: dict[str, dict[str, float]] = {
    # Crashes, fires, recalls, safety incidents. Impact rides the *product*:
    # the implicated product drags down its maker and everyone who operates it
    # (shared-fleet contagion). Rivals barely matter as a direct channel.
    "disaster": {
        "operates": 0.35,
        "operated_by": 0.75,
        "manufactures": 0.30,
        "made_by": 0.80,
        "supplies": 0.30,
        "supplied_by": 0.20,
        "has_member": 0.12,
        "competitor": 0.05,
    },
    # A rival beating expectations is mildly *negative* for you (relative
    # performance / rotation). Sign flips on the competitor channel.
    "earnings": {
        "competitor": -0.45,
        "has_member": 0.15,
    },
    # Sector-wide rules move the whole sector the same way.
    "regulation": {
        "has_member": 0.55,
        "competitor": 0.30,
    },
    "merger_acquisition": {
        "competitor": -0.30,
        "has_member": 0.10,
    },
    "product_launch": {
        "competitor": -0.40,
        "has_member": 0.10,
        "supplied_by": 0.25,
    },
    "legal": {
        "competitor": -0.15,
        "has_member": 0.10,
    },
    # Interest rates, FX, GDP, fiscal policy. Fans out from a macro DRIVER node
    # to exposed firms: same direction for those the driver helps, opposite for
    # those it hurts. The source direction is the DRIVER'S move (rate up = UP,
    # KES weakening = UP on a "KES/USD" node).
    "macro": {
        "helps_when_up": 0.70,
        "hurts_when_up": -0.70,
        "has_member": 0.25,
    },
    # Commodity / shared-input price moves (oil, tea, cement inputs). Same
    # polarity structure as macro: producers/sellers helped, consumers hurt.
    "commodity": {
        "helps_when_up": 0.75,
        "hurts_when_up": -0.75,
        "has_member": 0.10,
    },
}


def channel_multiplier(event_type: str, etype: str) -> float:
    """Signed transmission multiplier for (event_type, edge_type)."""
    overrides = CHANNEL.get(event_type or "", {})
    if etype in overrides:
        return overrides[etype]
    return DEFAULT_CHANNEL.get(etype, 0.0)


# Edge types grouped into a handful of channel *families*. Calibration fits one
# global gain per family (not per event×edge cell — that would over-fit), so the
# hand-set sign structure in CHANNEL is preserved while the overall strength of
# each transmission pathway is scaled to data.
EDGE_FAMILY: dict[str, str] = {
    "in_sector": "sector",
    "has_member": "sector",
    "competitor": "competitor",
    "operates": "product",
    "operated_by": "product",
    "manufactures": "product",
    "made_by": "product",
    "supplies": "supplier",
    "supplied_by": "supplier",
    # driver nodes (macro / commodity / shared input) — the sign is carried by
    # the edge TYPE: a firm helped when the driver rises vs one hurt by it.
    "helps_when_up": "driver",
    "hurts_when_up": "driver",
    "moved_by": "driver",
    # data-derived statistical edges share one calibratable family
    "comovement": "cohort",
    "association": "cohort",
}

#: Per-family multiplicative gain (fitted by calibrate.py). 1.0 = engine default.
CHANNEL_GAINS: dict[str, float] = {
    "sector": 1.0,
    "competitor": 1.0,
    "product": 1.0,
    "supplier": 1.0,
    "driver": 1.0,
    "cohort": 1.0,
}


def family_gain(etype: str, gains: dict[str, float] | None = None) -> float:
    """Return the family gain for an edge type (1.0 for unknown families)."""
    g = CHANNEL_GAINS if gains is None else gains
    return g.get(EDGE_FAMILY.get(etype, "other"), 1.0)


# ---------------------------------------------------------------------------
# Supplemental (non-NSE / product / supplier / driver) structure
#
# This is DATA, not code — it lives in graph_data.json next to this module and
# is loaded at import. Grow the curated graph by editing that file (or by
# pointing GRAPH_DATA_FILE elsewhere); no code changes needed. The same file is
# the single source consumed by production's knowledge-graph loader.
#
#   products  : [{"product", "made_by": [...], "operated_by": [...]}]
#   aliases   : {non-NSE name -> NSE ticker it resolves to}
#   suppliers : [[supplier, customer], ...]   (directed, supplier first)
#   drivers   : [{"driver", "kind", "helps_when_up": [...], "hurts_when_up": [...]}]
# ---------------------------------------------------------------------------

import os

GRAPH_DATA_FILE = Path(os.environ.get(
    "GRAPH_DATA_FILE", str(Path(__file__).with_name("graph_data.json"))
))


def load_graph_data(path: Path = None) -> dict:
    """Load the curated supplemental graph structure from JSON.

    Returns a dict with keys products / aliases / suppliers / drivers. Missing
    file or bad JSON yields empty structure (companies-only graph) plus a warning
    — the engine still runs, just without products/drivers/supplier edges.
    """
    path = path or GRAPH_DATA_FILE
    empty = {"products": [], "aliases": {}, "suppliers": [], "drivers": []}
    if not path.exists():
        import logging
        logging.getLogger(__name__).warning("graph_data.json not found at %s; using companies-only graph", path)
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to parse %s; using companies-only graph", path)
        return empty
    return {
        "products": data.get("products", []),
        "aliases": data.get("aliases", {}),
        "suppliers": [tuple(s) for s in data.get("suppliers", [])],
        "drivers": data.get("drivers", []),
    }


_CURATED = load_graph_data()
PRODUCTS: list[dict] = _CURATED["products"]
NON_NSE_ALIASES: dict[str, str] = _CURATED["aliases"]
SUPPLIER_RELATIONSHIPS: list[tuple[str, str]] = _CURATED["suppliers"]
DRIVERS: list[dict] = _CURATED["drivers"]

#: Known driver names — derived from the data so the extractor / callers stay in
#: sync with whatever drivers the curated file defines.
DRIVER_NAMES: list[str] = [d["driver"] for d in DRIVERS]


def _canonical(name: str) -> str:
    """Resolve a raw company name/ticker to its canonical graph node id."""
    return NON_NSE_ALIASES.get(name, name)


def resolve_entity_node(name: str, kind: str) -> str:
    """Map an extracted entity (name + kind) to its graph node id. Shared by the
    source-seeding path and the data-derived edge providers so they agree."""
    kind = (kind or "company").lower()
    if kind == "product":
        return f"product:{name}"
    if kind in ("driver", "macro", "commodity"):
        return f"driver:{name}"
    return _canonical(name)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Base coupling weights per structural relationship (magnitude only; sign comes
# from the event channel).
W_SECTOR = 0.9          # company <-> sector hub routing weight
W_COMPETITOR = 0.5
W_PRODUCT = 0.8
W_SUPPLIER = 0.6
W_DRIVER = 0.7          # driver hub -> exposed firm


def build_default_graph() -> KnowledgeGraph:
    """Assemble the full knowledge graph from companies.py + supplemental data."""
    g = KnowledgeGraph()

    # --- companies + sector hubs -----------------------------------------
    for c in NSE_COMPANIES:
        ticker = c["ticker"]
        g.add_node(ticker, "company", tradeable=True)
        sector_node = f"sector:{c['sector']}"
        g.add_node(sector_node, "sector")
        # company -> sector (routing, strong) ; sector -> company (spillover)
        g.add_biedge(ticker, sector_node, "in_sector", "has_member", W_SECTOR)

    # --- competitor edges -------------------------------------------------
    for a, b, _rel in COMPETITOR_RELATIONSHIPS:
        na, nb = _canonical(a), _canonical(b)
        g.add_node(na, "company", tradeable=na in VALID_TICKERS)
        g.add_node(nb, "company", tradeable=nb in VALID_TICKERS)
        # symmetric competitor coupling
        g.add_edge(na, nb, "competitor", W_COMPETITOR)
        g.add_edge(nb, na, "competitor", W_COMPETITOR)

    # --- products (maker / operator edges) --------------------------------
    for p in PRODUCTS:
        pnode = f"product:{p['product']}"
        g.add_node(pnode, "product")
        for maker in p.get("made_by", []):
            m = _canonical(maker)
            g.add_node(m, "company", tradeable=m in VALID_TICKERS)
            # maker -> product (manufactures), product -> maker (made_by)
            g.add_biedge(m, pnode, "manufactures", "made_by", W_PRODUCT)
        for user in p.get("operated_by", []):
            u = _canonical(user)
            g.add_node(u, "company", tradeable=u in VALID_TICKERS)
            # user -> product (operates), product -> user (operated_by)
            g.add_biedge(u, pnode, "operates", "operated_by", W_PRODUCT)

    # --- supplier chains --------------------------------------------------
    for supplier, customer in SUPPLIER_RELATIONSHIPS:
        s, cust = _canonical(supplier), _canonical(customer)
        g.add_node(s, "company", tradeable=s in VALID_TICKERS)
        g.add_node(cust, "company", tradeable=cust in VALID_TICKERS)
        g.add_biedge(s, cust, "supplies", "supplied_by", W_SUPPLIER)

    # --- driver hubs (macro / commodity / shared input) -------------------
    for d in DRIVERS:
        dnode = f"driver:{d['driver']}"
        g.add_node(dnode, "driver")
        for firm in d.get("helps_when_up", []):
            f = _canonical(firm)
            g.add_node(f, "company", tradeable=f in VALID_TICKERS)
            # driver -> firm (helps_when_up) ; firm -> driver (moved_by, routing)
            g.add_biedge(dnode, f, "helps_when_up", "moved_by", W_DRIVER)
        for firm in d.get("hurts_when_up", []):
            f = _canonical(firm)
            g.add_node(f, "company", tradeable=f in VALID_TICKERS)
            g.add_biedge(dnode, f, "hurts_when_up", "moved_by", W_DRIVER)

    return g


def build_graph(prices_csv: str | None = None,
                extraction_paths: list[str] | None = None,
                comovement_kwargs: dict | None = None,
                cooccurrence_kwargs: dict | None = None) -> KnowledgeGraph:
    """
    Build the graph from the curated seed, then optionally layer on DATA-DERIVED
    edges so the graph grows from evidence rather than being fully hardcoded:

      * ``prices_csv``       -> co-movement peer edges (correlated returns)
      * ``extraction_paths`` -> association edges from LLM co-occurrence, which
                                also register novel entities not in the seed.

    With no arguments this equals :func:`build_default_graph` (curated only), so
    existing callers/tests are unaffected.

    PRODUCTION CAVEATS (not yet handled — do not treat as solved):
    * Co-movement edges are computed from the WHOLE ``prices_csv``. That is fine
      for live/forward prediction (you only have past prices), but it is
      LOOKAHEAD LEAKAGE if you build one graph from all history and then score
      PAST events with it. A rigorous backtest must recompute co-movement using
      only data before each event. The harness is point-in-time for prices but
      NOT for these graph edges.
    * Article co-occurrence edges never age — a one-off co-occurrence persists
      forever (guarded only by min_count). Production should add recency
      weighting / expiry so stale associations decay.
    """
    g = build_default_graph()
    if prices_csv or extraction_paths:
        import graph_sources as GS  # lazy: keeps base import light
        if prices_csv:
            GS.add_comovement_edges(g, prices_csv, **(comovement_kwargs or {}))
        if extraction_paths:
            GS.add_cooccurrence_edges(g, extraction_paths, **(cooccurrence_kwargs or {}))
    return g


def export_graph(path: str | Path | None = None, **build_kwargs) -> Path:
    """
    Serialise the full graph (companies + nodes + edges) to a single JSON
    artifact. This is the canonical hand-off consumed by production's
    knowledge-graph loader, so the pipeline and the production DB share ONE
    source of truth instead of maintaining separate hardcoded copies.
    """
    g = build_graph(**build_kwargs)
    data = {
        "companies": [
            {"symbol": c["ticker"], "name": c["name"], "sector": c["sector"]}
            for c in NSE_COMPANIES
        ],
        "nodes": [
            {"id": n, "kind": g.kind[n], "tradeable": n in g.tradeable}
            for n in g.kind
        ],
        "edges": [
            {"src": src, "dst": e.dst, "etype": e.etype, "weight": e.weight}
            for src, elist in g.adj.items() for e in elist
        ],
    }
    out = Path(path) if path else Path(__file__).with_name("graph_export.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Shock model
# ---------------------------------------------------------------------------

_DIR_SIGN = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}


def shock_from_prediction(direction: str, severity: float = 0.5) -> float:
    """
    Convert an LLM-extracted (direction, severity) into a signed source shock
    in [-1, 1]. Severity scales magnitude; direction sets the sign.
    """
    sign = _DIR_SIGN.get((direction or "").upper(), 0.0)
    try:
        sev = float(severity)
    except (TypeError, ValueError):
        sev = 0.5
    sev = max(0.0, min(1.0, sev))
    return sign * sev


def _shock_to_direction(pct: float) -> str:
    if pct >= DIRECTION_THRESHOLD:
        return "UP"
    if pct <= -DIRECTION_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


@dataclass
class Impact:
    node: str
    shock: float          # signed accumulated shock in ~[-1, 1]
    confidence: float     # in [0, 1]
    direction: str        # UP / DOWN / NEUTRAL
    magnitude_pct: float  # signed predicted move, e.g. -3.5
    hops: int
    path: list[str]


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate(
    graph: KnowledgeGraph,
    sources: dict[str, float],
    event_type: str,
    base_confidence: float = 1.0,
    max_hops: int | None = None,
    hop_decay: float | None = None,
    min_shock: float | None = None,
    gains: dict[str, float] | None = None,
    magnitude_scale: float | None = None,
) -> dict[str, Impact]:
    """
    Spread signed shocks from ``sources`` through ``graph`` for a given
    ``event_type``.

    Parameters
    ----------
    sources
        {node_id: signed_shock} — the directly affected entities and their
        source shocks (see :func:`shock_from_prediction`). Node ids may be
        tickers or non-NSE names (they are canonicalised).
    base_confidence
        Confidence of the source extraction, in [0, 1]. Downstream confidence
        is this times the accumulated coupling magnitude.
    max_hops, hop_decay, min_shock, gains, magnitude_scale
        Propagation coefficients. ``None`` (the default) resolves to the module
        globals — which may themselves have been overridden by calibration.
        Passing them explicitly lets ``calibrate.py`` score candidate values
        without mutating global state.

    Returns
    -------
    dict[node_id, Impact] for every reached node *except* the sources
    themselves, keeping the strongest (largest |shock|) impact per node.
    """
    # Resolve coefficients from globals at call time (calibration-aware).
    max_hops = MAX_HOPS if max_hops is None else max_hops
    hop_decay = HOP_DECAY if hop_decay is None else hop_decay
    min_shock = MIN_SHOCK if min_shock is None else min_shock
    scale = MAGNITUDE_SCALE if magnitude_scale is None else magnitude_scale

    best: dict[str, Impact] = {}

    for raw_src, src_shock in sources.items():
        src = _canonical(raw_src)
        if src not in graph.kind:
            # unknown source — nothing to propagate through
            continue
        if abs(src_shock) < min_shock:
            continue

        # BFS carrying (node, signed_shock, coupling_magnitude, hops, path)
        start = (src, float(src_shock), 1.0, 0, [src])
        queue: deque = deque([start])
        # remember best |shock| reached per node *within this source's spread*
        seen_best: dict[str, float] = {src: abs(src_shock)}

        while queue:
            node, shock, coupling, hops, path = queue.popleft()
            if hops >= max_hops:
                continue
            for edge in graph.neighbors(node):
                mult = channel_multiplier(event_type, edge.etype)
                if mult == 0.0:
                    continue
                gain = family_gain(edge.etype, gains)
                if gain == 0.0:
                    continue
                transfer = edge.weight * mult * gain * hop_decay
                child_shock = shock * transfer
                if abs(child_shock) < min_shock:
                    continue
                dst = edge.dst
                # don't loop back onto the path
                if dst in path:
                    continue
                prev = seen_best.get(dst, 0.0)
                if abs(child_shock) <= prev:
                    continue
                seen_best[dst] = abs(child_shock)
                child_coupling = coupling * abs(transfer)
                child_path = path + [dst]
                queue.append((dst, child_shock, child_coupling, hops + 1, child_path))

                # record impact for tradeable companies that aren't a source
                if dst in graph.tradeable and dst not in sources and dst != src:
                    conf = max(0.0, min(1.0, base_confidence * child_coupling))
                    pct = child_shock * scale
                    imp = Impact(
                        node=dst,
                        shock=child_shock,
                        confidence=round(conf, 4),
                        direction=_shock_to_direction(pct),
                        magnitude_pct=round(pct, 3),
                        hops=hops + 1,
                        path=child_path,
                    )
                    keep = best.get(dst)
                    if keep is None or abs(imp.shock) > abs(keep.shock):
                        best[dst] = imp

    return best


# ---------------------------------------------------------------------------
# Prediction-level convenience wrapper
# ---------------------------------------------------------------------------

def collect_sources(
    prediction: dict,
    graph: KnowledgeGraph,
) -> tuple[dict[str, float], float]:
    """
    Resolve the propagation seeds for a prediction.

    Returns ``(sources, base_confidence)`` where ``sources`` maps graph node ids
    to signed source shocks, seeded from both ``directly_affected`` (NSE tickers)
    and ``source_entities`` (raw companies/products, NSE-or-not). Nodes absent
    from the graph are skipped. This is shared by :func:`enrich_prediction` and
    the calibration harness so both seed identically.
    """
    default_sev = float(prediction.get("severity", 0.5) or 0.5)
    sources: dict[str, float] = {}
    source_conf: list[float] = []

    def _seed(node: str, direction: str, severity: float) -> None:
        shock = shock_from_prediction(direction, severity)
        if shock == 0.0:
            return
        if node not in sources or abs(shock) > abs(sources[node]):
            sources[node] = shock

    # Seed 1: NSE tickers the event is directly about.
    for ent in prediction.get("directly_affected", []) or []:
        ticker = ent.get("ticker")
        if not ticker or ticker not in graph.kind:
            continue
        sev = float(ent.get("severity", default_sev) or default_sev)
        _seed(ticker, ent.get("direction", "NEUTRAL"), sev)
        if isinstance(ent.get("confidence"), (int, float)):
            source_conf.append(float(ent["confidence"]))

    # Seed 2: raw named entities (companies NSE-or-not, and products) — how a
    # foreign/unlisted event enters the graph and reaches NSE names.
    for ent in prediction.get("source_entities", []) or []:
        name = ent.get("name")
        if not name:
            continue
        node = resolve_entity_node(name, ent.get("kind"))
        if node not in graph.kind:
            continue
        sev = float(ent.get("severity", default_sev) or default_sev)
        _seed(node, ent.get("direction", "NEUTRAL"), sev)

    base_conf = sum(source_conf) / len(source_conf) if source_conf else 1.0
    return sources, base_conf


def enrich_prediction(
    prediction: dict,
    graph: KnowledgeGraph | None = None,
) -> dict:
    """
    Take an extractor-style prediction whose ``directly_affected`` list is the
    LLM-perceived source event, and compute ``indirectly_affected`` by graph
    propagation (replacing whatever the LLM guessed there).

    The source entities are taken from ``directly_affected`` and
    ``source_entities``; each contributes a shock derived from its direction and
    the prediction's ``severity`` (or a per-entity ``severity`` if present, else
    0.5). ``event_type`` selects the channel. The returned dict is a shallow copy
    with a rebuilt ``indirectly_affected`` and a ``propagation`` audit block.
    """
    graph = graph or build_default_graph()
    event_type = prediction.get("event_type", "")
    sources, base_conf = collect_sources(prediction, graph)
    impacts = propagate(graph, sources, event_type, base_confidence=base_conf)

    indirect = []
    for imp in sorted(impacts.values(), key=lambda i: abs(i.shock), reverse=True):
        if imp.direction == "NEUTRAL":
            continue
        indirect.append(
            {
                "ticker": imp.node,
                "impact_type": _impact_type_for(imp),
                "direction": imp.direction,
                "confidence": imp.confidence,
                "magnitude_pct": imp.magnitude_pct,
                "reasoning": f"Graph propagation via {' -> '.join(imp.path)}",
                "hops": imp.hops,
            }
        )

    out = dict(prediction)
    out["indirectly_affected"] = indirect
    out["propagation"] = {
        "event_type": event_type,
        "sources": list(sources.keys()),
        "reached": len(impacts),
    }
    return out


_ETYPE_HINT = {
    "operated_by": "supplier_chain",
    "made_by": "supplier_chain",
    "manufactures": "supplier_chain",
    "operates": "supplier_chain",
    "supplies": "supplier_chain",
    "supplied_by": "supplier_chain",
    "competitor": "competitor",
    "has_member": "sector_spillover",
    "in_sector": "sector_spillover",
}


def _impact_type_for(imp: Impact) -> str:
    """Infer the pipeline impact_type label from the last edge traversed."""
    # We don't store edge types on the path; approximate from hop count and the
    # node kinds along the path. A path through a product/sector hub is a
    # spillover/supplier channel; a direct company->company hop is competitor.
    if any(n.startswith("driver:") for n in imp.path):
        return "regulatory"      # macro / commodity / policy driver
    if any(n.startswith("product:") for n in imp.path):
        return "supplier_chain"
    if any(n.startswith("sector:") for n in imp.path):
        return "sector_spillover"
    return "competitor"


# Apply any persisted calibration now that every calibratable global
# (including CHANNEL_GAINS) is defined. propagate() resolves coefficients from
# these globals at call time, so this override reaches every code path.
load_calibration()
