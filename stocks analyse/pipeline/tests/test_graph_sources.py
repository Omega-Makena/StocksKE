"""Tests for the data-derived edge providers (price co-movement + co-occurrence)."""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import graph as G
import graph_sources as GS


# ---------------------------------------------------------------------------
# Price co-movement
# ---------------------------------------------------------------------------

def _write_prices(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Stock Code", "Date", "Day's Final Price"])
        w.writerows(rows)


def test_comovement_discovers_correlated_pair(tmp_path):
    csvp = tmp_path / "prices.csv"
    rows = []
    # KCB and EQTY move together; BAT moves oppositely
    base = 100.0
    for i in range(40):
        d = f"2024-01-{i+1:02d}"
        bump = 1 + (0.02 if i % 2 == 0 else -0.02)
        kcb = base * bump
        eqty = base * bump * 1.5
        bat = base * (1 - (0.02 if i % 2 == 0 else -0.02))
        rows += [["KCB", d, f"{kcb:.2f}"], ["EQTY", d, f"{eqty:.2f}"], ["BAT", d, f"{bat:.2f}"]]
    _write_prices(csvp, rows)

    edges = GS.comovement_edges(str(csvp), min_abs_corr=0.5, min_overlap=10)
    pairs = {frozenset((a, b)) for a, b, _ in edges}
    assert frozenset(("KCB", "EQTY")) in pairs      # positively correlated
    # BAT is anti-correlated -> not a positive co-movement edge with KCB
    assert frozenset(("KCB", "BAT")) not in pairs


def test_add_comovement_edges_transmit_same_direction(tmp_path):
    csvp = tmp_path / "prices.csv"
    rows = []
    for i in range(40):
        d = f"2024-02-{i+1:02d}"
        bump = 1 + (0.03 if i % 2 == 0 else -0.03)
        rows += [["KUKZ", d, f"{100*bump:.2f}"], ["SASN", d, f"{50*bump:.2f}"]]
    _write_prices(csvp, rows)

    g = G.build_default_graph()
    before = sum(len(v) for v in g.adj.values())
    n = GS.add_comovement_edges(g, str(csvp), min_abs_corr=0.5, min_overlap=10)
    assert n >= 1
    assert sum(len(v) for v in g.adj.values()) > before
    # a co-movement peer gets same-direction (positive) pressure under a generic
    # event; sign, not the thresholded direction, is the property under test
    imp = G.propagate(g, {"KUKZ": G.shock_from_prediction("UP", 1.0)}, "other")
    assert "SASN" in imp and imp["SASN"].shock > 0


def test_comovement_missing_file_is_safe():
    assert GS.comovement_edges("does_not_exist.csv") == []


# ---------------------------------------------------------------------------
# Article co-occurrence (self-population)
# ---------------------------------------------------------------------------

def _write_extractions(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({"article": {}, "prediction": p}) + "\n")


def test_cooccurrence_accrues_and_registers_new_entity(tmp_path):
    # A novel airline (not in the curated graph) repeatedly co-occurs with a
    # known product + KQ -> it should wire itself in.
    pred = {
        "event_type": "disaster",
        "source_entities": [
            {"name": "Jambojet", "kind": "company", "direction": "DOWN"},
            {"name": "Boeing 737 MAX", "kind": "product", "direction": "DOWN"},
        ],
        "directly_affected": [{"ticker": "KQ", "direction": "DOWN",
                               "impact_type": "direct", "confidence": 0.8}],
    }
    fp = tmp_path / "extractions_1.jsonl"
    _write_extractions(fp, [pred, pred, pred])  # seen 3x -> above min_count

    g = G.build_default_graph()
    assert "Jambojet" not in g.kind
    n = GS.add_cooccurrence_edges(g, [str(fp)], min_count=2)
    assert n >= 1
    assert "Jambojet" in g.kind                 # novel entity registered
    assert g.kind["Jambojet"] == "company"
    assert "Jambojet" not in g.tradeable        # not an NSE ticker


def test_comovement_as_of_is_point_in_time(tmp_path):
    csvp = tmp_path / "prices.csv"
    rows = []
    for i in range(40):
        d = f"2024-05-{i+1:02d}" if i < 31 else f"2024-06-{i-30:02d}"
        bump = 1 + (0.03 if i % 2 == 0 else -0.03)
        rows += [["KCB", d, f"{100*bump:.2f}"], ["EQTY", d, f"{60*bump:.2f}"]]
    _write_prices(csvp, rows)
    # full history finds the pair; an early as_of leaves too few days -> no edge
    full = GS.comovement_edges(str(csvp), min_abs_corr=0.5, min_overlap=15)
    early = GS.comovement_edges(str(csvp), min_abs_corr=0.5, min_overlap=15, as_of="2024-05-10")
    assert any(frozenset((a, b)) == frozenset(("KCB", "EQTY")) for a, b, _ in full)
    assert early == []          # not enough pre-as_of data -> nothing leaks


def test_cooccurrence_max_age_drops_old(tmp_path):
    pair = {"source_entities": [{"name": "Boeing 737 MAX", "kind": "product", "direction": "DOWN"}],
            "directly_affected": [{"ticker": "KQ", "direction": "DOWN", "confidence": 0.8}]}
    old = dict(pair, article_date="2020-01-01")
    recent = dict(pair, article_date="2024-06-01")
    fp = tmp_path / "extractions_1.jsonl"
    _write_extractions(fp, [old, recent])
    # without aging: both count -> edge; with a 90-day window from 2024-06-10 the
    # old one is dropped, leaving < min_count
    assert GS.cooccurrence_edges([str(fp)], min_count=2) != []
    aged = GS.cooccurrence_edges([str(fp)], min_count=2, as_of="2024-06-10", max_age_days=90)
    assert aged == []


def test_cooccurrence_half_life_downweights_old(tmp_path):
    pair = {"source_entities": [{"name": "Boeing 737 MAX", "kind": "product", "direction": "DOWN"}],
            "directly_affected": [{"ticker": "KQ", "direction": "DOWN", "confidence": 0.8}]}
    fp = tmp_path / "extractions_1.jsonl"
    _write_extractions(fp, [dict(pair, article_date="2023-06-10")])  # ~1yr old
    edges = GS.cooccurrence_edges([str(fp)], min_count=0.1, as_of="2024-06-10",
                                  half_life_days=180)
    assert edges and edges[0][3]["count"] < 1.0      # decayed below a fresh 1.0


def test_validate_driver_exposures_runs(tmp_path):
    # give the two CBK-rate groups clearly separable price behaviour
    csvp = tmp_path / "prices.csv"
    rows = []
    for i in range(40):
        d = f"2024-05-{i+1:02d}" if i < 31 else f"2024-06-{i-30:02d}"
        up = 1 + (0.03 if i % 2 == 0 else -0.03)     # helps group
        dn = 1 + (-0.03 if i % 2 == 0 else 0.03)     # hurts group (opposite)
        for t in ("KCB", "EQTY", "ABSA"):
            rows.append([t, d, f"{100*up:.2f}"])
        for t in ("HFCK", "KPLC", "BAMB"):
            rows.append([t, d, f"{50*dn:.2f}"])
    _write_prices(csvp, rows)
    reports = GS.validate_driver_exposures(str(csvp), min_overlap=10)
    cbk = [r for r in reports if r["driver"] == "CBK rate"]
    assert cbk and "suspect" in cbk[0] and "within_group_corr" in cbk[0]


def test_cooccurrence_ignores_singletons(tmp_path):
    pred = {"event_type": "earnings",
            "source_entities": [{"name": "OneOff Corp", "kind": "company", "direction": "UP"}],
            "directly_affected": [{"ticker": "KCB", "direction": "UP",
                                   "impact_type": "direct", "confidence": 0.9}]}
    fp = tmp_path / "extractions_1.jsonl"
    _write_extractions(fp, [pred])  # single occurrence
    edges = GS.cooccurrence_edges([str(fp)], min_count=2)
    assert edges == []


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_build_graph_layers_data_sources(tmp_path):
    csvp = tmp_path / "prices.csv"
    rows = []
    for i in range(40):
        d = f"2024-03-{i+1:02d}"
        bump = 1 + (0.03 if i % 2 == 0 else -0.03)
        rows += [["KCB", d, f"{100*bump:.2f}"], ["COOP", d, f"{40*bump:.2f}"]]
    _write_prices(csvp, rows)

    base = G.build_default_graph()
    base_edges = sum(len(v) for v in base.adj.values())
    rich = G.build_graph(prices_csv=str(csvp),
                         comovement_kwargs={"min_abs_corr": 0.5, "min_overlap": 10})
    rich_edges = sum(len(v) for v in rich.adj.values())
    assert rich_edges > base_edges
    # no-arg build_graph stays identical to the curated seed
    assert sum(len(v) for v in G.build_graph().adj.values()) == base_edges


# ---------------------------------------------------------------------------
# Export (production hand-off)
# ---------------------------------------------------------------------------

def test_export_graph_roundtrip(tmp_path):
    out = G.export_graph(tmp_path / "graph_export.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    # companies carry symbol/name/sector for the production Ticker/Sector tables
    assert data["companies"] and all(
        {"symbol", "name", "sector"} <= set(c) for c in data["companies"])
    syms = {c["symbol"] for c in data["companies"]}
    assert {"KCB", "KQ", "SCOM"} <= syms
    # nodes + edges reflect the in-memory graph
    g = G.build_default_graph()
    assert len(data["nodes"]) == len(g.kind)
    assert len(data["edges"]) == sum(len(v) for v in g.adj.values())
    kinds = {n["kind"] for n in data["nodes"]}
    assert {"company", "sector", "product", "driver"} <= kinds
    etypes = {e["etype"] for e in data["edges"]}
    assert {"in_sector", "competitor", "helps_when_up"} <= etypes
