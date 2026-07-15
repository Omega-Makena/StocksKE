"""Tests for the knowledge-graph impact-propagation engine."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import graph as G


def test_graph_builds_with_expected_node_kinds():
    g = G.build_default_graph()
    # NSE companies are tradeable company nodes
    assert g.kind.get("KQ") == "company"
    assert "KQ" in g.tradeable
    # sector hubs exist and are not tradeable
    assert g.kind.get("sector:Banking") == "sector"
    assert "sector:Banking" not in g.tradeable
    # product + non-NSE maker nodes exist, and Boeing is NOT tradeable
    assert g.kind.get("product:Boeing 737 MAX") == "product"
    assert g.kind.get("Boeing") == "company"
    assert "Boeing" not in g.tradeable


def test_ethiopian_crash_propagates_to_kenya_airways():
    """The motivating spec example: an Ethiopian Airlines Boeing crash should
    drag Kenya Airways (KQ) DOWN via the shared aircraft product node, without
    KQ being named in the source event."""
    g = G.build_default_graph()
    sources = {"Ethiopian Airlines": G.shock_from_prediction("DOWN", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="disaster")

    assert "KQ" in impacts, "Kenya Airways should be reached via shared Boeing fleet"
    kq = impacts["KQ"]
    # downward pressure (negative shock); the conservative direction threshold may
    # report NEUTRAL for a weak multi-hop signal — the sign is the tested property
    assert kq.shock < 0
    assert kq.magnitude_pct < 0
    # reached through the product node, not a direct competitor hop
    assert any(n.startswith("product:") for n in kq.path)
    # Boeing (maker) is reached too, but is non-tradeable so not in impacts
    assert "Boeing" not in impacts


def test_shared_fleet_contagion_is_same_direction():
    """A safety event is same-direction contagion for co-operators (both DOWN),
    not a competitor benefit."""
    g = G.build_default_graph()
    sources = {"KQ": G.shock_from_prediction("DOWN", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="disaster")
    # Ethiopian is non-tradeable, but any tradeable co-operator would be DOWN.
    for imp in impacts.values():
        if any(n.startswith("product:") for n in imp.path):
            assert imp.direction in ("DOWN", "NEUTRAL")


def test_earnings_competitor_channel_flips_sign():
    """A rival's earnings *beat* (UP) is mildly negative for competitors."""
    g = G.build_default_graph()
    sources = {"KCB": G.shock_from_prediction("UP", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="earnings")
    # EQTY is a listed competitor of KCB; the competitor channel flips the sign
    # (rival's UP -> negative shock for you), tested independently of the
    # conservative direction threshold
    assert "EQTY" in impacts
    assert impacts["EQTY"].shock < 0


def test_regulation_moves_whole_sector_same_direction():
    g = G.build_default_graph()
    sources = {"KCB": G.shock_from_prediction("DOWN", severity=0.8)}
    impacts = G.propagate(g, sources, event_type="regulation")
    # other banks should be dragged DOWN via the sector hub
    banks = [t for t in ("EQTY", "COOP", "ABSA", "NCBA") if t in impacts]
    assert banks, "regulation should spill across the Banking sector"
    # same-direction spillover: negative shock like the source (sign, not the
    # thresholded direction, is the property under test)
    assert all(impacts[t].shock < 0 for t in banks)


def test_magnitude_and_confidence_decay_with_distance():
    """Impact should weaken (smaller |magnitude|, lower confidence) with hops."""
    g = G.build_default_graph()
    sources = {"KCB": G.shock_from_prediction("DOWN", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="regulation")
    if len(impacts) >= 2:
        ordered = sorted(impacts.values(), key=lambda i: i.hops)
        near, far = ordered[0], ordered[-1]
        assert abs(near.magnitude_pct) >= abs(far.magnitude_pct)
        assert near.confidence >= far.confidence


def test_confidence_and_magnitude_bounds():
    g = G.build_default_graph()
    sources = {"KCB": G.shock_from_prediction("DOWN", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="disaster", base_confidence=0.9)
    for imp in impacts.values():
        assert 0.0 <= imp.confidence <= 1.0
        assert -100.0 < imp.magnitude_pct < 100.0
        assert imp.hops >= 1


def test_neutral_source_produces_no_impact():
    g = G.build_default_graph()
    sources = {"KCB": G.shock_from_prediction("NEUTRAL", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="earnings")
    assert impacts == {}


def test_propagation_terminates_on_dense_graph():
    """Sanity: spreading from every node must halt (no infinite loops)."""
    g = G.build_default_graph()
    sources = {c["ticker"]: 1.0 for c in G.NSE_COMPANIES}
    impacts = G.propagate(g, sources, event_type="regulation")
    assert isinstance(impacts, dict)  # returns without hanging


def test_enrich_prediction_rebuilds_indirectly_affected():
    pred = {
        "event_type": "disaster",
        "severity": 1.0,
        "directly_affected": [
            {"ticker": "KQ", "direction": "DOWN", "confidence": 0.9}
        ],
        "indirectly_affected": [{"ticker": "BOGUS", "direction": "UP"}],
    }
    out = G.enrich_prediction(pred)
    tickers = {e["ticker"] for e in out["indirectly_affected"]}
    assert "BOGUS" not in tickers  # LLM guess was replaced
    # every enriched entry carries a magnitude and a confidence
    for e in out["indirectly_affected"]:
        assert "magnitude_pct" in e and "confidence" in e
    assert out["propagation"]["sources"] == ["KQ"]


def test_enrich_seeds_from_foreign_source_entities():
    """A prediction with no NSE ticker in directly_affected, but foreign
    source_entities (Ethiopian Airlines + Boeing 737 MAX), must still reach KQ
    through the shared product node."""
    pred = {
        "event_type": "disaster",
        "severity": 1.0,
        "directly_affected": [],
        "source_entities": [
            {"name": "Ethiopian Airlines", "kind": "company", "direction": "DOWN", "severity": 1.0},
            {"name": "Boeing 737 MAX", "kind": "product", "direction": "DOWN", "severity": 1.0},
        ],
    }
    out = G.enrich_prediction(pred)
    tickers = {e["ticker"]: e for e in out["indirectly_affected"]}
    assert "KQ" in tickers
    assert tickers["KQ"]["direction"] == "DOWN"
    assert "Boeing 737 MAX" in out["propagation"]["sources"] or \
           "product:Boeing 737 MAX" in out["propagation"]["sources"]


def test_enrich_ignores_unknown_source_entities():
    pred = {
        "event_type": "disaster",
        "severity": 1.0,
        "directly_affected": [],
        "source_entities": [
            {"name": "Totally Fictional Corp", "kind": "company", "direction": "DOWN", "severity": 1.0},
            {"name": "Imaginary Widget 9000", "kind": "product", "direction": "DOWN", "severity": 1.0},
        ],
    }
    out = G.enrich_prediction(pred)
    assert out["indirectly_affected"] == []
    assert out["propagation"]["reached"] == 0


def test_cbk_rate_hike_lifts_banks_hurts_rate_sensitive():
    """Macro driver: a rate hike (CBK rate UP) should push bank margins UP and
    rate-sensitive / leveraged names DOWN — opposite signs from one event."""
    g = G.build_default_graph()
    sources = {"driver:CBK rate": G.shock_from_prediction("UP", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="macro")
    # banks helped
    for bank in ("KCB", "EQTY", "ABSA"):
        assert impacts[bank].direction == "UP", bank
    # rate-sensitive hurt
    for name in ("HFCK", "KPLC", "BAMB"):
        assert impacts[name].direction == "DOWN", name


def test_oil_price_spike_splits_marketer_from_consumers():
    g = G.build_default_graph()
    sources = {"driver:Oil price": G.shock_from_prediction("UP", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="commodity")
    assert impacts["TOTL"].direction == "UP"       # fuel marketer benefits
    assert impacts["KQ"].direction == "DOWN"        # jet-fuel cost up
    assert impacts["BAMB"].direction == "DOWN"      # kiln energy cost up


def test_weak_shilling_helps_exporters_hurts_importers():
    g = G.build_default_graph()
    # KES/USD UP == shilling weakening
    sources = {"driver:KES/USD": G.shock_from_prediction("UP", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="macro")
    assert impacts["WTK"].direction == "UP"         # tea exporter earns more KES
    assert impacts["KQ"].direction == "DOWN"        # USD costs/leases


def test_driver_channels_inactive_for_non_macro_events():
    """Seeding a driver under an ordinary event must not fan out — driver edges
    are only enabled for macro / commodity channels."""
    g = G.build_default_graph()
    sources = {"driver:CBK rate": 1.0}
    assert G.propagate(g, sources, event_type="earnings") == {}
    assert G.propagate(g, sources, event_type="disaster") == {}


def test_exposure_map_preserved_when_direction_is_conservative():
    """A modest-severity event yields few/no confident directional calls (the
    honest, non-over-calling default) but the EXPOSURE map still lists every
    connected NSE name — that is the reliable output."""
    pred = {
        "event_type": "macro", "severity": 0.4,
        "source_entities": [{"name": "CBK rate", "kind": "driver", "direction": "UP", "severity": 0.4}],
        "directly_affected": [],
    }
    out = G.enrich_prediction(pred)
    assert out["exposed"], "exposure map should list the connected NSE names"
    # confident directional calls are a subset of the exposure map
    assert len(out["indirectly_affected"]) <= len(out["exposed"])
    assert out["propagation"]["exposed"] == len(out["exposed"])
    assert all("reasoning" in e for e in out["exposed"])


def test_enrich_from_macro_driver_entity():
    pred = {
        "event_type": "macro",
        "severity": 0.8,
        "directly_affected": [],
        "source_entities": [
            {"name": "CBK rate", "kind": "driver", "direction": "UP", "severity": 0.8}
        ],
    }
    out = G.enrich_prediction(pred)
    by = {e["ticker"]: e for e in out["indirectly_affected"]}
    assert by["KCB"]["direction"] == "UP"
    assert by["KCB"]["impact_type"] == "regulatory"
    assert "driver:CBK rate" in out["propagation"]["sources"]


def test_supplier_chain_direction_fixed():
    """KenGen supplies power to Kenya Power (KEGN -> KPLC), not the reverse."""
    g = G.build_default_graph()
    # a disaster at the supplier should reach the customer via the supply edge
    sources = {"KEGN": G.shock_from_prediction("DOWN", severity=1.0)}
    impacts = G.propagate(g, sources, event_type="disaster")
    assert "KPLC" in impacts


def test_source_never_appears_in_its_own_impacts():
    g = G.build_default_graph()
    sources = {"KCB": 1.0}
    impacts = G.propagate(g, sources, event_type="regulation")
    assert "KCB" not in impacts
