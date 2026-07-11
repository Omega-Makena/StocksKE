"""
Load the knowledge graph into the production DB from the pipeline's canonical
export (single source of truth), replacing the old hardcoded mock.

Source resolution (first that works):
  1. $GRAPH_EXPORT_FILE
  2. <repo>/pipeline/graph_export.json  (monorepo default)
  3. import the pipeline's graph module and export on the fly

Populates:
  * sectors      — from the company registry
  * tickers      — the NSE-listed (tradeable) companies
  * graph_edges  — the full typed/weighted edge set (companies/products/drivers)
"""
import sys
import os
import json
from pathlib import Path

import structlog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import SessionLocal, engine
from app.models.base import Base
from app.models.ticker import Ticker, Sector
from app.models.graph_edge import GraphEdge

logger = structlog.get_logger(__name__)

# repo root = two levels up from this file (production/scripts -> repo)
_REPO = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO / "pipeline"


def _load_export() -> dict:
    """Return the graph export dict (companies / nodes / edges)."""
    candidates = []
    if os.environ.get("GRAPH_EXPORT_FILE"):
        candidates.append(Path(os.environ["GRAPH_EXPORT_FILE"]))
    candidates.append(_PIPELINE / "graph_export.json")

    for path in candidates:
        if path and path.exists():
            logger.info("Loading knowledge graph export", path=str(path))
            return json.loads(path.read_text(encoding="utf-8"))

    # Fall back to generating it directly from the pipeline (local/dev).
    logger.info("No export file found; generating from pipeline.graph", pipeline=str(_PIPELINE))
    sys.path.insert(0, str(_PIPELINE))
    import graph as G  # type: ignore
    out = G.export_graph()
    return json.loads(Path(out).read_text(encoding="utf-8"))


def load_graph():
    data = _load_export()
    # ensure tables exist (idempotent; Alembic is preferred in real deploys)
    Base.metadata.create_all(bind=engine)

    companies = data.get("companies", [])
    edges = data.get("edges", [])

    with SessionLocal() as db:
        # --- sectors ---
        sector_ids = {}
        for name in sorted({c["sector"] for c in companies if c.get("sector")}):
            sector = db.query(Sector).filter(Sector.name == name).first()
            if not sector:
                sector = Sector(name=name)
                db.add(sector)
                db.flush()
            sector_ids[name] = sector.id

        # --- tickers (NSE-listed companies) ---
        for c in companies:
            ticker = db.query(Ticker).filter(Ticker.symbol == c["symbol"]).first()
            if not ticker:
                ticker = Ticker(symbol=c["symbol"])
                db.add(ticker)
            ticker.name = c["name"]
            ticker.sector_id = sector_ids.get(c.get("sector"))

        # --- edges (full graph): refresh from source ---
        db.query(GraphEdge).delete()
        for e in edges:
            db.add(GraphEdge(src=e["src"], dst=e["dst"], etype=e["etype"],
                             weight=float(e.get("weight", 1.0))))

        db.commit()
        logger.info("Knowledge graph loaded",
                    sectors=len(sector_ids), tickers=len(companies), edges=len(edges))


if __name__ == "__main__":
    load_graph()
