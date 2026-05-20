import sys
import os
import csv
import structlog
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import SessionLocal
from app.models.ticker import Ticker, Sector

logger = structlog.get_logger(__name__)

def load_graph():
    logger.info("Loading initial Sectors and Tickers knowledge graph.")
    
    # Simple hardcoded mock data for Kenya NSE to satisfy system requirements.
    # Usually you'd read from a CSV: csv.reader(open('knowledge_graph.csv'))
    
    sectors_data = ["Telecommunications", "Banking", "Manufacturing", "Energy"]
    tickers_data = [
        {"symbol": "SAFCOM", "name": "Safaricom Plc", "sector": "Telecommunications"},
        {"symbol": "KCB", "name": "KCB Group", "sector": "Banking"},
        {"symbol": "EABL", "name": "East African Breweries", "sector": "Manufacturing"},
        {"symbol": "KENGEN", "name": "KenGen", "sector": "Energy"},
    ]
    
    with SessionLocal() as db:
        sector_map = {}
        for s_name in sectors_data:
            sector = db.query(Sector).filter(Sector.name == s_name).first()
            if not sector:
                sector = Sector(name=s_name)
                db.add(sector)
                db.commit()
            sector_map[s_name] = sector.id
            
        for t_data in tickers_data:
            ticker = db.query(Ticker).filter(Ticker.symbol == t_data['symbol']).first()
            if not ticker:
                ticker = Ticker(
                    symbol=t_data['symbol'],
                    name=t_data['name'],
                    sector_id=sector_map.get(t_data['sector'])
                )
                db.add(ticker)
        db.commit()
        logger.info("Knowledge Graph loaded successfully.")

if __name__ == "__main__":
    load_graph()
