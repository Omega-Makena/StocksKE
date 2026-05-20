import structlog
import random
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.price import Price
from app.models.ticker import Ticker
from app.ingestion.validator import validate_price_data

logger = structlog.get_logger(__name__)

class PriceScraper:
    def __init__(self, db: Session):
        self.db = db
        self.mock = True # Setup to mock by default for demonstration
        
    def scrape_eod_data(self):
        logger.info("Starting scheduled EOD price scrape")
        tickers = self.db.query(Ticker).all()
        
        if not tickers:
            logger.warning("No tickers in DB to scrape prices for.")
            return
            
        scraped_records = 0
        now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for ticker in tickers:
            # Here you would typically perform actual scraping via requests + BS4
            # e.g.:
            # url = f"https://my-data-source.com/quote/{ticker.symbol}"
            # page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0...'})
            # soup = BeautifulSoup(page.text, 'html.parser')
            # raw_price = soup.select_one('.price-class').text
            
            # Since we need a robust mock to ensure it runs:
            data_point = self.mock_fetch_price(ticker)
            
            if validate_price_data(data_point):
                record = Price(
                    time=now,
                    ticker_id=ticker.id,
                    open=data_point['open'],
                    high=data_point['high'],
                    low=data_point['low'],
                    close=data_point['close'],
                    volume=data_point['volume']
                )
                
                # Use merge to handle potential duplicate dates safely
                self.db.merge(record)
                scraped_records += 1
                
        self.db.commit()
        logger.info("EOD price scrape completed", count=scraped_records)
        
    def mock_fetch_price(self, ticker: Ticker):
        base_price = random.uniform(10.0, 150.0)
        return {
            "open": base_price,
            "high": base_price * 1.05,
            "low": base_price * 0.95,
            "close": base_price * random.uniform(0.98, 1.03),
            "volume": random.randint(1000, 500000)
        }
