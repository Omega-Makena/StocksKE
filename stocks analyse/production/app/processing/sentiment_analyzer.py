import json
import structlog
import redis
from sqlalchemy.orm import Session
from app.processing.llm_client import LLMClient
from app.models.sentiment import SentimentScore
from app.models.ticker import Ticker
from config.settings import settings

logger = structlog.get_logger(__name__)
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

class SentimentAnalyzer:
    def __init__(self, db: Session):
        self.db = db
        self.llm = LLMClient()
        
    def process_queue(self, batch_size=50):
        """
        Pulls articles out of the news_queue and passes them to the LLM. 
        Stores the result. Designed to be called by a worker repeatedly.
        """
        logger.info("Starting sentiment analyzer queue processing")
        processed = 0
        
        for _ in range(batch_size):
            item_raw = redis_client.rpop("news_queue")
            if not item_raw:
                break
                
            try:
                item = json.loads(item_raw)
                news_id = item["news_id"]
                title = item["title"]
                content = item["content"]
                
                result = self.llm.analyze_sentiment(title, content)
                
                # Attempt to map the ticker symbol from the LLM to database
                llm_ticker = result.get("ticker", "")
                ticker_record = None
                if llm_ticker:
                    ticker_record = self.db.query(Ticker).filter(Ticker.symbol.ilike(f"%{llm_ticker}%")).first()
                
                score = SentimentScore(
                    news_id=news_id,
                    ticker_id=ticker_record.id if ticker_record else None,
                    sentiment_score=float(result.get("sentiment_score", 0.0)),
                    raw_json=result
                )
                
                self.db.add(score)
                self.db.commit()
                processed += 1
                
            except Exception as e:
                logger.error("Failed to process sentiment piece", error=str(e))
                self.db.rollback()
                # Optionally push back to queue or a dead-letter queue
                # redis_client.lpush("news_queue", item_raw)

        logger.info("Sentiment queue processing ended.", processed=processed)
