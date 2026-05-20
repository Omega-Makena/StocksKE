import hashlib
import requests
import structlog
from datetime import datetime
from sqlalchemy.orm import Session
import json
import redis
import dateutil.parser

from config.settings import settings
from app.models.news import News

logger = structlog.get_logger(__name__)

# Basic in-memory redis client for queueing inside the worker process context
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

class NewsFetcher:
    def __init__(self, db: Session):
        self.db = db
        self.api_key = settings.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_recent_news(self):
        logger.info("Starting news fetch")
        if not self.api_key:
            logger.warning("No NEWS_API_KEY set. Skipping execution.")
            return

        params = {
            "q": "NSE OR (Kenya stock) OR (Nairobi Securities Exchange)",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
            "pageSize": 50
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles_processed = 0
            for article in data.get("articles", []):
                if self.process_article(article):
                    articles_processed += 1
            
            self.db.commit()
            logger.info("News fetch completed", count=articles_processed)
            
        except Exception as e:
            logger.error("Failed to fetch news", error=str(e))
            self.db.rollback()

    def process_article(self, article: dict) -> bool:
        title = article.get("title")
        content = article.get("content") or article.get("description", "")
        url = article.get("url")
        published_str = article.get("publishedAt")
        
        if not title or not url:
            return False
            
        # Deduplication by content hash
        # To avoid minor URL parameter differences breaking hashing
        content_hash = hashlib.md5(f"{title}{content}".encode('utf-8')).hexdigest()
        
        existing = self.db.query(News).filter(News.content_hash == content_hash).first()
        if existing:
            return False

        try:
            published_at = dateutil.parser.isoparse(published_str).replace(tzinfo=None)
        except Exception:
            published_at = datetime.utcnow()

        news_record = News(
            title=title,
            content=content,
            url=url,
            content_hash=content_hash,
            published_at=published_at
        )
        self.db.add(news_record)
        self.db.flush() # get the ID without committing
        
        # Push to redis queue for Sentiment Analyzer
        message = {
            "news_id": news_record.id,
            "title": title,
            "content": content
        }
        redis_client.lpush("news_queue", json.dumps(message))
        return True
