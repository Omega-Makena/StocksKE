import structlog
from app.scheduler.celery_app import celery_app
from app.main import SessionLocal
from app.ingestion.news_fetcher import NewsFetcher
from app.ingestion.price_scraper import PriceScraper
from app.processing.sentiment_analyzer import SentimentAnalyzer
from app.prediction.train import ModelTrainer
from app.prediction.predict import Predictor
from app.alerting.generator import AlertGenerator

logger = structlog.get_logger(__name__)

@celery_app.task
def task_fetch_news():
    logger.info("Task trigger: fetch_news")
    with SessionLocal() as db:
        fetcher = NewsFetcher(db)
        fetcher.fetch_recent_news()

@celery_app.task
def task_scrape_prices():
    logger.info("Task trigger: scrape_prices")
    with SessionLocal() as db:
        scraper = PriceScraper(db)
        scraper.scrape_eod_data()

@celery_app.task
def task_analyze_sentiment():
    logger.info("Task trigger: analyze_sentiment")
    with SessionLocal() as db:
        analyzer = SentimentAnalyzer(db)
        # Process a batch of up to 50 queue items per run
        analyzer.process_queue(batch_size=50)

@celery_app.task
def task_run_predictions():
    logger.info("Task trigger: run_predictions")
    with SessionLocal() as db:
        predictor = Predictor(db)
        predictor.run_predictions()
        
        # Trigger alerts evaluated on the newly created predictions
        alert_gen = AlertGenerator(db)
        alert_gen.check_and_send_alerts()

@celery_app.task
def task_retrain_model():
    logger.info("Task trigger: retrain_model")
    with SessionLocal() as db:
        trainer = ModelTrainer(db)
        trainer.train_all_models()
