from celery import Celery
from celery.schedules import crontab
from config.settings import settings

celery_app = Celery(
    "stocks_celery",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.scheduler.tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Scheduled Tasks
celery_app.conf.beat_schedule = {
    'fetch-news-hourly': {
        'task': 'app.scheduler.tasks.task_fetch_news',
        'schedule': crontab(minute='0'),  # hourly
    },
    'scrape-prices-daily': {
        'task': 'app.scheduler.tasks.task_scrape_prices',
        'schedule': crontab(hour='0', minute='5'),  # daily at 00:05
    },
    'analyze-sentiment-every-10m': {
        'task': 'app.scheduler.tasks.task_analyze_sentiment',
        'schedule': crontab(minute='*/10'),  # Poll redis queue
    },
    'run-predictions-hourly': {
        'task': 'app.scheduler.tasks.task_run_predictions',
        'schedule': crontab(minute='15'), # run after news fetch
    },
    'retrain-model-weekly': {
        'task': 'app.scheduler.tasks.task_retrain_model',
        'schedule': crontab(hour='2', minute='0', day_of_week='sun'), # weekly Sunday
    },
}
