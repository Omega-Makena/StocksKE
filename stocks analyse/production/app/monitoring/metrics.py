from prometheus_client import Counter, Histogram

# Example custom metrics. In full scale these would be incremented logically
# in places like app/ingestion/news_fetcher.py 

ARTICLES_PROCESSED = Counter('articles_processed_total', 'Total number of news articles processed')
PREDICTIONS_GENERATED = Counter('predictions_generated_total', 'Total number of predictions generated')
ALERTS_SENT = Counter('alerts_sent_total', 'Total number of Telegram alerts sent')
FEEDBACK_RECEIVED = Counter('feedback_received_total', 'Total number of user feedback inputs')

MODEL_TRAIN_TIME = Histogram('model_train_time_seconds', 'Time spent training models')
