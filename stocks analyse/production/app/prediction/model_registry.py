import joblib
import os
import structlog

logger = structlog.get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelRegistry:
    @staticmethod
    def save_model(model, ticker_symbol: str):
        path = os.path.join(MODEL_DIR, f"{ticker_symbol}_xgboost.joblib")
        joblib.dump(model, path)
        logger.info("Model saved", path=path, ticker=ticker_symbol)

    @staticmethod
    def load_model(ticker_symbol: str):
        path = os.path.join(MODEL_DIR, f"{ticker_symbol}_xgboost.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        return None
