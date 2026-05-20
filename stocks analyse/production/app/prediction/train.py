import structlog
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split

from app.models.ticker import Ticker
from app.prediction.features import build_features_for_ticker
from app.prediction.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)

FEATURE_COLS = ['ret_1d', 'ret_5d', 'ret_20d', 'vol_change', 'sma_20', 'rsi_14', 'score', 'score_trend_3d']

class ModelTrainer:
    def __init__(self, db: Session):
        self.db = db

    def train_all_models(self):
        logger.info("Starting weekly model retraining")
        tickers = self.db.query(Ticker).all()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365) # 1 year lookback

        trained_count = 0
        for ticker in tickers:
            df = build_features_for_ticker(self.db, ticker.id, start_date, end_date)
            
            if len(df) < 50:
                logger.warning("Not enough data to train model", ticker=ticker.symbol)
                continue
                
            X = df[FEATURE_COLS]
            y = df['target']
            
            # Simple train logic (80/20 time split implied by shuffle=False but let's just use regular for purely structural pipeline setup)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)
            
            # Basic valuation metric
            acc = model.score(X_test, y_test)
            logger.info("Model trained", ticker=ticker.symbol, accuracy=acc)
            
            ModelRegistry.save_model(model, ticker.symbol)
            trained_count += 1
            
        logger.info("Weekly model retraining complete", total_trained=trained_count)
