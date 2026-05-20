import structlog
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.models.ticker import Ticker
from app.models.prediction import Prediction
from app.prediction.features import build_features_for_ticker
from app.prediction.model_registry import ModelRegistry
from app.prediction.train import FEATURE_COLS

logger = structlog.get_logger(__name__)

class Predictor:
    def __init__(self, db: Session):
        self.db = db

    def run_predictions(self):
        logger.info("Starting predictions generation")
        tickers = self.db.query(Ticker).all()
        
        # We just need the most recent row, so fetch the last 60 days to build features 
        # and slice the final day out.
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=60)
        
        for ticker in tickers:
            model = ModelRegistry.load_model(ticker.symbol)
            if not model:
                continue
                
            df = build_features_for_ticker(self.db, ticker.id, start_date, end_date)
            if df.empty:
                continue
                
            latest_features = df.iloc[-1:]
            X_latest = latest_features[FEATURE_COLS]
            
            prob = float(model.predict_proba(X_latest)[0][1]) # Probability of class 1 (Up)
            direction = 1 if prob > 0.5 else 0
            
            features_snapshot = X_latest.to_dict(orient='records')[0]
            
            prediction_record = Prediction(
                ticker_id=ticker.id,
                predicted_direction=direction,
                probability=prob,
                features_snapshot=features_snapshot
            )
            
            self.db.add(prediction_record)
            
        self.db.commit()
        logger.info("Predictions generation complete.")
