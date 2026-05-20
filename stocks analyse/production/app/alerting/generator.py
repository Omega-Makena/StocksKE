import structlog
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.models.prediction import Prediction
from app.models.alert import Alert
from app.models.ticker import Ticker
from config.settings import settings
from app.alerting.notifiers.telegram import TelegramNotifier

logger = structlog.get_logger(__name__)

class AlertGenerator:
    def __init__(self, db: Session):
        self.db = db
        self.notifier = TelegramNotifier()
        self.threshold = settings.PREDICTION_THRESHOLD
        
    def check_and_send_alerts(self):
        logger.info("Checking for alerts based on new predictions")
        
        # Get latest predictions. Assuming this runs right after prediction step.
        latest_predictions = self.db.query(Prediction).order_by(Prediction.id.desc()).limit(100).all()
        
        for pred in latest_predictions:
            # Simple check if probability passes confident configurable threshold
            if pred.probability >= self.threshold:
                ticker = self.db.query(Ticker).filter(Ticker.id == pred.ticker_id).first()
                if not ticker:
                    continue
                    
                direction_str = "BUY/UP" if pred.predicted_direction == 1 else "SELL/DOWN"
                
                message = (
                    f"🚨 AI Stock Alert: {ticker.symbol} 🚨\n\n"
                    f"Model Confidence: {pred.probability*100:.1f}%\n"
                    f"Predicted Direction: {direction_str}\n\n"
                    f"Provide feedback via API: /feedback endpoint using Alert ID."
                )
                
                # Check recent alerts to avoid spamming the same stock 
                # (e.g. within last 24h)
                # Not fully expanded here for length, but the concept is to filter existing alerts.
                
                alert_record = Alert(
                    ticker_id=ticker.id,
                    message=message
                )
                
                self.db.add(alert_record)
                self.db.commit() # Commit to get ID
                
                # Append alert ID to message for feedback hook
                message += f"\nAlert Ref: {alert_record.id}"
                alert_record.message = message
                self.db.commit()
                
                self.notifier.send(message)
                logger.info("Sent alert", ticker=ticker.symbol)
