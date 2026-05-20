import structlog
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prometheus_client import make_asgi_app

from config.settings import settings
from app.models.base import Base
from app.models.feedback import Feedback
from app.models.alert import Alert

logger = structlog.get_logger(__name__)

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Kenyan Stock Prediction API")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class FeedbackCreate(BaseModel):
    alert_id: int
    feedback_type: str

@app.post("/feedback")
def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    logger.info("Received feedback request", alert_id=feedback.alert_id, feedback_type=feedback.feedback_type)
    
    alert = db.query(Alert).filter(Alert.id == feedback.alert_id).first()
    if not alert:
        logger.warning("Feedback submitted for unknown alert ID", alert_id=feedback.alert_id)
        raise HTTPException(status_code=404, detail="Alert not found")
        
    new_feedback = Feedback(
        alert_id=feedback.alert_id,
        feedback_type=feedback.feedback_type
    )
    db.add(new_feedback)
    
    # Optional: Update alert record with the feedback directly for convenience
    alert.user_feedback = feedback.feedback_type

    db.commit()
    return {"status": "success", "message": "Feedback recorded."}

@app.get("/health")
def health_check():
    return {"status": "ok"}
