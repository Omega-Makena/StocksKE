import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import structlog
from app.models.base import Base
from app.main import engine

logger = structlog.get_logger(__name__)

def init_db():
    logger.info("Initializing database schemas (via SQLAlchemy, but typically use Alembic instead).")
    Base.metadata.create_all(bind=engine)
    logger.info("Schema creation complete.")

if __name__ == "__main__":
    init_db()
