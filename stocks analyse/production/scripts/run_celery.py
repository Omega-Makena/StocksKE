import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.scheduler.celery_app import celery_app

if __name__ == "__main__":
    celery_app.start()
