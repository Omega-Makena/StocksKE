from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/stocks_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    NEWS_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    PREDICTION_THRESHOLD: float = 0.7
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
