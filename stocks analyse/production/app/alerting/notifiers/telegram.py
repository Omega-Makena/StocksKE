import requests
import structlog
from config.settings import settings
from .base import BaseNotifier

logger = structlog.get_logger(__name__)

class TelegramNotifier(BaseNotifier):
    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    def send(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat id not configured. Skipping alert.")
            return False
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send Telegram message", error=str(e))
            return False
