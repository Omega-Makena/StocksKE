import openai
import json
import structlog
from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

class LLMClient:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.mock_fallback = not bool(settings.OPENAI_API_KEY)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze_sentiment(self, title: str, content: str) -> dict:
        if self.mock_fallback:
            return self._mock_analyze(title, content)
            
        prompt = (
            "You are a Kenyan financial analyst. Given a news headline and summary, "
            "output a JSON object with keys: ticker (the most relevant NSE ticker symbol), "
            "sentiment_score (float between -1 and 1), sector (if identifiable)."
        )
        
        try:
            # Using ChatCompletion which is standard for newer usage
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Headline: {title}\nSummary: {content}"}
                ],
                temperature=0.0
            )
            
            # Attempt to parse json out of the GPT response
            reply_text = response.choices[0].message.content.strip()
            
            # Simple extraction in case it surrounds it with markdown codeblocks
            if "```json" in reply_text:
                reply_text = reply_text.split("```json")[1].split("```")[0].strip()
                
            result = json.loads(reply_text)
            return result
        except Exception as e:
            logger.error("LLM API call failed", error=str(e))
            raise e

    def _mock_analyze(self, title: str, content: str) -> dict:
        # Fallback logic block for testing if no API key is provided
        from random import random
        return {
            "ticker": "SAFCOM",
            "sentiment_score": (random() * 2) - 1,
            "sector": "Telecommunications"
        }
