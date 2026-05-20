import os
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", "1500"))

OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", "nse_dataset")
PRICE_CSV_PATH = os.environ.get("PRICE_CSV_PATH", "")

# Path to the compiled-securities folder produced by importer.py.
# When set, pipeline.py will auto-build a prices CSV from it before aligning.
COMPILED_DIR   = os.environ.get("COMPILED_DIR", "")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
PRICE_CHANGE_THRESHOLD = float(os.environ.get("PRICE_CHANGE_THRESHOLD", "1.5"))
LOOKAHEAD_DAYS = int(os.environ.get("LOOKAHEAD_DAYS", "3"))
