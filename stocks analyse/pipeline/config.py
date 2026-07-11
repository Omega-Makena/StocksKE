import os
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY   = os.environ.get("NEWS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "0.0"))
# Per-request timeout (seconds). Local models (Ollama) need a generous value —
# the first call loads the model into VRAM and can take 30-60s on a small GPU.
LLM_TIMEOUT    = int(os.environ.get("LLM_TIMEOUT", "120"))
# Output cap. The lean schema (graph derives indirect impact) rarely needs more
# than a few hundred tokens; 600 leaves headroom for multi-entity events.
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", "600"))
# Article text is truncated before sending to the LLM to bound input tokens.
MAX_ARTICLE_CHARS = int(os.environ.get("MAX_ARTICLE_CHARS", "1600"))
# When true, skip the LLM entirely for articles that mention no NSE entity or
# macro/commodity keyword (saves the whole call). See extractor.is_relevant.
PREFILTER_ARTICLES = os.environ.get("PREFILTER_ARTICLES", "1") not in ("0", "false", "False")

# If a robots.txt can't be read, allow the fetch (availability) vs block it
# (strict compliance). Set to 0 in production if your policy requires it.
ROBOTS_FAIL_OPEN = os.environ.get("ROBOTS_FAIL_OPEN", "1") not in ("0", "false", "False")
# Persist seen article URLs across runs so the pipeline doesn't re-process (and
# re-pay for) the same news every run. Set empty to disable.
DEDUP_STORE = os.environ.get("DEDUP_STORE", "seen_urls.json")

OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", "nse_dataset")
PRICE_CSV_PATH = os.environ.get("PRICE_CSV_PATH", "")

# Path to the compiled-securities folder produced by importer.py.
# When set, pipeline.py will auto-build a prices CSV from it before aligning.
COMPILED_DIR   = os.environ.get("COMPILED_DIR", "")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
PRICE_CHANGE_THRESHOLD = float(os.environ.get("PRICE_CHANGE_THRESHOLD", "1.5"))
LOOKAHEAD_DAYS = int(os.environ.get("LOOKAHEAD_DAYS", "3"))

# Placeholder key values that mean "not really configured".
_PLACEHOLDER_KEYS = {"", "sk-...", "your_openai_api_key_here", "changeme"}


def validate_config() -> list[str]:
    """Return a list of configuration problems (empty = OK). Call at startup so a
    misconfiguration fails loudly instead of silently producing no predictions."""
    problems = []
    if OPENAI_API_KEY in _PLACEHOLDER_KEYS:
        problems.append("OPENAI_API_KEY is empty or a placeholder — the LLM step will produce nothing.")
    if not OPENAI_BASE_URL.startswith(("http://", "https://")):
        problems.append(f"OPENAI_BASE_URL is not a URL: {OPENAI_BASE_URL!r}")
    if "localhost" in OPENAI_BASE_URL and OPENAI_API_KEY not in ("ollama",) and OPENAI_API_KEY in _PLACEHOLDER_KEYS:
        problems.append("Local endpoint configured but no model/key — is Ollama running?")
    if MAX_TOKENS < 100:
        problems.append(f"MAX_TOKENS={MAX_TOKENS} is very low; extractions may be truncated.")
    if not (0.0 <= CONFIDENCE_THRESHOLD <= 1.0):
        problems.append(f"CONFIDENCE_THRESHOLD out of [0,1]: {CONFIDENCE_THRESHOLD}")
    return problems
