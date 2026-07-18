import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path

import requests

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    OUTPUT_DIR, MAX_ARTICLE_CHARS, PREFILTER_ARTICLES, LLM_TIMEOUT,
)
from companies import NSE_COMPANIES, VALID_TICKERS

logger = logging.getLogger(__name__)
USER_AGENT = "NSE-Research-Bot/1.0"

# Company list is generated from the single source of truth (companies.py) so
# the prompt can never drift from the graph.
_COMPANY_LIST = "|".join(f"{c['ticker']}-{c['name']}" for c in NSE_COMPANIES)

# The prompt is deliberately compact: relationship/competitor reasoning was
# removed because the knowledge graph now derives all indirect impact, and the
# output schema is trimmed to only the fields consumed downstream. This is the
# largest per-call token cost, so keep it lean and STATIC (a static prefix also
# lets providers with prompt caching reuse it across calls).
SYSTEM_PROMPT = f"""You are a financial analyst for the Nairobi Securities Exchange (NSE), Kenya.
Read the article and return ONLY a JSON object (no prose, no markdown fences).

Return exactly this shape:
{{"event_type": one of [earnings, regulation, product_launch, disaster, merger_acquisition, macro, commodity, legal, management_change, other],
 "severity": 0.0-1.0 (0.1 minor, 0.5 notable, 1.0 catastrophic/market-moving),
 "source_entities": [{{"name": raw entity as reported, "kind": "company"|"product"|"driver", "direction": "UP"|"DOWN"|"NEUTRAL", "severity": 0.0-1.0}}],
 "directly_affected": [{{"ticker": NSE ticker from COMPANY LIST, "direction": "UP"|"DOWN"|"NEUTRAL", "impact_type": "direct", "confidence": 0.0-1.0}}],
 "indirectly_affected": [],
 "article_date": "YYYY-MM-DD"}}

RULES:
- directly_affected tickers MUST be in the COMPANY LIST; never invent tickers. Drop entities with confidence < 0.3.
- source_entities are the RAW entities the event is about — INCLUDE foreign/unlisted companies (e.g. Boeing, Ethiopian Airlines), products (e.g. Boeing 737 MAX), and macro/commodity drivers. Do NOT restrict these to the COMPANY LIST.
- A knowledge graph derives indirect impact from source_entities, so leave "indirectly_affected" as []. Do NOT list competitors, peers, or sector spillovers yourself.
- A driver "name" MUST be one of: "CBK rate" (UP=rates rising), "KES/USD" (UP=shilling weakening vs USD), "Oil price", "Tea price", "Construction demand"; its "direction" is the driver's own move. Use event_type "macro" for rates/FX/GDP/fiscal and "commodity" for oil/tea/inputs.

COMPANY LIST (TICKER-Name):
{_COMPANY_LIST}

EXAMPLES (input -> output):
"KCB Group posts 20% profit jump" -> {{"event_type":"earnings","severity":0.6,"source_entities":[{{"name":"KCB Group","kind":"company","direction":"UP","severity":0.6}}],"directly_affected":[{{"ticker":"KCB","direction":"UP","impact_type":"direct","confidence":0.9}}],"indirectly_affected":[],"article_date":"2024-03-01"}}
"Ethiopian Airlines Boeing 737 MAX crashes, all aboard killed" -> {{"event_type":"disaster","severity":1.0,"source_entities":[{{"name":"Ethiopian Airlines","kind":"company","direction":"DOWN","severity":1.0}},{{"name":"Boeing 737 MAX","kind":"product","direction":"DOWN","severity":1.0}}],"directly_affected":[],"indirectly_affected":[],"article_date":"2024-05-10"}}
"CBK raises benchmark rate 150bps to tame inflation" -> {{"event_type":"macro","severity":0.7,"source_entities":[{{"name":"CBK rate","kind":"driver","direction":"UP","severity":0.7}}],"directly_affected":[],"indirectly_affected":[],"article_date":"2024-06-05"}}"""


# ---------------------------------------------------------------------------
# Relevance pre-filter — skip the LLM entirely for articles that mention no NSE
# entity and no macro/commodity trigger. This is the biggest total-token lever:
# most of a general news feed is irrelevant, and a skipped article costs zero
# LLM tokens instead of the full ~2k-token call.
# ---------------------------------------------------------------------------

# Generic words that don't distinguish a company (so "Kenya Bank Group" alone
# isn't a match, but "Safaricom" or "Bamburi" is).
_STOPWORDS = {
    "bank", "group", "holdings", "kenya", "ltd", "plc", "company", "co",
    "east", "african", "africa", "insurance", "investments", "investment",
    "marketing", "securities", "exchange", "limited", "the", "and", "of",
    "reinsurance", "corp", "capital", "power", "lighting", "media", "cement",
    "paints", "breweries", "tea", "group's",
}

# Macro / commodity trigger phrases that make an article relevant even with no
# company named.
_MACRO_KEYWORDS = {
    "cbk", "central bank", "interest rate", "benchmark rate", "lending rate",
    "monetary policy", "inflation", "shilling", "forex", "exchange rate",
    "dollar", "kes/usd", "devalu", "gdp", "treasury", "fuel", "petrol",
    "diesel", "oil price", "crude", "epra", "pump price", "tea price",
    "coffee", "cement", "construction", "budget", "tax", "levy",
}


# Company-name tokens that are also common English / place / political words, so
# on their own they trigger false positives (e.g. "Limuru" the town vs LIMT the
# tea stock). Excluded from single-word matching — these companies still match
# via their ticker or a more distinctive token. Tune as false positives surface.
_AMBIGUOUS_TOKENS = {
    "limuru", "liberty", "crown", "flame", "tree", "home", "general",
    "nation", "standard", "jubilee", "olympia", "centum",
    # from "Nairobi Securities Exchange Plc" — 'nairobi' alone matches nearly
    # every Kenyan article; 'securities'/'exchange' are generic finance words.
    "nairobi", "securities", "exchange",
}


def _build_relevance_terms() -> tuple[set[str], set[str]]:
    """(single_word_terms, phrase_terms) that mark an article as NSE-relevant."""
    words: set[str] = {t.lower() for t in VALID_TICKERS}
    phrases: set[str] = set(_MACRO_KEYWORDS)
    for c in NSE_COMPANIES:
        for tok in re.split(r"[^a-z0-9]+", c["name"].lower()):
            if len(tok) >= 4 and tok not in _STOPWORDS and tok not in _AMBIGUOUS_TOKENS:
                words.add(tok)
    # a few distinctive non-NSE anchors worth catching
    phrases.update({"boeing", "airbus", "ethiopian airlines", "rwandair",
                    "safaricom", "airtel"})
    return words, phrases


_REL_WORDS, _REL_PHRASES = _build_relevance_terms()
_REL_WORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _REL_WORDS), key=len, reverse=True)) + r")\b")

# Kenyan-context terms. Required in addition to an NSE/macro signal, because
# global news collides badly with short NSE tickers (e.g. "TCL" the electronics
# brand vs TCL = TransCentury; Indian "HCLTech" earnings). Genuine Kenyan
# business news reliably carries one of these; foreign news does not.
_KENYA_CONTEXT = {
    "kenya", "kenyan", "nairobi", "shilling", "cbk", "mombasa", "kisumu",
    "east africa", "capital markets authority", "epra", "central bank of kenya",
    "safaricom", "nse-listed", "nairobi securities", "treasury bond", "kra",
}
_KENYA_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _KENYA_CONTEXT), key=len, reverse=True)) + r")\b")


# Kenyan news sources are Kenyan by construction — trust their articles on the
# NSE/macro signal alone. Everything else (NewsAPI's global results, which are
# ~99% foreign noise for NSE-ticker queries) must additionally prove Kenyan
# context, which is how we drop foreign entities that collide with NSE tickers.
_TRUSTED_SOURCES = {"standard", "nation", "capital fm", "kbc", "kenyans",
                    "business daily", "the star", "citizen", "people daily"}


def is_relevant(article: dict) -> bool:
    """True if the article plausibly concerns an NSE name or macro/commodity
    driver. Kenyan sources are trusted on the signal alone; other sources must
    also carry a Kenyan-context term (stops foreign news whose entities collide
    with NSE tickers — a major noise source — from being processed)."""
    text = " ".join(str(article.get(k) or "") for k in ("title", "description", "content")).lower()
    if not text.strip():
        return False
    signal = bool(_REL_WORD_RE.search(text)) or any(p in text for p in _REL_PHRASES)
    if not signal:
        return False
    src = str(article.get("source") or "").lower()
    if any(s in src for s in _TRUSTED_SOURCES):
        return True
    return bool(_KENYA_RE.search(text))


def build_user_message(article: dict) -> str:
    """
    Returns a JSON string with keys: article_text, article_date, source.
    article_text = article["content"] or article["description"] or article["title"]
    Truncate article_text to MAX_ARTICLE_CHARS to bound input tokens.
    """
    text = (article.get("content") or article.get("description") or article.get("title") or "")
    text = text[:MAX_ARTICLE_CHARS]
    payload = {
        "article_text": text,
        "article_date": article.get("published_at", article.get("publishedAt", "")),
        "source": article.get("source", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _strip_fences(s: str) -> str:
    # remove triple backtick fences if present
    s = re.sub(r"^```(?:\w+)?\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()


def call_llm(article: dict, retries: int = 3) -> dict | None:
    """
    POSTs to OPENAI_BASE_URL/chat/completions.
    Uses SYSTEM_PROMPT as system message.
    TEMPERATURE=0.0, MAX_TOKENS from config.
    Strips accidental markdown fences from response.
    Parses JSON. Returns None if all retries fail.
    Exponential backoff: sleep 2^attempt seconds between retries.
    On 429: sleep 60s before retry.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not configured; skipping LLM call")
        return None

    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(article)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    attempt = 0
    while attempt < retries:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code == 429:
                logger.warning("LLM rate limited (429). Sleeping 60s before retry")
                time.sleep(60)
                attempt += 1
                continue
            if resp.status_code >= 400:
                logger.warning("LLM returned %s: %s", resp.status_code, resp.text[:200])
                time.sleep(2 ** attempt)
                attempt += 1
                continue

            data = resp.json()
            content = ""
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    # support both chat and older schemas
                    msg = choices[0].get("message") or choices[0]
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            content = _strip_fences(content)
            # attempt to extract JSON substring
            try:
                return json.loads(content)
            except Exception:
                # try grab first { ... }
                m = re.search(r"(\{.*\})", content, flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        logger.exception("Failed to parse extracted JSON from LLM response")
                        return None
                logger.exception("LLM response not JSON parseable")
                return None
        except requests.RequestException as e:
            logger.exception("Network error calling LLM: %s", e)
            time.sleep(2 ** attempt)
            attempt += 1
    logger.error("All LLM call attempts failed")
    return None


def extract_all(articles: list[dict]) -> list[dict]:
    """
    Calls call_llm for each article.
    Skips articles that are too short (< 50 chars) or, when PREFILTER_ARTICLES is
    on, that mention no NSE entity / macro trigger (is_relevant) — the latter
    avoids the LLM call entirely and is the main token saver.
    Saves to OUTPUT_DIR/extractions/extractions_{timestamp}.jsonl
    Returns list of {article, prediction} dicts.
    Sleeps 0.3s between calls.
    """
    out_dir = Path(OUTPUT_DIR) / "extractions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_file = out_dir / f"extractions_{ts}.jsonl"

    results = []
    skipped_short = 0
    skipped_irrelevant = 0
    called = 0
    for a in articles:
        combined_len = len((a.get("content") or "") + (a.get("description") or "") + (a.get("title") or ""))
        if combined_len < 50:
            skipped_short += 1
            logger.debug("Skipping short article: %s", a.get("title") or a.get("url"))
            continue
        if PREFILTER_ARTICLES and not is_relevant(a):
            skipped_irrelevant += 1
            logger.debug("Skipping irrelevant article (no NSE/macro hit): %s", a.get("title") or a.get("url"))
            continue
        pred = call_llm(a)
        called += 1
        # Don't trust the model to echo the date back — backfill from the article
        # so downstream price alignment never silently fails on a missing date.
        if isinstance(pred, dict) and not pred.get("article_date"):
            pred["article_date"] = a.get("published_at") or a.get("publishedAt") or ""
        results.append({"article": a, "prediction": pred})
        time.sleep(0.3)

    logger.info(
        "extract_all: %d articles -> %d LLM calls (skipped %d short, %d irrelevant)",
        len(articles), called, skipped_short, skipped_irrelevant,
    )

    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d extractions to %s", len(results), out_file)
    except Exception:
        logger.exception("Failed to save extractions file")

    return results
