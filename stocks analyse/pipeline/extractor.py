import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path

import requests

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR

logger = logging.getLogger(__name__)
USER_AGENT = "NSE-Research-Bot/1.0"

SYSTEM_PROMPT = """
You are a financial analysis agent for the Nairobi Securities Exchange (NSE), Kenya.

Read the news article provided and return a JSON object with your analysis.

STRICT RULES:
1. Only reference companies from the COMPANY LIST below. Any ticker not in
   this list does not exist on the NSE. Do not invent tickers.
2. Only assert competitor relationships from the RELATIONSHIP LIST below.
   Do not invent relationships.
3. direction must be exactly one of: "UP", "DOWN", "NEUTRAL"
4. impact_type must be exactly one of: "direct", "competitor",
   "sector_spillover", "supplier_chain", "regulatory"
5. confidence is a float between 0.0 and 1.0. Do not include companies
   with confidence below 0.3.
6. Return ONLY valid JSON. No prose, no markdown fences.

COMPANY LIST:
ABSA-Absa Bank Kenya Plc|BKG-BK Group Plc|DTK-Diamond Trust Bank Kenya|
EQTY-Equity Group Holdings Plc|HFCK-HF Group Plc|IMH-I&M Holdings Plc|
KCB-KCB Group Plc|NCBA-NCBA Group Plc|SCBK-Standard Chartered Bank Kenya|
COOP-Co-operative Bank of Kenya|BRIT-Britam Holdings Plc|CIC-CIC Insurance Group|
JUB-Jubilee Holdings Ltd|KNRE-Kenya Reinsurance Corp|LBTY-Liberty Kenya Holdings|
SLAM-Sanlam Kenya Plc|SCOM-Safaricom Plc|KEGN-KenGen Plc|
KPLC-Kenya Power & Lighting Co.|TOTL-TotalEnergies Marketing Kenya|
UMME-Umeme Ltd|BAT-British American Tobacco Kenya|EABL-East African Breweries Ltd|
UNGA-Unga Group Plc|EVRD-Eveready East Africa|CARB-Carbacid Investments|
BOC-BOC Kenya|FTGH-Flame Tree Group Holdings|BAMB-Bamburi Cement Plc|
PORT-East African Portland Cement|CRWN-Crown Paints Kenya Plc|
CABL-East African Cables|EGAD-Eaagads Ltd|KUKZ-Kakuzi Plc|
KAPC-Kapchorua Tea|LIMT-Limuru Tea|SASN-Sasini Plc|WTK-Williamson Tea Kenya|
NMG-Nation Media Group|SGL-Standard Group Plc|SCAN-Scangroup Plc|
KQ-Kenya Airways Plc|TPSE-TPS Eastern Africa (Serena Hotels)|
CTUM-Centum Investment Company|TCL-TransCentury Plc|OCH-Olympia Capital Holdings|
NSE-Nairobi Securities Exchange Plc|HAFR-Home Afrika Ltd|CGEN-Car & General Kenya

RELATIONSHIP LIST (NSE-listed pairs only):
EQTY↔KCB|EQTY↔COOP|EQTY↔NCBA|EQTY↔ABSA|KCB↔NCBA|KCB↔ABSA|ABSA↔SCBK|
IMH↔KCB|IMH↔EQTY|DTK↔IMH|DTK↔KCB|HFCK↔COOP|HFCK↔KCB|
BRIT↔CIC|BRIT↔JUB|BRIT↔SLAM|CIC↔JUB|LBTY↔SLAM|LBTY↔BRIT|
CARB↔BOC|BAMB↔PORT|SASN↔KUKZ|WTK↔LIMT|LIMT↔KAPC|EGAD↔WTK|
NMG↔SGL|CTUM↔TCL|OCH↔CTUM

EXAMPLES:
Input: {"article_text": "KCB Group posts 20% profit jump", "article_date": "2024-03-01", "source": "Business Daily"}
Output: {"article_summary": "KCB reports 20% profit increase.", "event_type": "earnings", "primary_sector": "Banking", "directly_affected": [{"ticker": "KCB", "company": "KCB Group Plc", "impact_type": "direct", "direction": "UP", "confidence": 0.91, "reasoning": "Strong earnings beat typically drives buying pressure."}], "indirectly_affected": [{"ticker": "EQTY", "company": "Equity Group Holdings Plc", "impact_type": "competitor", "direction": "NEUTRAL", "confidence": 0.42, "reasoning": "Competitor earnings beat may shift investor attention but not necessarily Equity's fundamentals."}], "not_nse_listed": [], "macro_flags": {"currency_risk": false, "interest_rate_sensitive": false, "commodity_price_sensitive": false, "regulatory_change": false}, "data_quality": {"article_date": "2024-03-01", "source_credibility": "high", "ambiguity_notes": ""}}
"""


def build_user_message(article: dict) -> str:
    """
    Returns a JSON string with keys: article_text, article_date, source.
    article_text = article["content"] or article["description"] or article["title"]
    Truncate article_text to 2000 characters max.
    """
    text = (article.get("content") or article.get("description") or article.get("title") or "")
    text = text[:2000]
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
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
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
    Skips articles where content+description+title combined < 50 chars.
    Saves to OUTPUT_DIR/extractions/extractions_{timestamp}.jsonl
    Returns list of {article, prediction} dicts.
    Sleeps 0.3s between calls.
    """
    out_dir = Path(OUTPUT_DIR) / "extractions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_file = out_dir / f"extractions_{ts}.jsonl"

    results = []
    for a in articles:
        combined_len = len((a.get("content") or "") + (a.get("description") or "") + (a.get("title") or ""))
        if combined_len < 50:
            logger.info("Skipping short article: %s", a.get("title") or a.get("url"))
            continue
        pred = call_llm(a)
        results.append({"article": a, "prediction": pred})
        time.sleep(0.3)

    try:
        with out_file.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d extractions to %s", len(results), out_file)
    except Exception:
        logger.exception("Failed to save extractions file")

    return results
