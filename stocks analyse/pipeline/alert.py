"""
alert.py — turn a graph-enriched prediction into an analyst-facing EVENT ALERT.

This is the product the evidence actually supports. A rigorous point-in-time
event study (see harness.py / the RESULTS notes) found:

  * predicting the DIRECTION of a stock's move from news has NO demonstrated edge
    on the available data — so we do NOT sell direction as a prediction.
  * but EVENT-FUL news (earnings, M&A, disasters, product/regulatory events) is
    associated with a higher probability of an abnormal MOVE than macro/general
    news, and the knowledge graph reliably maps WHICH NSE names are exposed.

So an alert answers the questions we can honestly answer:
  "What happened, how severe, which NSE names does it touch, and how likely is
   each of them to have an abnormal move?" — with direction shown for context
   only, explicitly not as a prediction.

Coefficients (EVENT_MOVE_PRIOR, BASE_MOVE_RATE) are calibratable and only
weakly grounded so far (few independent events); treat the score as a ranking,
not a probability to bet on.
"""
from __future__ import annotations

# Base rate of a >=1.5% abnormal move for a random liquid NSE ticker-day,
# measured on real prices (~35%). An event only matters if it lifts this.
BASE_MOVE_RATE = 0.35

# Per-event-type prior probability of an abnormal move for an exposed name.
# "Eventful" catalysts lift above base; macro/general news sits at base (noise).
EVENT_MOVE_PRIOR = {
    "merger_acquisition": 0.50,
    "disaster": 0.50,
    "earnings": 0.48,
    "product_launch": 0.45,
    "regulation": 0.42,
    "legal": 0.42,
    "management_change": 0.40,
    "commodity": 0.40,
    "macro": 0.35,
    "other": 0.35,
}

# Move-likelihood tiers for triage.
TIER_HIGH = 0.45
TIER_MEDIUM = 0.38


def move_likelihood(event_type: str, severity: float, confidence: float) -> float:
    """Estimated probability that an exposed name has an abnormal move.

    Starts from the event-type prior and regresses toward the base rate for mild
    events / weakly-coupled names — so only severe, strongly-connected exposures
    get the full lift. Returns a value in roughly [BASE_MOVE_RATE, prior].
    """
    prior = EVENT_MOVE_PRIOR.get(event_type or "", BASE_MOVE_RATE)
    try:
        sev = max(0.0, min(1.0, float(severity)))
    except (TypeError, ValueError):
        sev = 0.5
    try:
        conf = max(0.0, min(1.0, float(confidence) / 0.3))  # 0.3 coupling ~= full weight
    except (TypeError, ValueError):
        conf = 0.5
    strength = 0.6 * sev + 0.4 * conf
    return round(BASE_MOVE_RATE + (prior - BASE_MOVE_RATE) * strength, 3)


def _tier(score: float) -> str:
    return "HIGH" if score >= TIER_HIGH else "MEDIUM" if score >= TIER_MEDIUM else "LOW"


def build_alert(enriched: dict) -> dict:
    """Build an event alert from an enriched prediction (must have ``exposed``)."""
    event_type = enriched.get("event_type", "other")
    severity = enriched.get("severity", 0.5)
    sources = enriched.get("source_entities", []) or []
    exposed = enriched.get("exposed", []) or []

    names = []
    for e in exposed:
        score = move_likelihood(event_type, severity, e.get("confidence", 0.0))
        names.append({
            "ticker": e.get("ticker"),
            "move_likelihood": score,
            "tier": _tier(score),
            "direction_informational": e.get("direction"),   # NOT a prediction
            "confidence": e.get("confidence"),
            "impact_type": e.get("impact_type"),
            "via": e.get("reasoning"),
        })
    names.sort(key=lambda n: n["move_likelihood"], reverse=True)

    return {
        "event_type": event_type,
        "severity": severity,
        "source_entities": [s.get("name") for s in sources],
        "article_date": enriched.get("article_date"),
        "exposed_count": len(names),
        "top_tier": names[0]["tier"] if names else "LOW",
        "names": names,
    }


def render_alert(alert: dict, top: int = 8) -> str:
    """Human-readable one-block alert for an analyst feed."""
    lines = []
    sev = alert.get("severity")
    srcs = ", ".join(alert.get("source_entities") or []) or "—"
    lines.append(f"[{alert.get('top_tier','LOW')}] {alert.get('event_type','?').upper()} "
                 f"(severity {sev}) — {srcs}  {alert.get('article_date','')}")
    lines.append(f"  {alert.get('exposed_count',0)} NSE names exposed | "
                 f"move-likelihood ranked (direction = context only, NOT a prediction):")
    for n in alert.get("names", [])[:top]:
        lines.append(f"    {n['ticker']:6} move~{n['move_likelihood']:.0%} [{n['tier']:6}] "
                     f"dir(info)={n['direction_informational']:<7} via {n['impact_type']}")
    return "\n".join(lines)
