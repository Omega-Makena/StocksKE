"""
End-to-end demo on REAL data with the local Ollama model.

  real Kenyan news (RSS + article body)  ->  local LLM extracts the source event
  ->  knowledge-graph propagation  ->  affected NSE tickers (direction / magnitude
  / confidence).

Runs on a small batch (LIMIT) because the local 3B model is slow on a 4GB GPU.
Writes a readable report to nse_dataset/demo_report.txt.
"""
import sys, time, json
from pathlib import Path

import collector, extractor, validator, graph
from config import OUTPUT_DIR

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4

out_lines = []
def log(s=""):
    print(s, flush=True)
    out_lines.append(s)

log("=" * 70)
log("  StocksKE end-to-end demo  (real news -> local LLM -> knowledge graph)")
log("=" * 70)

# 1) real news
t0 = time.time()
items = collector.scrape_rss_feeds()
relevant = [a for a in items if extractor.is_relevant(a)][:LIMIT]
log(f"\nCollected {len(items)} live headlines; {len(relevant)} shown (NSE/macro-relevant).")
collector.enrich_articles_with_body(relevant, should_fetch=extractor.is_relevant, limit=LIMIT)

g = graph.build_default_graph()

for i, art in enumerate(relevant, 1):
    log("\n" + "-" * 70)
    log(f"[{i}/{len(relevant)}] {art['source']} | {art.get('published_at','')}")
    log(f"HEADLINE: {art['title']}")
    log(f"(body: {len(art.get('content') or '')} chars)")

    t = time.time()
    pred = extractor.call_llm(art)
    dt = time.time() - t
    if not pred:
        log(f"  -> LLM returned nothing ({dt:.0f}s)")
        continue

    errs = validator.validate(pred)
    if errs:
        log(f"  -> validation issues: {errs}")

    log(f"  LLM ({dt:.0f}s): event={pred.get('event_type')} severity={pred.get('severity')}")
    src = pred.get("source_entities", []) or []
    log(f"  source_entities: " + ", ".join(f"{e.get('name')}[{e.get('kind')},{e.get('direction')}]" for e in src))
    da = pred.get("directly_affected", []) or []
    if da:
        log("  directly_affected: " + ", ".join(f"{e.get('ticker')}({e.get('direction')})" for e in da))

    enriched = graph.enrich_prediction(pred, g)
    ind = enriched.get("indirectly_affected", []) or []
    log(f"  GRAPH propagation -> {len(ind)} affected NSE names:")
    for e in ind[:8]:
        log(f"     {e['ticker']:5} {e['direction']:5} {e.get('magnitude_pct',0):+6.2f}%  "
            f"conf={e.get('confidence',0):.2f}  [{e.get('impact_type')}]")

log("\n" + "=" * 70)
log(f"Done in {time.time()-t0:.0f}s. Full pipeline ran on REAL news with a LOCAL model, no API key.")
log("Next for a scored backtest: pull Innova prices for this news window so the")
log("harness can compare predictions against realised abnormal returns.")

report = Path(OUTPUT_DIR) / "demo_report.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text("\n".join(out_lines), encoding="utf-8")
print(f"\n[saved report -> {report}]")
