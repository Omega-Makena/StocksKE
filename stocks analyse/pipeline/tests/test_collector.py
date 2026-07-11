"""Tests for the news collector — RSS parsing and date handling (offline)."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import collector as C


RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Biz</title>
  <item>
    <title>Safaricom posts record profit</title>
    <description>The telco reported a jump in earnings.</description>
    <link>https://example.com/a1</link>
    <pubDate>Fri, 10 Jul 2026 12:54:18 +0300</pubDate>
  </item>
  <item>
    <title>CBK raises benchmark rate</title>
    <description>Central bank tightens policy.</description>
    <link>https://example.com/a2</link>
    <pubDate>Thu, 09 Jul 2026 07:00:00 +0300</pubDate>
  </item>
</channel></rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>KCB earnings beat</title>
    <summary>Profit up 20 percent.</summary>
    <link href="https://example.com/atom1"/>
    <updated>2026-07-08T09:00:00Z</updated>
  </entry>
</feed>"""


class TestParseRss:
    def test_extracts_rss_items(self):
        items = C.parse_rss(RSS_SAMPLE, "Standard")
        assert len(items) == 2
        a = items[0]
        assert a["title"] == "Safaricom posts record profit"
        assert a["description"].startswith("The telco")
        assert a["url"] == "https://example.com/a1"
        assert a["source"] == "Standard"
        assert a["published_at"] == "2026-07-10"     # RFC-822 normalised to ISO
        assert a["ticker"] == "UNKNOWN"

    def test_parses_atom_entries(self):
        items = C.parse_rss(ATOM_SAMPLE, "Nation")
        assert len(items) == 1
        assert items[0]["title"] == "KCB earnings beat"
        assert items[0]["url"] == "https://example.com/atom1"
        assert items[0]["published_at"] == "2026-07-08"

    def test_malformed_xml_returns_empty(self):
        assert C.parse_rss(b"<not xml", "X") == []

    def test_skips_items_without_title_or_link(self):
        xml = b"<rss><channel><item><description>x</description></item></channel></rss>"
        assert C.parse_rss(xml, "X") == []


class TestRssDate:
    def test_rfc822(self):
        assert C._rss_date_to_iso("Fri, 10 Jul 2026 12:54:18 +0300") == "2026-07-10"

    def test_iso_input(self):
        assert C._rss_date_to_iso("2026-07-08T09:00:00Z") in ("2026-07-08", "")

    def test_empty_and_garbage(self):
        assert C._rss_date_to_iso("") == ""
        assert C._rss_date_to_iso("not a date") == ""


class TestScrapeRssFeeds:
    def test_http_error_is_skipped(self):
        resp = MagicMock(status_code=403)
        with patch("collector.requests.get", return_value=resp):
            out = C.scrape_rss_feeds([("Src", "http://x/feed")], sleep_s=0)
        assert out == []

    def test_success_returns_parsed_items(self):
        resp = MagicMock(status_code=200, content=RSS_SAMPLE)
        with patch("collector.requests.get", return_value=resp):
            out = C.scrape_rss_feeds([("Standard", "http://x/feed")], sleep_s=0)
        assert len(out) == 2
        assert {i["title"] for i in out} == {"Safaricom posts record profit",
                                             "CBK raises benchmark rate"}

    def test_network_error_never_raises(self):
        import requests
        with patch("collector.requests.get", side_effect=requests.RequestException("boom")):
            assert C.scrape_rss_feeds([("Src", "http://x/feed")], sleep_s=0) == []


ARTICLE_HTML = """
<html><body>
  <nav><p>Home About Menu</p></nav>
  <article>
    <p>Short.</p>
    <p>Safaricom reported a sharp rise in full-year net profit driven by M-Pesa revenue growth and a larger subscriber base across the region.</p>
    <p>The telco said data usage climbed and it expects the momentum to continue into the next financial year.</p>
  </article>
  <footer><p>Copyright 2026 all rights reserved contact us</p></footer>
</body></html>
"""


class TestFetchArticleBody:
    def test_extracts_main_paragraphs(self):
        resp = MagicMock(status_code=200, text=ARTICLE_HTML,
                         headers={"Content-Type": "text/html; charset=utf-8"})
        with patch("collector._robots_allowed", return_value=True), \
             patch("collector.requests.get", return_value=resp):
            body = C.fetch_article_body("https://example.com/a")
        assert "Safaricom reported a sharp rise" in body
        assert "M-Pesa" in body
        assert "Home About Menu" not in body        # nav stripped
        assert "Short." not in body                 # <30-char scrap dropped

    def test_respects_robots_disallow(self):
        with patch("collector._robots_allowed", return_value=False):
            assert C.fetch_article_body("https://example.com/a") == ""

    def test_non_html_returns_empty(self):
        resp = MagicMock(status_code=200, text="{}", headers={"Content-Type": "application/json"})
        with patch("collector._robots_allowed", return_value=True), \
             patch("collector.requests.get", return_value=resp):
            assert C.fetch_article_body("https://example.com/a.json") == ""

    def test_network_error_returns_empty(self):
        import requests
        with patch("collector._robots_allowed", return_value=True), \
             patch("collector.requests.get", side_effect=requests.RequestException("x")):
            assert C.fetch_article_body("https://example.com/a") == ""


class TestEnrichBodies:
    def test_respects_limit_and_should_fetch(self):
        arts = [{"url": f"http://x/{i}", "content": "", "title": f"t{i}"} for i in range(5)]
        # only even-indexed are "relevant"; cap at 2
        should = lambda a: a["title"] in ("t0", "t2", "t4")
        with patch("collector.fetch_article_body", return_value="BODY"):
            n = C.enrich_articles_with_body(arts, should_fetch=should, limit=2, sleep_s=0)
        assert n == 2
        assert arts[0]["content"] == "BODY" and arts[2]["content"] == "BODY"
        assert arts[1]["content"] == ""             # not relevant, untouched

    def test_skips_articles_that_already_have_content(self):
        arts = [{"url": "http://x/1", "content": "already", "title": "t"}]
        with patch("collector.fetch_article_body", return_value="NEW") as m:
            n = C.enrich_articles_with_body(arts, sleep_s=0)
        assert n == 0
        m.assert_not_called()
        assert arts[0]["content"] == "already"


class TestDedupStore:
    def test_roundtrip(self, tmp_path):
        with patch("collector.OUTPUT_DIR", str(tmp_path)), \
             patch("collector.DEDUP_STORE", "seen.json"):
            assert C._load_seen() == set()
            C._save_seen({"http://a", "http://b"})
            assert C._load_seen() == {"http://a", "http://b"}

    def test_disabled_when_empty(self, tmp_path):
        with patch("collector.DEDUP_STORE", ""):
            assert C._seen_store_path() is None
            C._save_seen({"http://a"})   # no-op, must not raise
            assert C._load_seen() == set()


class TestRobotsFailMode:
    def test_fail_open_allows_on_error(self):
        with patch("collector.ROBOTS_FAIL_OPEN", True), \
             patch("collector.robotparser.RobotFileParser", side_effect=Exception("boom")):
            assert C._robots_allowed("https://x/page") is True

    def test_fail_closed_blocks_on_error(self):
        with patch("collector.ROBOTS_FAIL_OPEN", False), \
             patch("collector.robotparser.RobotFileParser", side_effect=Exception("boom")):
            assert C._robots_allowed("https://x/page") is False
