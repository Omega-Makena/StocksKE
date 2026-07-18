"""Tests for nse_predictor/extractor.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from extractor import build_user_message, _strip_fences, is_relevant, SYSTEM_PROMPT
import json


class TestRelevancePrefilter:
    # relevance now needs BOTH an NSE/macro signal AND a Kenyan-context term,
    # so foreign news whose entities collide with NSE tickers is dropped.
    def test_kenyan_company_news_is_relevant(self):
        assert is_relevant({"title": "Safaricom reports record profit"})  # Safaricom = context
        assert is_relevant({"title": "KCB shares rally", "description": "on the Nairobi bourse"})
        assert is_relevant({"title": "Bamburi Cement dividend", "description": "Kenya cement maker"})

    def test_kenyan_macro_is_relevant(self):
        assert is_relevant({"title": "CBK raises benchmark interest rate"})
        assert is_relevant({"description": "The shilling weakened against the dollar"})

    def test_trusted_kenyan_source_relevant_on_signal_alone(self):
        # a Kenyan RSS source is trusted without an explicit geographic term
        assert is_relevant({"title": "KCB posts strong half-year earnings", "source": "Standard"})
        assert is_relevant({"title": "Equity Group profit up", "source": "Nation"})

    def test_untrusted_source_still_needs_context(self):
        # same headline from a global source (NewsAPI) without Kenyan context -> dropped
        assert not is_relevant({"title": "KCB posts strong half-year earnings", "source": "Reuters"})

    def test_foreign_news_colliding_with_tickers_is_dropped(self):
        # the whole point: these must NOT be processed
        assert not is_relevant({"title": "TCL launches new 27-inch QHD display"})   # TCL brand vs TransCentury
        assert not is_relevant({"title": "HCLTech posts strong quarterly earnings"})
        assert not is_relevant({"title": "Boeing 737 MAX grounded after US incident"})
        assert not is_relevant({"content": "Global oil price surged past $90"})     # no Kenyan context

    def test_irrelevant_article_is_skipped(self):
        assert not is_relevant({"title": "Local football derby ends in a draw"})
        assert not is_relevant({"title": "New smartphone released in the US market"})

    def test_empty_article_is_not_relevant(self):
        assert not is_relevant({})

    def test_ambiguous_place_or_common_words_do_not_match(self):
        # even with Kenyan context, place/political collisions must not trigger
        assert not is_relevant({"title": "Ol Kalou poll makes 'Limuru Four' inevitable in Kenya"})
        assert not is_relevant({"title": "Jubilee party leaders meet in Nairobi"})
        # a distinctive Kenyan name still triggers
        assert is_relevant({"title": "Safaricom half-year results beat forecasts"})


class TestPromptIsLean:
    def test_prompt_has_no_relationship_list(self):
        # relationship reasoning moved to the graph; must not bloat the prompt
        assert "RELATIONSHIP LIST" not in SYSTEM_PROMPT

    def test_prompt_lists_all_tickers(self):
        from companies import VALID_TICKERS
        for t in VALID_TICKERS:
            assert t in SYSTEM_PROMPT

    def test_prompt_is_smaller_than_legacy(self):
        # legacy prompt was ~6.6k chars; lean target is well under 4k
        assert len(SYSTEM_PROMPT) < 4000


class TestBuildUserMessage:
    def _article(self, **kwargs):
        base = {
            "title": "KCB posts profits",
            "description": "KCB Group reported...",
            "content": "Full article text here.",
            "published_at": "2024-03-01T10:00:00Z",
            "source": "Business Daily",
        }
        base.update(kwargs)
        return base

    def test_returns_valid_json(self):
        msg = build_user_message(self._article())
        parsed = json.loads(msg)
        assert "article_text" in parsed
        assert "article_date" in parsed
        assert "source" in parsed

    def test_prefers_content_over_description(self):
        msg = build_user_message(self._article(content="CONTENT", description="DESC"))
        assert json.loads(msg)["article_text"] == "CONTENT"

    def test_falls_back_to_description_when_no_content(self):
        # empty string is falsy — should fall back to description
        msg = build_user_message(self._article(content=""))
        assert json.loads(msg)["article_text"] == "KCB Group reported..."
        msg2 = build_user_message(self._article(content=None, description="DESC"))
        assert json.loads(msg2)["article_text"] == "DESC"

    def test_falls_back_to_title_when_no_content_or_description(self):
        msg = build_user_message(self._article(content=None, description=None))
        assert json.loads(msg)["article_text"] == "KCB posts profits"

    def test_truncates_long_content_to_configured_cap(self):
        from config import MAX_ARTICLE_CHARS
        long = "x" * 5000
        msg = build_user_message(self._article(content=long))
        assert len(json.loads(msg)["article_text"]) == MAX_ARTICLE_CHARS

    def test_uses_published_at_as_article_date(self):
        msg = build_user_message(self._article(published_at="2024-03-01"))
        assert json.loads(msg)["article_date"] == "2024-03-01"

    def test_falls_back_to_publishedAt_key(self):
        article = self._article()
        del article["published_at"]
        article["publishedAt"] = "2024-04-01"
        msg = build_user_message(article)
        assert json.loads(msg)["article_date"] == "2024-04-01"

    def test_empty_article_does_not_raise(self):
        msg = build_user_message({})
        parsed = json.loads(msg)
        assert parsed["article_text"] == ""


class TestStripFences:
    def test_strips_json_fence(self):
        s = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_fences(s)
        assert result == '{"key": "value"}'

    def test_strips_plain_fence(self):
        s = "```\n{\"key\": \"value\"}\n```"
        result = _strip_fences(s)
        assert result == '{"key": "value"}'

    def test_no_fence_unchanged(self):
        s = '{"key": "value"}'
        assert _strip_fences(s) == s

    def test_strips_whitespace(self):
        s = "   {\"key\": \"value\"}   "
        assert _strip_fences(s) == '{"key": "value"}'
