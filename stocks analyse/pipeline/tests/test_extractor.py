"""Tests for nse_predictor/extractor.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from extractor import build_user_message, _strip_fences
import json


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

    def test_truncates_long_content_to_2000_chars(self):
        long = "x" * 5000
        msg = build_user_message(self._article(content=long))
        assert len(json.loads(msg)["article_text"]) == 2000

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
