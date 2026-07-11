"""Tests for config validation."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


def _problems(**over):
    """Run validate_config with temporarily patched module globals."""
    saved = {k: getattr(config, k) for k in over}
    try:
        for k, v in over.items():
            setattr(config, k, v)
        return config.validate_config()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def test_flags_placeholder_key():
    probs = _problems(OPENAI_API_KEY="sk-...", OPENAI_BASE_URL="https://api.openai.com/v1")
    assert any("OPENAI_API_KEY" in p for p in probs)


def test_flags_bad_base_url():
    probs = _problems(OPENAI_API_KEY="real-key", OPENAI_BASE_URL="not-a-url")
    assert any("OPENAI_BASE_URL" in p for p in probs)


def test_ok_config_has_no_problems():
    probs = _problems(OPENAI_API_KEY="sk-realkey123", OPENAI_BASE_URL="https://api.openai.com/v1",
                      MAX_TOKENS=600, CONFIDENCE_THRESHOLD=0.3)
    assert probs == []


def test_ollama_local_is_ok():
    probs = _problems(OPENAI_API_KEY="ollama", OPENAI_BASE_URL="http://localhost:11434/v1",
                      MAX_TOKENS=600, CONFIDENCE_THRESHOLD=0.3)
    assert probs == []
