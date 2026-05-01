"""Unit tests for content-addressable screen fingerprinting."""

from __future__ import annotations


def test_normalize_text_lowercases_and_collapses_whitespace():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("  Hello   World  ") == "hello world"


def test_normalize_text_strips_non_alphanumeric_except_spaces():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("Hello, World! 123.") == "hello world 123"


def test_normalize_text_empty_input_returns_empty_string():
    from peeklet.core.fingerprint import _normalize_text

    assert _normalize_text("") == ""
    assert _normalize_text("   \t\n  ") == ""


def test_normalize_url_preserves_slashes_and_dots():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("https://Example.COM/path/to?q=1") == "https://example.com/path/to"


def test_normalize_url_strips_query_and_fragment():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("https://example.com/x?a=b&c=d#frag") == "https://example.com/x"


def test_normalize_url_collapses_whitespace_inside_token():
    from peeklet.core.fingerprint import _normalize_url

    assert _normalize_url("  https://example.com  ") == "https://example.com"
