"""Tests for source_keys.parse_source_key."""
from __future__ import annotations

import pytest

import source_keys
import tiled_config


@pytest.fixture()
def two_servers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        tiled_config, "get_tiled_servers",
        lambda: {
            "local": {"uri": "http://127.0.0.1:8010"},
            "remote": {"uri": "http://example.com:9000/"},
        },
    )
    monkeypatch.setattr(source_keys, "get_tiled_servers", tiled_config.get_tiled_servers)


class TestLocalKeys:
    def test_parses_local_prefix(self):
        assert source_keys.parse_source_key("local:foo/bar.tif") == {
            "kind": "local", "server_uri": None, "path": "foo/bar.tif",
        }

    def test_empty_path_after_prefix(self):
        assert source_keys.parse_source_key("local:") == {
            "kind": "local", "server_uri": None, "path": "",
        }


class TestTiledKeys:
    def test_matches_known_server_uri(self, two_servers):
        result = source_keys.parse_source_key("tiled:http://127.0.0.1:8010:browse/sample")
        assert result == {"kind": "tiled", "server_uri": "http://127.0.0.1:8010", "path": "browse/sample"}

    def test_trailing_slash_on_configured_uri_is_normalized(self, two_servers):
        result = source_keys.parse_source_key("tiled:http://example.com:9000:browse/x")
        assert result["server_uri"] == "http://example.com:9000"
        assert result["path"] == "browse/x"

    def test_prefers_longest_matching_uri(self, monkeypatch: pytest.MonkeyPatch):
        # "http://x:1" is a prefix of "http://x:1/extra" — the longer match
        # must win so the path split lands in the right place.
        monkeypatch.setattr(
            tiled_config, "get_tiled_servers",
            lambda: {
                "short": {"uri": "http://x:1"},
                "long": {"uri": "http://x:1/extra"},
            },
        )
        monkeypatch.setattr(source_keys, "get_tiled_servers", tiled_config.get_tiled_servers)
        result = source_keys.parse_source_key("tiled:http://x:1/extra:browse/s")
        assert result == {"kind": "tiled", "server_uri": "http://x:1/extra", "path": "browse/s"}

    def test_unmatched_uri_falls_back_to_default_server_with_colon(self, two_servers):
        result = source_keys.parse_source_key("tiled::browse/sample")
        assert result == {"kind": "tiled", "server_uri": None, "path": "browse/sample"}

    def test_unmatched_uri_falls_back_to_default_server_without_colon(self, two_servers):
        result = source_keys.parse_source_key("tiled:browse/sample")
        assert result == {"kind": "tiled", "server_uri": None, "path": "browse/sample"}

    def test_no_configured_servers_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tiled_config, "get_tiled_servers", lambda: {})
        monkeypatch.setattr(source_keys, "get_tiled_servers", tiled_config.get_tiled_servers)
        result = source_keys.parse_source_key("tiled:browse/sample")
        assert result == {"kind": "tiled", "server_uri": None, "path": "browse/sample"}

    def test_server_with_no_uri_configured_is_skipped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tiled_config, "get_tiled_servers",
            lambda: {"broken": {}, "ok": {"uri": "http://ok:1"}},
        )
        monkeypatch.setattr(source_keys, "get_tiled_servers", tiled_config.get_tiled_servers)
        result = source_keys.parse_source_key("tiled:http://ok:1:browse/s")
        assert result["server_uri"] == "http://ok:1"


class TestUnknownKeys:
    def test_unrecognized_prefix_returns_unknown_kind(self):
        assert source_keys.parse_source_key("weird:thing") == {
            "kind": "unknown", "server_uri": None, "path": "weird:thing",
        }

    def test_empty_string(self):
        assert source_keys.parse_source_key("") == {
            "kind": "unknown", "server_uri": None, "path": "",
        }
