"""Tests for tiled_clients.py using plain nested dicts as fake Tiled nodes
(only __getitem__/__len__ are needed) and a monkeypatched tiled.client.from_uri
so get_tiled_client never makes a real network call.
"""
from __future__ import annotations

import pytest

import tiled_clients
import tiled_config


@pytest.fixture(autouse=True)
def clear_client_cache():
    tiled_clients._client_cache.clear()
    yield
    tiled_clients._client_cache.clear()


# ---------------------------------------------------------------------------
# api_key_for_uri
# ---------------------------------------------------------------------------

class TestApiKeyForUri:
    def test_none_uri_returns_none(self):
        assert tiled_clients.api_key_for_uri(None) is None

    def test_matches_configured_server_ignoring_trailing_slash(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tiled_config, "get_tiled_servers",
            lambda: {"local": {"uri": "http://x:1/", "api_key": "secret"}},
        )
        monkeypatch.setattr(tiled_clients, "get_tiled_servers", tiled_config.get_tiled_servers)
        assert tiled_clients.api_key_for_uri("http://x:1") == "secret"

    def test_unmatched_uri_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tiled_config, "get_tiled_servers", lambda: {"local": {"uri": "http://x:1"}})
        monkeypatch.setattr(tiled_clients, "get_tiled_servers", tiled_config.get_tiled_servers)
        assert tiled_clients.api_key_for_uri("http://other:2") is None


# ---------------------------------------------------------------------------
# get_tiled_client
# ---------------------------------------------------------------------------

class TestGetTiledClient:
    def test_caches_by_uri_and_api_key(self, monkeypatch: pytest.MonkeyPatch):
        calls = []

        def fake_from_uri(uri, **kwargs):
            calls.append((uri, kwargs))
            return object()

        monkeypatch.setattr("tiled.client.from_uri", fake_from_uri)
        monkeypatch.setattr(tiled_clients, "get_tiled_base", lambda: "http://default:1")
        monkeypatch.setattr(tiled_clients, "get_tiled_api_key", lambda: None)
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)

        first = tiled_clients.get_tiled_client("http://x:1")
        second = tiled_clients.get_tiled_client("http://x:1")
        assert first is second
        assert len(calls) == 1

    def test_different_uri_creates_a_new_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: object())
        monkeypatch.setattr(tiled_clients, "get_tiled_base", lambda: "http://default:1")
        monkeypatch.setattr(tiled_clients, "get_tiled_api_key", lambda: None)
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)

        a = tiled_clients.get_tiled_client("http://x:1")
        b = tiled_clients.get_tiled_client("http://y:1")
        assert a is not b

    def test_falls_back_to_default_base_uri(self, monkeypatch: pytest.MonkeyPatch):
        seen_uris = []
        monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: seen_uris.append(uri) or object())
        monkeypatch.setattr(tiled_clients, "get_tiled_base", lambda: "http://default:1")
        monkeypatch.setattr(tiled_clients, "get_tiled_api_key", lambda: None)
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)

        tiled_clients.get_tiled_client(None)
        assert seen_uris == ["http://default:1"]

    def test_explicit_api_key_wins_over_configured_and_default(self, monkeypatch: pytest.MonkeyPatch):
        seen_kwargs = []
        monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: seen_kwargs.append(kwargs) or object())
        monkeypatch.setattr(tiled_clients, "get_tiled_base", lambda: "http://default:1")
        monkeypatch.setattr(tiled_clients, "get_tiled_api_key", lambda: "global-key")
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: "configured-key")

        tiled_clients.get_tiled_client("http://x:1", "explicit-key")
        assert seen_kwargs == [{"api_key": "explicit-key"}]

    def test_no_api_key_omits_the_kwarg(self, monkeypatch: pytest.MonkeyPatch):
        seen_kwargs = []
        monkeypatch.setattr("tiled.client.from_uri", lambda uri, **kwargs: seen_kwargs.append(kwargs) or object())
        monkeypatch.setattr(tiled_clients, "get_tiled_base", lambda: "http://default:1")
        monkeypatch.setattr(tiled_clients, "get_tiled_api_key", lambda: None)
        monkeypatch.setattr(tiled_clients, "api_key_for_uri", lambda uri: None)

        tiled_clients.get_tiled_client("http://x:1")
        assert seen_kwargs == [{}]


# ---------------------------------------------------------------------------
# get_browse_container / get_browse_container_for
# ---------------------------------------------------------------------------

class TestGetBrowseContainer:
    def test_env_var_path_wins_when_present_and_non_empty(self, monkeypatch: pytest.MonkeyPatch):
        client = {"custom": {"path": {"a": 1, "b": 2}}}
        monkeypatch.setenv("TILED_BROWSE_PATH", "custom/path")
        node, prefix = tiled_clients.get_browse_container(client)
        assert node == {"a": 1, "b": 2}
        assert prefix == "custom/path"

    def test_env_var_path_ignored_when_empty_container(self, monkeypatch: pytest.MonkeyPatch):
        client = {"custom": {"path": {}}, "browse": {"s1": {}}}
        monkeypatch.setenv("TILED_BROWSE_PATH", "custom/path")
        node, prefix = tiled_clients.get_browse_container(client)
        assert prefix == "browse"

    def test_env_var_path_ignored_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        client = {"browse": {"s1": {}}}
        monkeypatch.setenv("TILED_BROWSE_PATH", "does/not/exist")
        node, prefix = tiled_clients.get_browse_container(client)
        assert prefix == "browse"

    def test_first_matching_candidate_wins_in_priority_order(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {
            "beamlines": {"bl733": {"s1": {}}},
            "browse": {"s2": {}},
        }
        node, prefix = tiled_clients.get_browse_container(client)
        # "beamlines/bl733" is checked before the plain "browse" fallback.
        assert prefix == "beamlines/bl733"

    def test_falls_back_to_plain_browse(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {"browse": {"s1": {}}}
        node, prefix = tiled_clients.get_browse_container(client)
        assert prefix == "browse"
        assert node == {"s1": {}}

    def test_empty_candidate_containers_are_skipped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {"browse": {"generated_data": {}}, "beamlines": {"bl733": {"s1": {}}}}
        node, prefix = tiled_clients.get_browse_container(client)
        assert prefix == "beamlines/bl733"

    def test_falls_back_to_client_root_when_nothing_matches(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {}
        node, prefix = tiled_clients.get_browse_container(client)
        assert node is client
        assert prefix == ""


class TestGetBrowseContainerFor:
    def test_explicit_path_navigates_directly(self):
        client = {"a": {"b": {"x": 1}}}
        node, prefix = tiled_clients.get_browse_container_for(client, "a/b")
        assert node == {"x": 1}
        assert prefix == "a/b"

    def test_strips_leading_and_trailing_slashes(self):
        client = {"a": {"x": 1}}
        node, prefix = tiled_clients.get_browse_container_for(client, "/a/")
        assert node == {"x": 1}
        assert prefix == "a"

    def test_no_path_falls_back_to_heuristic_discovery(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {"browse": {"s1": {}}}
        node, prefix = tiled_clients.get_browse_container_for(client, None)
        assert prefix == "browse"

    def test_empty_string_path_falls_back_to_heuristic_discovery(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
        client = {"browse": {"s1": {}}}
        node, prefix = tiled_clients.get_browse_container_for(client, "   ")
        assert prefix == "browse"
