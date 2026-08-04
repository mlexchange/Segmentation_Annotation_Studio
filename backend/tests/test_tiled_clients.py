"""Security-boundary tests for configured Tiled clients."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest


def _reload_tiled_modules() -> tuple[object, object]:
    """Reload configuration and client modules after an environment change."""
    import tiled_clients
    import tiled_config

    importlib.reload(tiled_config)
    return tiled_config, importlib.reload(tiled_clients)


def test_unknown_tiled_uri_is_rejected_before_connection() -> None:
    """An arbitrary URI must never receive the default Tiled credential."""
    env = {
        "TILED_URI": "http://127.0.0.1:8010",
        "TILED_API_KEY": "default-secret",
    }
    with patch.dict(os.environ, env, clear=True):
        _, tiled_clients = _reload_tiled_modules()
        with patch("tiled.client.from_uri") as from_uri:
            with pytest.raises(tiled_clients.TiledServerNotConfigured):
                tiled_clients.get_tiled_client("https://attacker.example")

        from_uri.assert_not_called()


def test_configured_server_uses_only_its_own_key() -> None:
    """A configured remote server must not inherit the default server key."""
    env = {
        "TILED_URI": "http://127.0.0.1:8010",
        "TILED_API_KEY": "default-secret",
        "TILED_SERVER_1_NAME": "Remote",
        "TILED_SERVER_1_URI": "https://data.example/api/",
        "TILED_SERVER_1_API_KEY": "remote-secret",
    }
    sentinel = object()
    with patch.dict(os.environ, env, clear=True):
        _, tiled_clients = _reload_tiled_modules()
        with patch("tiled.client.from_uri", return_value=sentinel) as from_uri:
            client = tiled_clients.get_tiled_client("HTTPS://DATA.EXAMPLE/api")

        assert client is sentinel
        from_uri.assert_called_once_with(
            "https://data.example/api",
            api_key="remote-secret",
        )


def test_configured_anonymous_server_cannot_inherit_global_key() -> None:
    """Fail closed when Tiled itself would implicitly consume TILED_API_KEY."""
    env = {
        "TILED_URI": "http://127.0.0.1:8010",
        "TILED_API_KEY": "default-secret",
        "TILED_SERVER_1_NAME": "Public",
        "TILED_SERVER_1_URI": "https://public.example",
        "TILED_SERVER_1_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=True):
        _, tiled_clients = _reload_tiled_modules()
        with patch("tiled.client.from_uri") as from_uri:
            with pytest.raises(tiled_clients.TiledServerCredentialError):
                tiled_clients.get_tiled_client("https://public.example")

        from_uri.assert_not_called()


@pytest.mark.parametrize(
    "uri",
    [
        "https://data.example/api?api_key=stolen",
        "https://user:password@data.example/api",
        "file:///etc/passwd",
    ],
)
def test_unsafe_tiled_uri_forms_are_rejected(uri: str) -> None:
    """Credentials, query strings, and non-HTTP schemes are never accepted."""
    env = {
        "TILED_URI": "https://data.example/api",
        "TILED_API_KEY": "secret",
    }
    with patch.dict(os.environ, env, clear=True):
        _, tiled_clients = _reload_tiled_modules()
        with pytest.raises(tiled_clients.TiledServerNotConfigured):
            tiled_clients.get_tiled_client(uri)
