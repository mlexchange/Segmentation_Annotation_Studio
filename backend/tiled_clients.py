"""Tiled client factory + browse-container discovery.

Clients are cached per ``(uri, api_key)`` so each request doesn't pay the cost
of a fresh HTTP connection. In practice the same process only talks to one or
two servers, so a tiny unbounded dict is fine.
"""

from __future__ import annotations

from typing import Any

from tiled_config import get_tiled_api_key, get_tiled_base, get_tiled_servers

_client_cache: dict[tuple[str, str], Any] = {}


def api_key_for_uri(uri: str | None) -> str | None:
    """Look up a configured API key for *uri*, falling back to None."""
    if not uri:
        return None
    needle = uri.rstrip("/")
    for cfg in get_tiled_servers().values():
        if (cfg.get("uri") or "").rstrip("/") == needle:
            return cfg.get("api_key")
    return None


def get_tiled_client(server_uri: str | None = None, server_api_key: str | None = None) -> Any:
    """Return a (cached) ``tiled.client`` connected to the requested server."""
    from tiled.client import from_uri  # local import — heavy dep

    uri = (server_uri or get_tiled_base()).rstrip("/")
    api_key = server_api_key or api_key_for_uri(uri) or get_tiled_api_key()
    cache_key = (uri, api_key or "")

    client = _client_cache.get(cache_key)
    if client is not None:
        return client

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = from_uri(uri, **kwargs)
    _client_cache[cache_key] = client
    return client


# Paths checked in order when looking for the Browse root container. The first
# non-empty path wins. Falling back to the root container is valid but usually
# means the user hasn't seeded any data yet.
_BROWSE_CANDIDATES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("browse", "generated_data"), "browse/generated_data"),
    (("beamlines", "bl733", "projects", "10k"), "beamlines/bl733/projects/10k"),
    (("beamlines", "bl733"), "beamlines/bl733"),
    (("beamlines", "bl901"), "beamlines/bl901"),
)


def get_browse_container(client: Any) -> tuple[Any, str]:
    """Return ``(container_node, path_prefix)`` for the Metadata Browser root.

    Walks known beamline-ish paths in priority order and returns the first
    container that exists and is non-empty. Falls back to the client's root.
    """
    for keys, prefix in _BROWSE_CANDIDATES:
        try:
            node: Any = client
            for k in keys:
                node = node[k]
            if len(node) > 0:
                return node, prefix
        except (KeyError, TypeError):
            continue
    return client, ""
