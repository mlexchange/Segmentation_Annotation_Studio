"""Tiled client factory + browse-container discovery.

Clients are cached per ``(uri, api_key)`` so each request doesn't pay the cost
of a fresh HTTP connection. In practice the same process only talks to one or
two servers, so a tiny unbounded dict is fine.
"""

from __future__ import annotations

import os
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


# Well-known demo paths. Used as candidates alongside every top-level child of
# the catalog root; the auto-discoverer then picks the *largest* non-empty one
# so a leftover single ingest under ``browse/`` does not hide a fuller tree
# such as ``petioles/``.
_BROWSE_CANDIDATES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("browse", "generated_data"), "browse/generated_data"),
    (("beamlines", "bl733", "projects", "10k"), "beamlines/bl733/projects/10k"),
    (("beamlines", "bl733"), "beamlines/bl733"),
    (("beamlines", "bl901"), "beamlines/bl901"),
    (("petioles",), "petioles"),
    # Plain drag-and-drop ingest writes samples under `browse/`.
    (("browse",), "browse"),
)


def _try_navigate(client: Any, keys: tuple[str, ...]) -> Any | None:
    """Return ``client[keys...]`` or ``None`` if missing / not a container."""
    try:
        node: Any = client
        for k in keys:
            node = node[k]
        if len(node) > 0:
            return node
    except (KeyError, TypeError, AttributeError):
        return None
    return None


def get_browse_container(client: Any) -> tuple[Any, str]:
    """Return ``(container_node, path_prefix)`` for the Metadata Browser root.

    Resolution order:

    1. ``TILED_BROWSE_PATH`` env var (explicit override).
    2. Among known candidate paths **and** every non-empty top-level child of
       the root, pick the container with the most children (most samples).
    3. Fall back to the client's root.
    """
    browse_path = (os.getenv("TILED_BROWSE_PATH") or "").strip().strip("/")
    if browse_path:
        try:
            node: Any = client
            for k in browse_path.split("/"):
                node = node[k]
            if len(node) > 0:
                return node, browse_path
        except (KeyError, TypeError, AttributeError):
            pass

    scored: list[tuple[int, str, Any]] = []
    seen_prefixes: set[str] = set()

    for keys, prefix in _BROWSE_CANDIDATES:
        node = _try_navigate(client, keys)
        if node is None:
            continue
        scored.append((len(node), prefix, node))
        seen_prefixes.add(prefix.split("/")[0])

    # Top-level containers (e.g. ``petioles``) not already covered by a candidate.
    try:
        for key in list(client):
            name = str(key)
            if name in seen_prefixes:
                continue
            node = _try_navigate(client, (name,))
            if node is None:
                continue
            scored.append((len(node), name, node))
    except (TypeError, AttributeError):
        pass

    if scored:
        # Most samples first; stable tie-break by path for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))
        count, prefix, node = scored[0]
        return node, prefix

    return client, ""


def get_browse_container_for(client: Any, container_path: str | None) -> tuple[Any, str]:
    """Return ``(container_node, path_prefix)`` for a specific browse target.

    When *container_path* is given (slash-separated, e.g. ``browse/testset``),
    navigate directly to that node. Otherwise fall back to the heuristic
    discovery in :func:`get_browse_container`.
    """
    path = (container_path or "").strip().strip("/")
    if not path:
        return get_browse_container(client)
    node: Any = client
    for k in path.split("/"):
        node = node[k]
    return node, path
