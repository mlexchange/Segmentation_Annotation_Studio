"""Tiled client factory and browse-container discovery.

Only servers declared by :func:`tiled_config.get_tiled_servers` may be opened.
The public API can still send the URI returned by ``/api/config/servers`` for
backwards compatibility, but it cannot turn the backend into a generic HTTP
client or choose which credential is attached to a request.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tiled_config import get_tiled_base, get_tiled_servers

_client_cache: dict[tuple[str, str], Any] = {}


class TiledClientConfigurationError(RuntimeError):
    """Base class for safe, user-facing Tiled configuration failures."""


class TiledServerNotConfigured(TiledClientConfigurationError):
    """Raised before networking when a URI is not in the server allowlist."""


class TiledServerCredentialError(TiledClientConfigurationError):
    """Raised when credential isolation cannot be guaranteed."""


def _canonical_uri(uri: str) -> str:
    """Return a comparison-only canonical HTTP(S) URI.

    The configured URI, rather than this normalized value, is ultimately sent
    to Tiled. Rejecting URL credentials and query strings also prevents API
    keys from being smuggled through a URI and written to logs.
    """
    try:
        parsed = urlsplit(uri.strip())
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise TiledServerNotConfigured("Tiled server is not configured") from exc

    scheme = parsed.scheme.lower()
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TiledServerNotConfigured("Tiled server is not configured")

    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise TiledServerNotConfigured("Tiled server is not configured") from exc
    if ":" in host:
        host = f"[{host}]"
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, host, path, "", ""))


def resolve_tiled_server(server_uri: str | None = None) -> tuple[str, str | None]:
    """Resolve a public URI to one exact server-side configuration.

    Args:
        server_uri: URI previously returned by ``/api/config/servers``. ``None``
            selects the configured default server.

    Returns:
        ``(configured_uri, configured_api_key)``. The configured URI is
        returned verbatim except for a trailing slash.

    Raises:
        TiledServerNotConfigured: If the requested URI is not allowlisted.
        TiledServerCredentialError: If an anonymous remote could implicitly
            inherit the process-wide default key from the Tiled library.
    """
    requested = _canonical_uri(server_uri or get_tiled_base())
    for cfg in get_tiled_servers().values():
        configured_uri = str(cfg.get("uri") or "").rstrip("/")
        try:
            configured = _canonical_uri(configured_uri)
        except TiledServerNotConfigured:
            # A bad operator entry is never a reason to weaken the allowlist.
            continue
        if configured != requested:
            continue

        api_key = cfg.get("api_key")
        if api_key is None and os.getenv("TILED_API_KEY"):
            # tiled.client.from_uri(api_key=None) silently reads TILED_API_KEY.
            # Refuse the ambiguous configuration instead of sending the default
            # server's key to a different host. Operators may explicitly assign
            # a per-server key, or run without a process-wide default key.
            raise TiledServerCredentialError(
                "Configured Tiled server needs an explicit per-server credential"
            )
        return configured_uri, api_key

    raise TiledServerNotConfigured("Tiled server is not configured")


def get_tiled_client(server_uri: str | None = None) -> Any:
    """Return a cached client for an allowlisted, server-configured URI."""
    from tiled.client import from_uri  # local import — heavy dep

    uri, api_key = resolve_tiled_server(server_uri)
    credential_id = hashlib.sha256((api_key or "").encode()).hexdigest()
    cache_key = (uri, credential_id)

    client = _client_cache.get(cache_key)
    if client is not None:
        return client

    if api_key:
        client = from_uri(uri, api_key=api_key)
    else:
        # resolve_tiled_server has proved there is no process-wide default key,
        # so Tiled cannot silently add a credential here.
        client = from_uri(uri)
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
    # Plain drag-and-drop ingest writes samples directly under `browse/`; catch
    # this before falling back to the root (which would list `browse` itself as
    # a single sample instead of its contents).
    (("browse",), "browse"),
)


def get_browse_container(client: Any) -> tuple[Any, str]:
    """Return ``(container_node, path_prefix)`` for the Metadata Browser root.

    Checks ``TILED_BROWSE_PATH`` env var first (slash-separated path into the
    Tiled tree, e.g. ``20260221_135217_petiole22_``). Then walks known
    beamline-ish paths in priority order. Falls back to the client's root.
    """
    browse_path = (os.getenv("TILED_BROWSE_PATH") or "").strip().strip("/")
    if browse_path:
        try:
            node: Any = client
            for k in browse_path.split("/"):
                node = node[k]
            if len(node) > 0:
                return node, browse_path
        except (KeyError, TypeError):
            pass

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


def get_browse_container_for(
    client: Any, container_path: str | None
) -> tuple[Any, str]:
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
