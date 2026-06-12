"""Canonical source-key parsing (mirrors frontend ``buildSourceKey``)."""

from __future__ import annotations

from tiled_config import get_tiled_servers


def parse_source_key(source_key: str) -> dict[str, str | None]:
    """Parse a draft/source key into kind, server URI, and path.

    Formats:
        ``tiled:<serverUri>:<tiledPath>``
        ``local:<relPath>``

    Args:
        source_key: Key used by the draft store and annotation store.

    Returns:
        Dict with ``kind`` (``tiled`` | ``local``), ``server_uri``, and ``path``.
    """
    if source_key.startswith("local:"):
        return {"kind": "local", "server_uri": None, "path": source_key[len("local:"):]}

    if source_key.startswith("tiled:"):
        rest = source_key[len("tiled:"):]
        uris = sorted(
            [
                (cfg["uri"].rstrip("/"), name)
                for name, cfg in get_tiled_servers().items()
                if cfg.get("uri")
            ],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        for uri, _ in uris:
            prefix = uri + ":"
            if rest.startswith(prefix):
                return {"kind": "tiled", "server_uri": uri, "path": rest[len(prefix):]}
        # Fallback: ``tiled:<path>`` with default server (empty URI segment).
        if rest.startswith(":"):
            return {"kind": "tiled", "server_uri": None, "path": rest[1:]}
        return {"kind": "tiled", "server_uri": None, "path": rest}

    return {"kind": "unknown", "server_uri": None, "path": source_key}
