"""Tiled server configuration loaded from environment variables.

Reads ``backend/.env`` (or process env) at import time and exposes a simple
read-only view of the configured servers. A default "Local Data (port 8010)"
entry is always present so the Browse UI has at least one option to show.

Environment variables
---------------------
* ``TILED_URI``            — default server URI (fallback: ``http://127.0.0.1:8010``)
* ``TILED_API_KEY``        — API key for the default URI
* ``TILED_LOCAL_API_KEY``  — override API key for the local (8010) server
* ``TILED_SERVER_{N}_NAME`` / ``..._URI`` / ``..._API_KEY`` — additional named
  servers (contiguous ``N = 1, 2, ...``; stop at first missing slot)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_LOCAL_URI = "http://127.0.0.1:8010"

# Repo-root tiled/config.yml (backend/ -> repo root -> tiled/config.yml).
_TILED_CONFIG_PATH = Path(__file__).resolve().parent.parent / "tiled" / "config.yml"


def clear_empty_tiled_api_key_env() -> None:
    """Delete blank ``TILED_API_KEY`` / ``TILED_LOCAL_API_KEY`` from the process env.

    A fresh install ships ``.env`` with an empty ``TILED_API_KEY``; ``start_all.sh``
    exports it via ``set -a``. The Tiled client library reads that env var directly
    and builds an ``Authorization: Apikey `` (trailing space) header, which httpx
    rejects (``Illegal header value b'Apikey '``). Removing blank key vars entirely
    means neither our code nor the Tiled library ever sees an empty key.
    """
    for var in ("TILED_API_KEY", "TILED_LOCAL_API_KEY"):
        if var in os.environ and not os.environ[var].strip():
            del os.environ[var]


load_dotenv()
clear_empty_tiled_api_key_env()


def _stripped(name: str) -> str:
    return (os.getenv(name) or "").strip()


def single_user_api_key_from_tiled_config(path: Path | None = None) -> str | None:
    """Return ``authentication.single_user_api_key`` from ``tiled/config.yml``, or None.

    Anonymous access is read-only; writes (ingest) need this key. Used as a fallback
    when no API key is set in the environment so the backend can still authenticate
    for writes without the operator duplicating the key into ``.env``.
    """
    cfg_path = path or _TILED_CONFIG_PATH
    try:
        import yaml

        with open(cfg_path) as f:
            doc = yaml.safe_load(f) or {}
        key = ((doc.get("authentication") or {}).get("single_user_api_key")) or None
        if not key:
            return None
        # config.yml may hold a ${TILED_API_KEY} placeholder (Tiled expands it the
        # same way at load time); expand from the env and reject an unset placeholder.
        key = os.path.expandvars(str(key)).strip()
        if not key or "$" in key:
            return None
        return key
    except Exception as exc:
        logger.debug("tiled_config: could not read single_user_api_key from %s: %s", cfg_path, exc)
        return None


def local_uri() -> str:
    """Local Tiled base URI.

    ``TILED_URI`` (set by ``start_all.sh`` to whatever port Tiled actually bound —
    it may fall back off the default 8010 if that port is busy) takes precedence;
    otherwise the built-in default.
    """
    return _stripped("TILED_URI") or _LOCAL_URI


def _local_api_key() -> str | None:
    return (
        _stripped("TILED_LOCAL_API_KEY")
        or _stripped("TILED_API_KEY")
        or single_user_api_key_from_tiled_config()
        or None
    )


def get_tiled_servers() -> dict[str, dict[str, str | None]]:
    """Return mapping of ``{name: {"uri": str, "api_key": str | None}}``.

    Discovery order:

    1. Any ``TILED_SERVER_{N}_*`` trio in env (contiguous, starting at 1).
    2. A default "Local Data (port 8010)" entry (always added if missing).
    """
    servers: dict[str, dict[str, str | None]] = {}

    i = 1
    while True:
        name = _stripped(f"TILED_SERVER_{i}_NAME")
        uri = _stripped(f"TILED_SERVER_{i}_URI").rstrip("/")
        if not name or not uri:
            break
        servers[name] = {
            "uri": uri,
            "api_key": _stripped(f"TILED_SERVER_{i}_API_KEY") or None,
        }
        i += 1

    lu = local_uri()
    if not any(cfg["uri"] == lu for cfg in servers.values()):
        port = lu.rsplit(":", 1)[-1]
        name = f"Local Data (port {port})" if port.isdigit() else "Local Data"
        servers.setdefault(name, {"uri": lu, "api_key": _local_api_key()})

    return servers


def get_tiled_base() -> str:
    """Return the default Tiled base URI."""
    return local_uri()


def get_tiled_api_key() -> str | None:
    """Return the default API key (local override takes precedence)."""
    return _local_api_key()
