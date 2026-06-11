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

import os

from dotenv import load_dotenv

load_dotenv()

_LOCAL_URI = "http://127.0.0.1:8010"


def _stripped(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _local_api_key() -> str | None:
    return _stripped("TILED_LOCAL_API_KEY") or _stripped("TILED_API_KEY") or None


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

    if not any(cfg["uri"] == _LOCAL_URI for cfg in servers.values()):
        servers.setdefault(
            "Local Data (port 8010)",
            {"uri": _LOCAL_URI, "api_key": _local_api_key()},
        )

    return servers


def get_tiled_base() -> str:
    """Return the default Tiled base URI."""
    return _stripped("TILED_URI") or _LOCAL_URI


def get_tiled_api_key() -> str | None:
    """Return the default API key (local override takes precedence)."""
    return _local_api_key()
