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

When env keys are blank, the local server falls back to
``authentication.single_user_api_key`` in ``tiled/config.yml``. Tiled anonymous
access is read-only; that key is required for write scopes (ingest, create node).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_LOCAL_URI = "http://127.0.0.1:8010"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TILED_CONFIG_PATH = _REPO_ROOT / "tiled" / "config.yml"
_logger = logging.getLogger("tiled_config")

# Env var names the tiled Python client (and this backend) may treat as API keys.
# An *empty* value is worse than unset: tiled builds ``Authorization: Apikey ``
# which httpx rejects as an illegal header value.
_API_KEY_ENV_NAMES: tuple[str, ...] = (
    "TILED_API_KEY",
    "TILED_LOCAL_API_KEY",
)


def clear_empty_tiled_api_key_env() -> None:
    """Remove blank Tiled API key env vars so clients do not send ``Apikey ``.

    ``backend/.env.example`` ships ``TILED_API_KEY=`` for documentation. When that
    file is sourced (`set -a; source .env`) or loaded via dotenv, the empty string
    stays in ``os.environ``. Tiled's client then does ``os.getenv("TILED_API_KEY")``
    and sets an illegal Authorization header.
    """
    for name in _API_KEY_ENV_NAMES:
        if name in os.environ and not os.environ[name].strip():
            del os.environ[name]


clear_empty_tiled_api_key_env()


def _stripped(name: str) -> str:
    return (os.getenv(name) or "").strip()


def single_user_api_key_from_tiled_config(
    config_path: Path | None = None,
) -> str | None:
    """Return ``authentication.single_user_api_key`` from the local Tiled config.

    Args:
        config_path: Optional override path (defaults to repo ``tiled/config.yml``).

    Returns:
        The key string, or ``None`` if missing/unreadable.
    """
    path = config_path or _TILED_CONFIG_PATH
    try:
        import yaml

        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        key = (raw.get("authentication") or {}).get("single_user_api_key")
        if isinstance(key, str) and key.strip():
            return key.strip()
    except FileNotFoundError:
        _logger.debug("tiled config not found at %s", path)
    except Exception as exc:  # noqa: BLE001 — best-effort fallback
        _logger.warning("could not read single_user_api_key from %s: %s", path, exc)
    return None


def _local_api_key() -> str | None:
    return (
        _stripped("TILED_LOCAL_API_KEY")
        or _stripped("TILED_API_KEY")
        or single_user_api_key_from_tiled_config()
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
