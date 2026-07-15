"""Tests for Tiled env config — empty API keys and local single-user key fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _restore_tiled_api_key_env() -> None:
    """Preserve caller env around mutations of TILED_* API key vars."""
    names = ("TILED_API_KEY", "TILED_LOCAL_API_KEY")
    prior = {name: os.environ.get(name) for name in names}
    yield
    for name, value in prior.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_empty_tiled_api_key_is_cleared_from_environ() -> None:
    """Blank TILED_API_KEY must be unset so tiled client never sends ``Apikey ``."""
    os.environ["TILED_API_KEY"] = ""
    os.environ["TILED_LOCAL_API_KEY"] = "   "

    import tiled_config

    tiled_config.clear_empty_tiled_api_key_env()

    assert "TILED_API_KEY" not in os.environ
    assert "TILED_LOCAL_API_KEY" not in os.environ


def test_nonempty_tiled_api_key_is_preserved() -> None:
    """A real API key must survive the empty-key cleanup."""
    os.environ["TILED_API_KEY"] = "real-key-value"

    import tiled_config

    tiled_config.clear_empty_tiled_api_key_env()

    assert os.environ.get("TILED_API_KEY") == "real-key-value"
    assert tiled_config.get_tiled_api_key() == "real-key-value"


def test_local_api_key_falls_back_to_tiled_config_yml() -> None:
    """When env keys are blank, use single_user_api_key from tiled/config.yml.

    Tiled anonymous access is read-only; writes need this key (create:node, etc.).
    """
    os.environ.pop("TILED_API_KEY", None)
    os.environ.pop("TILED_LOCAL_API_KEY", None)

    import tiled_config

    config_path = Path(tiled_config.__file__).resolve().parent.parent / "tiled" / "config.yml"
    assert config_path.is_file(), f"missing {config_path}"
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    expected = (raw.get("authentication") or {}).get("single_user_api_key")
    assert expected, "tiled/config.yml must define authentication.single_user_api_key for local writes"

    assert tiled_config.get_tiled_api_key() == expected
    local = tiled_config.get_tiled_servers()["Local Data (port 8010)"]
    assert local["api_key"] == expected
