"""Tests for Tiled key hardening (empty-key clearing + config-key fallback)."""

from __future__ import annotations

import importlib
import os
from unittest import mock

import tiled_config


def test_clear_empty_tiled_api_key_env_removes_blank_vars() -> None:
    """Blank/whitespace key env vars are deleted; real ones are kept."""
    with mock.patch.dict(
        os.environ,
        {"TILED_API_KEY": "  ", "TILED_LOCAL_API_KEY": ""},
        clear=False,
    ):
        tiled_config.clear_empty_tiled_api_key_env()
        assert "TILED_API_KEY" not in os.environ
        assert "TILED_LOCAL_API_KEY" not in os.environ

    with mock.patch.dict(os.environ, {"TILED_API_KEY": "realkey"}, clear=False):
        tiled_config.clear_empty_tiled_api_key_env()
        assert os.environ["TILED_API_KEY"] == "realkey"


def test_single_user_api_key_from_tiled_config(tmp_path) -> None:
    """Reads authentication.single_user_api_key from a config.yml."""
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        "authentication:\n"
        "  allow_anonymous_access: true\n"
        '  single_user_api_key: "abc123"\n'
    )
    assert tiled_config.single_user_api_key_from_tiled_config(cfg) == "abc123"

    # Missing key / missing file → None (not an error).
    empty = tmp_path / "empty.yml"
    empty.write_text("allow_origins: []\n")
    assert tiled_config.single_user_api_key_from_tiled_config(empty) is None
    assert tiled_config.single_user_api_key_from_tiled_config(tmp_path / "nope.yml") is None


def test_config_key_env_interpolation(tmp_path) -> None:
    """A ${TILED_API_KEY} placeholder in config.yml expands from the env; unset → None."""
    cfg = tmp_path / "config.yml"
    cfg.write_text('authentication:\n  single_user_api_key: "${TILED_API_KEY}"\n')

    with mock.patch.dict(os.environ, {"TILED_API_KEY": "envexpanded"}, clear=False):
        assert tiled_config.single_user_api_key_from_tiled_config(cfg) == "envexpanded"

    # Unset var → placeholder stays literal → treated as "no key" (not the literal).
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TILED_API_KEY", None)
        assert tiled_config.single_user_api_key_from_tiled_config(cfg) is None


def test_local_api_key_falls_back_to_config(tmp_path) -> None:
    """_local_api_key prefers env, else falls back to the config single-user key."""
    cfg = tmp_path / "config.yml"
    cfg.write_text('authentication:\n  single_user_api_key: "cfgkey"\n')

    # Env blank → falls back to config key.
    with mock.patch.dict(os.environ, {}, clear=False), \
         mock.patch.object(tiled_config, "_TILED_CONFIG_PATH", cfg):
        os.environ.pop("TILED_API_KEY", None)
        os.environ.pop("TILED_LOCAL_API_KEY", None)
        assert tiled_config._local_api_key() == "cfgkey"

    # Env set → env wins over config.
    with mock.patch.dict(os.environ, {"TILED_API_KEY": "envkey"}, clear=False), \
         mock.patch.object(tiled_config, "_TILED_CONFIG_PATH", cfg):
        assert tiled_config._local_api_key() == "envkey"


def test_module_reload_clears_blank_key() -> None:
    """Importing the module clears a blank TILED_API_KEY from the env."""
    with mock.patch.dict(os.environ, {"TILED_API_KEY": ""}, clear=False):
        importlib.reload(tiled_config)
        assert "TILED_API_KEY" not in os.environ
