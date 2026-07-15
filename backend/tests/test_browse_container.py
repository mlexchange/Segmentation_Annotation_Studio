"""Tests for browse-container auto-discovery.

Discovery must prefer the fullest sample collection (e.g. ``petioles/`` with
dozens of children) over a thin ``browse/`` leftover with a single ingest.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from tiled_clients import get_browse_container


class _FakeNode:
    """Minimal mapping that supports ``node[key]`` and ``len(node)``."""

    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children = children or {}

    def __getitem__(self, key: str) -> Any:
        return self._children[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def keys(self) -> Any:
        return self._children.keys()


def test_prefers_largest_top_level_over_small_browse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A populated ``petioles`` container must beat a 1-item ``browse``."""
    monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
    petioles = _FakeNode({f"s{i}": object() for i in range(20)})
    browse = _FakeNode({"only_one": object()})
    root = _FakeNode({"browse": browse, "petioles": petioles})

    node, prefix = get_browse_container(root)
    assert prefix == "petioles"
    assert node is petioles
    assert len(node) == 20


def test_env_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TILED_BROWSE_PATH`` overrides size-based discovery."""
    monkeypatch.setenv("TILED_BROWSE_PATH", "browse")
    petioles = _FakeNode({f"s{i}": object() for i in range(20)})
    browse = _FakeNode({"a": object(), "b": object()})
    root = _FakeNode({"browse": browse, "petioles": petioles})

    node, prefix = get_browse_container(root)
    assert prefix == "browse"
    assert node is browse


def test_falls_back_to_root_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty catalog → browse at the client root."""
    monkeypatch.delenv("TILED_BROWSE_PATH", raising=False)
    root = _FakeNode({})
    node, prefix = get_browse_container(root)
    assert prefix == ""
    assert node is root
