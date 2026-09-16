"""Tests for locating a dataset's renderable 3-D volume.

This is the module that answers "which node do I point the viewer at?". Getting
it wrong is not a subtle failure — the viewer reports
``openOmeZarr: missing multiscales in root .zattrs``, which tells the user
nothing about the real answer ("look at the sidecar" / "none has been built").

A fake catalog stands in for Tiled: the logic under test is metadata inspection
and path walking, both of which a real server would only slow down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import volume_nodes  # noqa: E402
from tiff_stack_source import VOLUME_SUFFIX  # noqa: E402

MULTISCALES = {"attributes": {"multiscales": [{"datasets": [{"path": "scale0"}]}]}}


class FakeNode:
    """A catalog node: metadata plus children, indexable like a Tiled container."""

    def __init__(self, metadata=None, children=None):
        self.metadata = metadata or {}
        self._children = children or {}

    def __getitem__(self, key):
        return self._children[key]


@pytest.fixture
def catalog(monkeypatch):
    """A catalog with one of each shape the resolver has to tell apart."""
    root = FakeNode(children={
        "browse": FakeNode(children={
            # A registered Zarr volume: multiscale itself, with level children.
            "zarr_vol": FakeNode(
                metadata=dict(MULTISCALES, zarr_path="/data/zarr_vol.zarr"),
                children={"scale0": FakeNode(children={"image": FakeNode()})},
            ),
            # A TIFF stack: per-slice 2-D arrays, no multiscales anywhere.
            "tiff_stack": FakeNode(metadata={"n_images": 604}),
            # ...and the sidecar built for it.
            f"tiff_stack{VOLUME_SUFFIX}": FakeNode(
                metadata=dict(MULTISCALES, tiff_dir="/data/tiff_stack"),
            ),
            # A stack nobody has built a volume for.
            "lonely_stack": FakeNode(metadata={"n_images": 12}),
        })
    })
    monkeypatch.setattr(volume_nodes, "get_tiled_client", lambda *a, **k: root)
    monkeypatch.setattr(volume_nodes, "api_key_for_uri", lambda *a, **k: None)
    return root


class TestHasMultiscales:
    def test_true_where_tiled_will_serve_it(self):
        # Tiled's .zattrs route returns metadata["attributes"] verbatim, so that
        # is the only place the viewer can see multiscales.
        assert volume_nodes.has_multiscales(FakeNode(metadata=MULTISCALES))

    def test_false_when_nested_anywhere_else(self):
        # A common near-miss: right key, wrong level. It would never reach .zattrs.
        assert not volume_nodes.has_multiscales(
            FakeNode(metadata={"multiscales": [{"datasets": []}]})
        )

    def test_false_for_empty_or_missing(self):
        assert not volume_nodes.has_multiscales(FakeNode(metadata={}))
        assert not volume_nodes.has_multiscales(FakeNode(metadata={"attributes": {}}))
        assert not volume_nodes.has_multiscales(
            FakeNode(metadata={"attributes": {"multiscales": []}})
        )


class TestResolveVolume:
    def test_open_node_is_already_a_volume(self, catalog):
        result = volume_nodes.resolve_volume(None, "browse/zarr_vol")
        assert result["mode"] == "self"
        assert result["path"] == "browse/zarr_vol"

    def test_walks_up_from_a_pyramid_level(self, catalog):
        # Opening `<volume>/scale0/image` is ordinary — that array has no
        # multiscales of its own, but its grandparent does.
        result = volume_nodes.resolve_volume(None, "browse/zarr_vol/scale0/image")
        assert result["mode"] == "ancestor"
        assert result["path"] == "browse/zarr_vol"

    def test_finds_the_sidecar_for_a_tiff_stack(self, catalog):
        # The bug this module exists for: the open node is a container of 2-D
        # slices, and the volume is its __volume sibling.
        result = volume_nodes.resolve_volume(None, "browse/tiff_stack")
        assert result["mode"] == "sidecar"
        assert result["path"] == f"browse/tiff_stack{VOLUME_SUFFIX}"

    def test_reports_none_with_an_actionable_message(self, catalog):
        result = volume_nodes.resolve_volume(None, "browse/lonely_stack")
        assert result["mode"] == "none"
        assert result["path"] is None
        assert "built" in result["message"]

    def test_reports_none_for_a_missing_node(self, catalog):
        result = volume_nodes.resolve_volume(None, "browse/does_not_exist")
        assert result["mode"] == "none"

    def test_handles_an_empty_source(self, catalog):
        result = volume_nodes.resolve_volume(None, "")
        assert result["mode"] == "none"
        assert result["message"]

    def test_tolerates_surrounding_slashes(self, catalog):
        assert volume_nodes.resolve_volume(None, "/browse/zarr_vol/")["mode"] == "self"

    def test_never_ascends_past_the_catalog_root(self, catalog):
        # Walking up must not wander off the top and resolve some unrelated node.
        assert volume_nodes.resolve_volume(None, "browse")["mode"] == "none"


class TestSourceDir:
    def test_surfaces_the_registered_tiff_directory(self, catalog):
        result = volume_nodes.resolve_volume(None, "browse/tiff_stack")
        assert result["source_dir"] == "/data/tiff_stack"

    def test_surfaces_the_registered_zarr_path(self, catalog):
        result = volume_nodes.resolve_volume(None, "browse/zarr_vol")
        assert result["source_dir"] == "/data/zarr_vol.zarr"

    def test_is_none_when_nothing_was_recorded(self):
        assert volume_nodes.source_dir_of(FakeNode(metadata={})) is None
