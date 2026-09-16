"""Catalog project/session persistence."""

from __future__ import annotations

import os

import pytest

from ipred.catalog import Catalog, project_id_for


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def test_project_id_stable() -> None:
    a = project_id_for(kind="local", source="a.tif", root="/data")
    b = project_id_for(kind="local", source="a.tif", root="/data")
    c = project_id_for(kind="local", source="b.tif", root="/data")
    assert a == b
    assert a != c


def test_open_session_creates_project(catalog: Catalog) -> None:
    s1 = catalog.open_session(kind="local", source="img.tif", root=str(os.environ["LOCAL_DATA_ROOT"]))
    s2 = catalog.open_session(kind="local", source="img.tif", root=str(os.environ["LOCAL_DATA_ROOT"]))
    assert s1.project_id == s2.project_id
    assert s1.session_id != s2.session_id
    got = catalog.get_session(s1.session_id)
    assert got is not None
    assert got.current_feature_id is None


def test_set_session_currents(catalog: Catalog) -> None:
    s = catalog.open_session(kind="local", source="x.png")
    updated = catalog.set_session_currents(s.session_id, feature_id="abc")
    assert updated.current_feature_id == "abc"
