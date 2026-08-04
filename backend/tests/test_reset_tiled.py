"""HTTP-boundary tests for POST /api/browse/reset-tiled and its preview route.

A deliberately blunt "empty the catalog": deletes every TOP-LEVEL Tiled
container so it doesn't need to know anything about ingest's own layout
(browse/<dataset>/<sample>/...) — whatever exists at the root is what gets
wiped. Never touches saved training runs (entirely outside Tiled). By default
it ALSO clears local annotation drafts/versions for this server — a draft is
keyed by path, not by what's actually in Tiled, so leaving one behind after
wiping the data it describes means it silently reattaches to whatever
unrelated data lands at that same path next.

Every test isolates LOCAL_DATA_ROOT (via tmp_path + reload) so nothing here
can ever read or touch the real developer machine's ~/data/.drafts.
"""

from __future__ import annotations

import importlib
import os

import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
from annotation_server import app

FAKE_SERVER = "http://fake"


class _FakeRootClient:
    """Minimal stand-in for the Tiled root client this route walks."""

    def __init__(self, children: dict[str, object]) -> None:
        self._children = dict(children)
        self.deleted: list[str] = []

    def keys(self):
        return list(self._children)

    def delete_contents(self, key, recursive: bool = False, external_only: bool = True) -> None:
        assert recursive is True, "a shallow delete would leave nested samples behind"
        assert external_only is False, "internally-managed arrays need external_only=False"
        if key not in self._children:
            raise KeyError(key)
        self.deleted.append(key)
        del self._children[key]


@pytest.fixture
def isolated_drafts(tmp_path, monkeypatch):
    """Point drafts.py at a throwaway directory for the duration of the test."""
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    import drafts

    importlib.reload(drafts)
    yield drafts
    # Leave the module in a sane state for whatever runs next in this process.
    monkeypatch.delenv("LOCAL_DATA_ROOT", raising=False)
    importlib.reload(drafts)


@pytest.fixture
def fake_tiled(isolated_drafts, monkeypatch):
    annotation_server._items_cache.set(("sentinel",), {"still": "here"})
    annotation_server._column_cache.set(("sentinel",), {"still": "here"})
    annotation_server._field_mapping_cache.set(("sentinel",), {"still": "here"})
    monkeypatch.setattr(annotation_server, "_require_tiled_server", lambda uri: FAKE_SERVER)

    def _install(children: dict[str, object]) -> _FakeRootClient:
        client = _FakeRootClient(children)
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: client)
        return client

    yield _install
    annotation_server._items_cache.clear()
    annotation_server._column_cache.clear()
    annotation_server._field_mapping_cache.clear()


async def _reset(**params) -> tuple[int, dict]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/browse/reset-tiled", params=params)
    return response.status_code, response.json()


async def _preview() -> tuple[int, dict]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/browse/reset-tiled/preview")
    return response.status_code, response.json()


@pytest.mark.asyncio
async def test_deletes_every_top_level_key(fake_tiled) -> None:
    client = fake_tiled({"browse": object(), "other_root_container": object()})

    status, body = await _reset()

    assert status == 200
    assert set(body["deleted_keys"]) == {"browse", "other_root_container"}
    assert client.deleted == body["deleted_keys"]
    assert body["errors"] == []


@pytest.mark.asyncio
async def test_an_already_empty_catalog_is_success_not_an_error(fake_tiled) -> None:
    fake_tiled({})

    status, body = await _reset()

    assert status == 200
    assert body["deleted_keys"] == []
    assert body["errors"] == []


@pytest.mark.asyncio
async def test_one_failing_key_does_not_abort_the_rest(fake_tiled, monkeypatch) -> None:
    """One bad container must not leave the others un-reset — same
    per-item-isolation principle as ingest's own per-file error handling."""
    client = fake_tiled({"good_a": object(), "bad": object(), "good_b": object()})
    real_delete = client.delete_contents

    def _flaky_delete(key, **kwargs):
        if key == "bad":
            raise RuntimeError("500: /bad http://127.0.0.1:8010/api/v1/metadata/bad")
        return real_delete(key, **kwargs)

    monkeypatch.setattr(client, "delete_contents", _flaky_delete)

    status, body = await _reset()

    assert status == 200
    assert set(body["deleted_keys"]) == {"good_a", "good_b"}
    assert len(body["errors"]) == 1
    assert body["errors"][0]["key"] == "bad"


@pytest.mark.asyncio
async def test_clears_browse_caches_so_stale_listings_are_not_served(fake_tiled) -> None:
    fake_tiled({"browse": object()})
    assert annotation_server._items_cache.get(("sentinel",)) is not None

    await _reset()

    assert annotation_server._items_cache.get(("sentinel",)) is None
    assert annotation_server._column_cache.get(("sentinel",)) is None
    assert annotation_server._field_mapping_cache.get(("sentinel",)) is None


@pytest.mark.asyncio
async def test_an_unconfigured_server_uri_is_rejected_before_touching_anything(fake_tiled, monkeypatch) -> None:
    def _reject(uri):
        raise annotation_server.HTTPException(403, "Tiled server is not configured")

    monkeypatch.setattr(annotation_server, "_require_tiled_server", _reject)

    status, body = await _reset()

    assert status == 403


def test_openapi_reset_route_is_registered() -> None:
    schema = app.openapi()
    assert "/api/browse/reset-tiled" in schema["paths"]
    assert "post" in schema["paths"]["/api/browse/reset-tiled"]
    assert "/api/browse/reset-tiled/preview" in schema["paths"]


# ---------------------------------------------------------------------------
# Clearing local annotation drafts (the actual regression this closes: a
# draft for a path re-ingest reuses silently reattaches its old annotations
# — full overlay and all — onto whatever new, unrelated data lands there)
# ---------------------------------------------------------------------------


def _make_draft(drafts_mod, source_key: str, n_versions: int = 0) -> None:
    drafts_mod.save_draft(source_key, {"classes": [], "slices": {"0": []}})
    for _ in range(n_versions):
        drafts_mod.save_version(source_key, {"classes": [], "slices": {}, "split_by_slice": {}, "negative_slices": []})


@pytest.mark.asyncio
async def test_clear_drafts_true_by_default_removes_matching_drafts(fake_tiled, isolated_drafts) -> None:
    fake_tiled({"browse": object()})
    source_key = f"tiled:{FAKE_SERVER}:browse/dataset"
    _make_draft(isolated_drafts, source_key, n_versions=2)
    assert isolated_drafts.load_draft(source_key) is not None

    status, body = await _reset()

    assert status == 200
    assert body["drafts_deleted"] == 1
    assert isolated_drafts.load_draft(source_key) is None
    assert isolated_drafts.list_versions(source_key) == []


@pytest.mark.asyncio
async def test_clear_drafts_false_leaves_drafts_alone(fake_tiled, isolated_drafts) -> None:
    fake_tiled({"browse": object()})
    source_key = f"tiled:{FAKE_SERVER}:browse/dataset"
    _make_draft(isolated_drafts, source_key)

    status, body = await _reset(clear_drafts="false")

    assert status == 200
    assert body["drafts_deleted"] == 0
    assert isolated_drafts.load_draft(source_key) is not None


@pytest.mark.asyncio
async def test_only_drafts_for_this_server_are_cleared(fake_tiled, isolated_drafts) -> None:
    """A draft for a DIFFERENT configured server, or a local (non-Tiled)
    source, must survive — this server's reset has nothing to do with them."""
    fake_tiled({"browse": object()})
    this_server = f"tiled:{FAKE_SERVER}:browse/dataset"
    other_server = "tiled:http://a-different-server:9999:browse/dataset"
    local_source = "local:some/relative/path.tif"
    for key in (this_server, other_server, local_source):
        _make_draft(isolated_drafts, key)

    status, body = await _reset()

    assert status == 200
    assert body["drafts_deleted"] == 1
    assert isolated_drafts.load_draft(this_server) is None
    assert isolated_drafts.load_draft(other_server) is not None
    assert isolated_drafts.load_draft(local_source) is not None


@pytest.mark.asyncio
async def test_annotation_guides_are_never_touched(fake_tiled, isolated_drafts) -> None:
    """Guides (Reference tab content) live in the same directory as drafts
    but are a completely different concept — a reset must never delete one."""
    import guides as guides_mod

    importlib.reload(guides_mod)

    fake_tiled({"browse": object()})
    source_key = f"tiled:{FAKE_SERVER}:browse/dataset"
    _make_draft(isolated_drafts, source_key)
    guides_mod.save_guide(source_key, {"classes": []})

    await _reset()

    assert guides_mod.load_guide(source_key) is not None


@pytest.mark.asyncio
async def test_preview_reports_draft_and_version_counts_before_anything_is_deleted(
    fake_tiled, isolated_drafts
) -> None:
    fake_tiled({"browse": object()})
    _make_draft(isolated_drafts, f"tiled:{FAKE_SERVER}:browse/a", n_versions=2)
    _make_draft(isolated_drafts, f"tiled:{FAKE_SERVER}:browse/b", n_versions=3)
    _make_draft(isolated_drafts, "tiled:http://other:1234:browse/c", n_versions=5)

    status, body = await _preview()

    assert status == 200
    assert body["draft_count"] == 2
    assert body["version_count"] == 5
    # Nothing was actually touched by a GET.
    assert isolated_drafts.load_draft(f"tiled:{FAKE_SERVER}:browse/a") is not None
