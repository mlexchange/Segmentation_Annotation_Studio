"""HTTP-level tests for annotation_server.py's local-filesystem, connection
summary, Tiled listing, and image meta/slice routes. Continues the pattern
established in test_annotation_server_routes.py/test_annotation_server_browse_routes.py:
httpx.AsyncClient + ASGITransport, monkeypatched I/O boundaries
(local_fs/get_tiled_client/arrays_mod/images_mod), no real Tiled server or
filesystem beyond what a test itself creates."""
from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
import arrays as arrays_mod
import images as images_mod
import local_fs

app = annotation_server.app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class FakeNode:
    def __init__(self, children=None, is_container=True):
        self._children = children or {}
        self.structure_family = "container" if is_container else "array"

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def keys(self):
        return list(self._children.keys())

    def __len__(self):
        return len(self._children)


# ---------------------------------------------------------------------------
# /api/local/list, /api/local/samples
# ---------------------------------------------------------------------------

class TestLocalList:
    @pytest.mark.asyncio
    async def test_lists_directory_entries(self, client, monkeypatch):
        monkeypatch.setattr(local_fs, "list_dir", lambda rel, root: [{"name": "a", "is_dir": True}])
        response = await client.get("/api/local/list", params={"rel": ""})
        assert response.status_code == 200
        assert response.json() == [{"name": "a", "is_dir": True}]

    @pytest.mark.asyncio
    async def test_passes_rel_and_root_through(self, client, monkeypatch):
        calls = []
        monkeypatch.setattr(local_fs, "list_dir", lambda rel, root: calls.append((rel, root)) or [])
        await client.get("/api/local/list", params={"rel": "sub", "root": "/data"})
        assert calls == [("sub", "/data")]


class TestLocalSamples:
    @pytest.mark.asyncio
    async def test_returns_items_and_total(self, client, monkeypatch):
        monkeypatch.setattr(
            local_fs, "list_image_files",
            lambda rel, root: [{"name": "a.tif", "path": "a.tif"}, {"name": "b.tif", "path": "b.tif"}],
        )
        response = await client.get("/api/local/samples", params={"rel": "folder"})
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    @pytest.mark.asyncio
    async def test_requires_rel_query_param(self, client):
        response = await client.get("/api/local/samples")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/connect/summary
# ---------------------------------------------------------------------------

class TestConnectSummaryLocal:
    @pytest.mark.asyncio
    async def test_counts_local_files(self, client, monkeypatch):
        monkeypatch.setattr(local_fs, "count_image_files", lambda rel, root: 42)
        response = await client.get(
            "/api/connect/summary", params={"kind": "local", "rel": "sub", "root": "/data"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sample_count"] == 42
        assert body["label"] == "/data/sub"
        assert body["kind"] == "local"

    @pytest.mark.asyncio
    async def test_label_falls_back_to_default_when_no_root(self, client, monkeypatch):
        monkeypatch.setattr(local_fs, "count_image_files", lambda rel, root: 0)
        response = await client.get("/api/connect/summary", params={"kind": "local"})
        assert response.json()["label"] == "Local Data Root"


class TestConnectSummaryTiled:
    @pytest.mark.asyncio
    async def test_counts_container_via_len(self, client, monkeypatch):
        root = FakeNode({"a": 1, "b": 2, "c": 3})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        monkeypatch.setattr(annotation_server, "get_browse_container_for", lambda client, path: (root, "browse"))
        monkeypatch.setattr(annotation_server, "get_tiled_servers", lambda: {})
        response = await client.get("/api/connect/summary", params={"kind": "tiled", "server_uri": "http://x"})
        assert response.status_code == 200
        body = response.json()
        assert body["sample_count"] == 3
        assert body["kind"] == "tiled"

    @pytest.mark.asyncio
    async def test_label_uses_configured_server_name(self, client, monkeypatch):
        root = FakeNode({})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        monkeypatch.setattr(annotation_server, "get_browse_container_for", lambda client, path: (root, "browse"))
        monkeypatch.setattr(
            annotation_server, "get_tiled_servers",
            lambda: {"local": {"uri": "http://x:1", "name": "My Server"}},
        )
        response = await client.get("/api/connect/summary", params={"kind": "tiled", "server_uri": "http://x:1"})
        assert response.json()["label"] == "My Server"

    @pytest.mark.asyncio
    async def test_container_path_appended_to_label(self, client, monkeypatch):
        root = FakeNode({})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        monkeypatch.setattr(annotation_server, "get_browse_container_for", lambda client, path: (root, path or ""))
        monkeypatch.setattr(annotation_server, "get_tiled_servers", lambda: {})
        response = await client.get(
            "/api/connect/summary",
            params={"kind": "tiled", "server_uri": "http://x", "container_path": "browse/sample"},
        )
        assert "browse/sample" in response.json()["label"]

    @pytest.mark.asyncio
    async def test_count_failure_falls_back_to_zero(self, client, monkeypatch):
        def boom(uri):
            raise RuntimeError("down")

        monkeypatch.setattr(annotation_server, "get_tiled_client", boom)
        monkeypatch.setattr(annotation_server, "get_tiled_servers", lambda: {})
        response = await client.get("/api/connect/summary", params={"kind": "tiled", "server_uri": "http://x"})
        assert response.status_code == 200
        assert response.json()["sample_count"] == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_search_when_len_unsupported(self, client, monkeypatch):
        class NoLenNode(FakeNode):
            def __len__(self):
                raise TypeError("no len")

        root = NoLenNode({"a": 1})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        monkeypatch.setattr(annotation_server, "get_browse_container_for", lambda client, path: (root, "browse"))
        monkeypatch.setattr(annotation_server, "get_tiled_servers", lambda: {})
        monkeypatch.setattr(annotation_server, "tiled_search_items", lambda container, filters, limit: {"total": 7})
        response = await client.get("/api/connect/summary", params={"kind": "tiled", "server_uri": "http://x"})
        assert response.json()["sample_count"] == 7


class TestConnectSummaryUnknownKind:
    @pytest.mark.asyncio
    async def test_unknown_kind_is_400(self, client):
        response = await client.get("/api/connect/summary", params={"kind": "weird"})
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# /api/tiled/list
# ---------------------------------------------------------------------------

class TestTiledList:
    @pytest.mark.asyncio
    async def test_lists_children_sorted_containers_first(self, client, monkeypatch):
        root = FakeNode({
            "z_array": FakeNode(is_container=False),
            "a_container": FakeNode({}),
        })
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        response = await client.get("/api/tiled/list", params={"path": ""})
        assert response.status_code == 200
        body = response.json()
        assert body[0]["name"] == "a_container"
        assert body[0]["is_dir"] is True
        assert body[1]["name"] == "z_array"
        assert body[1]["is_array"] is True

    @pytest.mark.asyncio
    async def test_missing_path_segment_is_404(self, client, monkeypatch):
        root = FakeNode({})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        response = await client.get("/api/tiled/list", params={"path": "nope"})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_leaf_array_node_is_400(self, client, monkeypatch):
        class LeafNoKeys(FakeNode):
            def keys(self):
                raise RuntimeError("no children")

        root = FakeNode({"leaf": LeafNoKeys(is_container=False)})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        response = await client.get("/api/tiled/list", params={"path": "leaf"})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_nested_path_joins_correctly(self, client, monkeypatch):
        root = FakeNode({"a": FakeNode({"b": FakeNode(is_container=False)})})
        monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: root)
        response = await client.get("/api/tiled/list", params={"path": "a"})
        assert response.json()[0]["path"] == "a/b"

    @pytest.mark.asyncio
    async def test_unexpected_error_is_500(self, client, monkeypatch):
        def boom(uri):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(annotation_server, "get_tiled_client", boom)
        response = await client.get("/api/tiled/list", params={"path": ""})
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /api/image/meta, /api/image/slice
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_image_source(monkeypatch: pytest.MonkeyPatch):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri, root: "node")
    monkeypatch.setattr(arrays_mod, "pyramid_info", lambda source, kind, server_uri, root: None)
    monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node, pyramid: {
        "n_slices": 1, "height": 10, "width": 10, "dtype": "float32", "is_rgb": False, "shape_kind": "HW",
    })
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: arr)
    monkeypatch.setattr(arrays_mod, "node_keywords", lambda node: ["tag1"])
    return arr


class TestImageMeta:
    @pytest.mark.asyncio
    async def test_returns_shape_and_value_range(self, client, fake_image_source):
        response = await client.get(
            "/api/image/meta", params={"source": "local:foo.tif", "kind": "local"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_slices"] == 1
        assert body["height"] == 10
        assert body["width"] == 10
        assert body["value_range"] == [0.0, 99.0]
        assert body["keywords"] == ["tag1"]

    @pytest.mark.asyncio
    async def test_resolve_failure_is_500(self, client, monkeypatch):
        monkeypatch.setattr(
            arrays_mod, "resolve_array",
            lambda source, kind, server_uri, root: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        response = await client.get(
            "/api/image/meta", params={"source": "local:foo.tif", "kind": "local"},
        )
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_http_exception_is_passed_through(self, monkeypatch, client):
        from fastapi import HTTPException

        def raise_404(source, kind, server_uri, root):
            raise HTTPException(404, "not found")

        monkeypatch.setattr(arrays_mod, "resolve_array", raise_404)
        response = await client.get(
            "/api/image/meta", params={"source": "local:foo.tif", "kind": "local"},
        )
        assert response.status_code == 404


class TestImageSlice:
    @pytest.mark.asyncio
    async def test_renders_a_png(self, client, fake_image_source, monkeypatch):
        monkeypatch.setattr(images_mod, "_sample_global_stats", lambda node, meta: (0.0, 99.0))
        response = await client.get(
            "/api/image/slice", params={"source": "local:foo.tif", "kind": "local"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_unknown_denoise_method_is_400(self, client, fake_image_source):
        response = await client.get(
            "/api/image/slice",
            params={"source": "local:foo.tif", "kind": "local", "denoise_method": "not_a_method"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_denoised_result_is_cached_across_requests(self, client, fake_image_source, monkeypatch):
        calls = []
        real_denoised = annotation_server._denoised_slice

        def spy_denoised(node, meta, idx, method, strength, crop):
            calls.append(1)
            return real_denoised(node, meta, idx, method, strength, crop)

        monkeypatch.setattr(annotation_server, "_denoised_slice", spy_denoised)
        monkeypatch.setattr(images_mod, "_sample_global_stats", lambda node, meta: (0.0, 99.0))
        params = {"source": "local:foo.tif", "kind": "local", "denoise_method": "median"}
        r1 = await client.get("/api/image/slice", params=params)
        r2 = await client.get("/api/image/slice", params=params)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.content == r2.content
        assert len(calls) == 1  # second request served from cache

    @pytest.mark.asyncio
    async def test_render_failure_is_500(self, client, fake_image_source, monkeypatch):
        monkeypatch.setattr(
            images_mod, "render_slice",
            lambda sl, opts, gr: (_ for _ in ()).throw(RuntimeError("render broke")),
        )
        monkeypatch.setattr(images_mod, "_sample_global_stats", lambda node, meta: (0.0, 99.0))
        response = await client.get(
            "/api/image/slice", params={"source": "local:foo.tif", "kind": "local"},
        )
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_slice_norm_skips_global_stats(self, client, fake_image_source, monkeypatch):
        calls = []
        monkeypatch.setattr(
            images_mod, "_sample_global_stats",
            lambda node, meta: calls.append(1) or (0.0, 99.0),
        )
        response = await client.get(
            "/api/image/slice",
            params={"source": "local:foo.tif", "kind": "local", "norm": "slice"},
        )
        assert response.status_code == 200
        assert calls == []
