"""HTTP-level tests for annotation_server.py's /api/browse/* routes.

Only `get_tiled_client` is faked (returns a lightweight fake Tiled root node,
same duck-typed FakeNode as test_browse_helpers.py) — the real
`get_browse_container_for` runs against it unmocked, since it's pure
navigation logic (`node[k]` per path segment). Every test supplies an
explicit container_path so get_browse_container_for never falls through to
the heuristic root-discovery path (get_browse_container), which is out of
scope here. The underlying field-mapping/distinct/search logic itself is
already thoroughly covered directly in test_browse_helpers.py — these tests
are about routing, query-param parsing, and error handling, not re-proving
that logic.
"""
from __future__ import annotations

from collections import Counter

import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server

tiled_queries = pytest.importorskip("tiled.queries")


class FakeNode:
    def __init__(self, children=None, metadata=None, is_container=True, search_raises=False):
        self._children = children or {}
        self.metadata = metadata or {}
        self.structure_family = "container" if is_container else "array"
        self._search_raises = search_raises

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)

    def search(self, query):
        if self._search_raises:
            raise RuntimeError("search exploded")
        matched = {}
        for k, child in self._children.items():
            meta = child.metadata or {}
            if isinstance(query, tiled_queries.Contains):
                val = meta.get(query.key)
                if isinstance(val, (list, tuple)) and query.value in val:
                    matched[k] = child
            else:  # Eq
                if meta.get(query.key) == query.value:
                    matched[k] = child
        return FakeNode(children=matched, metadata=self.metadata)

    def distinct(self, key, counts=True):
        counter: Counter = Counter()
        for child in self._children.values():
            val = (child.metadata or {}).get(key)
            if val is None:
                continue
            counter[val] += 1
        return {"metadata": {key: [{"value": v, "count": n} for v, n in counter.items()]}}


def _sample(metadata):
    return FakeNode(metadata=metadata, is_container=False)


@pytest.fixture()
def browse_root():
    """browse/sample1 (PI=Smith), browse/sample2 (PI=Jones) — enough to
    exercise facet discovery (>=2 distinct PI values), column distinct, and
    item search/filtering."""
    samples = {
        "sample1": _sample({"PI": "Smith", "sample_name": "s1", "technique": "GIWAXS"}),
        "sample2": _sample({"PI": "Jones", "sample_name": "s2", "technique": "GIWAXS"}),
    }
    browse = FakeNode(children=samples)
    return FakeNode(children={"browse": browse})


@pytest.fixture(autouse=True)
def fake_tiled_client(monkeypatch: pytest.MonkeyPatch, browse_root):
    monkeypatch.setattr(annotation_server, "get_tiled_client", lambda *a, **k: browse_root)
    # Field-mapping / column / item caches are module-level and would leak
    # a fake result from one test's fake node into the next test's assertions
    # (same server_uri/technique/container_path -> same cache key).
    annotation_server._field_mapping_cache._store.clear()
    annotation_server._column_cache._store.clear()
    annotation_server._items_cache._store.clear()


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


app = annotation_server.app


@pytest.mark.asyncio
async def test_browse_facets_finds_multi_valued_field(client):
    response = await client.get("/api/browse/facets", params={"container_path": "browse"})
    assert response.status_code == 200
    assert "PI" in response.json()["facets"]


@pytest.mark.asyncio
async def test_browse_facets_returns_empty_lists_on_internal_error(client, monkeypatch):
    monkeypatch.setattr(annotation_server, "get_tiled_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    response = await client.get("/api/browse/facets", params={"container_path": "browse"})
    assert response.status_code == 200
    assert response.json() == {"facets": [], "techniques": []}


@pytest.mark.asyncio
async def test_browse_column_returns_distinct_values(client):
    response = await client.get(
        "/api/browse/column", params={"field": "PI", "container_path": "browse"},
    )
    assert response.status_code == 200
    values = {v["value"] for v in response.json()["values"]}
    assert values == {"Smith", "Jones"}


@pytest.mark.asyncio
async def test_browse_column_502s_on_internal_error(client, monkeypatch):
    monkeypatch.setattr(annotation_server, "get_tiled_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    response = await client.get(
        "/api/browse/column", params={"field": "PI", "container_path": "browse"},
    )
    assert response.status_code == 502


@pytest.mark.asyncio
async def test_browse_items_returns_matching_samples(client):
    response = await client.get(
        "/api/browse/items",
        params={"container_path": "browse", "filters": '{"PI": "Smith"}'},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["sample"] == "sample1"


@pytest.mark.asyncio
async def test_browse_items_no_filters_returns_all(client):
    response = await client.get("/api/browse/items", params={"container_path": "browse"})
    assert response.status_code == 200
    assert response.json()["total"] == 2


@pytest.mark.asyncio
async def test_browse_items_malformed_filter_json_treated_as_no_filter(client):
    response = await client.get(
        "/api/browse/items", params={"container_path": "browse", "filters": "not json"},
    )
    assert response.status_code == 200
    assert response.json()["total"] == 2


@pytest.mark.asyncio
async def test_browse_items_502s_on_internal_error(client, monkeypatch):
    monkeypatch.setattr(annotation_server, "get_tiled_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    response = await client.get("/api/browse/items", params={"container_path": "browse"})
    assert response.status_code == 502


@pytest.mark.asyncio
async def test_browse_slices_lists_container_children(client):
    response = await client.get("/api/browse/slices", params={"path": "browse"})
    assert response.status_code == 200
    assert response.json()["total"] == 2
