"""HTTP-boundary tests for /api/browse/slices.

The Browse drill-in maps a returned row's *position* to a slice index ("open
this dataset at slice N"), so the route's list must match the backend's own
slice ordering (``arrays._stack_keys``). Sidecar containers written alongside an
annotated array — ``<name>__v_thumbs``, ``<name>__masks`` — are not slices; if
they leak into this list they appear as phantom rows and shift the index of
every slice after them.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import annotation_server
from annotation_server import app


@pytest.fixture
def fake_tiled(monkeypatch):
    """Stub the Tiled plumbing so the route runs without a live server.

    Returns a setter that installs the item list ``tiled_search_items`` yields.
    """
    annotation_server._items_cache.clear()
    monkeypatch.setattr(annotation_server, "_require_tiled_server", lambda uri: "http://fake")
    monkeypatch.setattr(annotation_server, "get_tiled_client", lambda uri: object())
    monkeypatch.setattr(annotation_server, "get_browse_container_for", lambda client, path: (object(), path))

    def _install(samples: list[str]) -> None:
        items = [
            {"path": f"browse/dataset/{s}", "sample": s, "metadata": {}, "n_slices": 1}
            for s in samples
        ]
        monkeypatch.setattr(
            annotation_server,
            "tiled_search_items",
            lambda container, limit, container_path_prefix: {"items": items, "total": len(items)},
        )

    yield _install
    annotation_server._items_cache.clear()


async def _get_slices(path: str = "browse/dataset") -> dict:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/browse/slices", params={"path": path})
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar", ["slice_0001__v_thumbs", "slice_0001__masks"])
async def test_sidecar_containers_are_not_listed_as_slices(fake_tiled, sidecar: str) -> None:
    fake_tiled(["slice_0001", sidecar, "slice_0002"])

    body = await _get_slices()

    assert [it["sample"] for it in body["items"]] == ["slice_0001", "slice_0002"]


@pytest.mark.asyncio
async def test_total_reflects_the_filtered_count(fake_tiled) -> None:
    """`total` must not keep counting rows the response no longer contains."""
    fake_tiled(["slice_0001", "slice_0001__v_thumbs", "slice_0002__masks"])

    body = await _get_slices()

    assert body["total"] == 1
    assert len(body["items"]) == 1


@pytest.mark.asyncio
async def test_slice_positions_stay_aligned_with_backend_slice_order(fake_tiled) -> None:
    """The whole point of filtering: a sidecar sorting *before* a real slice
    would otherwise shift that slice's index, so Browse would open the wrong
    one. Positions here must match arrays._stack_keys' sorted, sidecar-free
    ordering."""
    from arrays import _stack_keys

    keys = ["a_0001", "a_0001__masks", "b_0002", "c_0003"]
    fake_tiled(keys)

    body = await _get_slices()
    samples = [it["sample"] for it in body["items"]]

    assert samples == _stack_keys({k: object() for k in keys})
    assert samples.index("b_0002") == 1  # not 2


@pytest.mark.asyncio
async def test_ordinary_slices_are_returned_sorted(fake_tiled) -> None:
    """No sidecars present — behaviour is unchanged (lexical == slice order)."""
    fake_tiled(["slice_0003", "slice_0001", "slice_0002"])

    body = await _get_slices()

    assert [it["sample"] for it in body["items"]] == ["slice_0001", "slice_0002", "slice_0003"]
