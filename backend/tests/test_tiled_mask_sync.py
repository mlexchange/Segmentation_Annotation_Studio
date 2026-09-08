"""Unit tests for tiled_mask_sync.build_mask_volumes (pure rasterization → volumes)."""
import io
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

import tiled_mask_sync
from schemas import AnnotationClass, ExportSourceItem, PredictedSlicePointer
from tiled_mask_sync import build_mask_volumes, merge_mask_volumes

H = W = 32


def _classes():
    return [
        AnnotationClass(classId=10, label="Cell", color="#ff0000"),
        AnnotationClass(classId=20, label="Wall", color="#00ff00"),
    ]


def test_build_mask_volumes_stacks_and_labels():
    slices = {
        "5": [{"id": "b", "kind": "polygon", "classId": 20,
               "points": [10, 10, 20, 10, 20, 20, 10, 20]}],
        "2": [{"id": "a", "kind": "rectangle", "classId": 10,
               "x": 2, "y": 2, "w": 6, "h": 6}],
    }
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", server_uri=None, slices=slices)
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    # Sorted numeric slice order.
    assert vols["slice_indices"] == [2, 5]
    assert vols["semantic"].shape == (2, H, W)
    assert vols["semantic"].dtype == np.uint8

    # Slice 0 (key "2") = the rectangle → class id 1 (Cell); slice 1 = polygon → id 2 (Wall).
    assert vols["semantic"][0].max() == 1
    assert vols["semantic"][1].max() == 2
    assert vols["semantic"][0, 4, 4] == 1   # inside the rect
    assert vols["semantic"][1, 15, 15] == 2  # inside the polygon

    # One binary volume per class, 0/255, present only on its slice.
    assert set(vols["class_vols"]) == {"Cell", "Wall"}
    cell, wall = vols["class_vols"]["Cell"], vols["class_vols"]["Wall"]
    assert cell.shape == (2, H, W) and wall.shape == (2, H, W)
    assert set(np.unique(cell)).issubset({0, 255})
    assert cell[0].sum() > 0 and cell[1].sum() == 0   # Cell only on slice 0
    assert wall[1].sum() > 0 and wall[0].sum() == 0   # Wall only on slice 1

    # Legend mirrors the zip's legend shape.
    assert vols["legend"] == [
        {"id": 1, "name": "Cell", "color": "#ff0000"},
        {"id": 2, "name": "Wall", "color": "#00ff00"},
    ]


def test_negative_slices_emitted_as_zero_frames():
    slices = {"1": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 4, "h": 4}]}
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices=slices, negative_slices=["3"])
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    assert vols["slice_indices"] == [1, 3]
    # Negative slice 3 (index 1 in the stack) is all background.
    assert vols["semantic"][1].sum() == 0


def test_returns_none_without_slices():
    item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices={})
    assert build_mask_volumes(item, _classes(), {"height": H, "width": W}) is None


def _fake_commit_png(raw_label_map: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(raw_label_map, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_predicted_slices_reads_straight_from_the_run_commit_png(monkeypatch: pytest.MonkeyPatch):
    """Regression test for the lazy-vectorization plan item: a slice with no
    real shapes but a predicted-slice pointer must be rasterized directly
    from the ipred run's commit.png (fetched server-side), never requiring
    the frontend to have vectorized it into polygon shapes first."""
    raw = np.zeros((H, W), dtype=np.uint8)
    raw[2:6, 2:6] = 10  # raw frontend classId, matching AnnotationClass(classId=10, "Cell")
    calls: list[str] = []

    def fake_run_commit_png(run_id: str) -> bytes:
        calls.append(run_id)
        assert run_id == "run-abc"
        return _fake_commit_png(raw)

    monkeypatch.setattr(tiled_mask_sync.ipred_client, "run_commit_png", fake_run_commit_png)

    item = ExportSourceItem(
        kind="tiled", source="browse/ds/img", slices={},
        predicted_slices={"7": PredictedSlicePointer(run_id="run-abc", class_ids=[10, 20])},
    )
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    assert calls == ["run-abc"]
    assert vols["slice_indices"] == [7]
    assert vols["semantic"][0, 3, 3] == 1  # remapped from raw classId 10 -> legend id 1 (Cell)
    assert vols["class_vols"]["Cell"][0, 3, 3] == 255
    assert vols["class_vols"]["Wall"][0].sum() == 0


def test_real_shapes_win_over_a_predicted_pointer_for_the_same_slice(monkeypatch: pytest.MonkeyPatch):
    """A slice already vectorized/edited into real shapes must never fall
    back to its (possibly stale) predicted pointer — matches the frontend's
    own precedence in handleCommitVolumeApply/predictedRasterStore."""
    def fake_run_commit_png(run_id: str) -> bytes:
        raise AssertionError("must not fetch commit.png when real shapes already cover this slice")

    monkeypatch.setattr(tiled_mask_sync.ipred_client, "run_commit_png", fake_run_commit_png)

    slices = {"7": [{"id": "a", "kind": "rectangle", "classId": 20, "x": 1, "y": 1, "w": 3, "h": 3}]}
    item = ExportSourceItem(
        kind="tiled", source="browse/ds/img", slices=slices,
        predicted_slices={"7": PredictedSlicePointer(run_id="run-abc", class_ids=[10, 20])},
    )
    vols = build_mask_volumes(item, _classes(), {"height": H, "width": W})

    assert vols is not None
    assert vols["slice_indices"] == [7]
    assert vols["semantic"][0].max() == 2  # Wall (classId 20), from the real shape — not the pointer


def _volumes(slices, negatives=None):
    item = ExportSourceItem(
        kind="tiled", source="browse/ds/img", slices=slices, negative_slices=negatives or [],
    )
    return build_mask_volumes(item, _classes(), {"height": H, "width": W})


def test_merge_updates_pushed_slice_and_keeps_others():
    # Existing container holds slices 2 (Cell) and 5 (Wall).
    existing_vols = _volumes({
        "2": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 6, "h": 6}],
        "5": [{"id": "b", "kind": "rectangle", "classId": 20, "x": 2, "y": 2, "w": 6, "h": 6}],
    })
    existing = {
        "slice_indices": existing_vols["slice_indices"],
        "semantic": existing_vols["semantic"],
        "class_arrays": existing_vols["class_vols"],
        "legend": existing_vols["legend"],
    }

    # Re-push ONLY slice 2, now as Wall (class changed on that slice).
    new = _volumes({"2": [{"id": "c", "kind": "rectangle", "classId": 20, "x": 2, "y": 2, "w": 6, "h": 6}]})
    merged = merge_mask_volumes(existing, new)

    # Both slices survive; only slice 2 reported as updated.
    assert merged["slice_indices"] == [2, 5]
    assert merged["updated_indices"] == [2]
    # Slice 2 (index 0) is now Wall (id 2), slice 5 (index 1) still Wall.
    assert merged["semantic"][0, 4, 4] == 2   # updated slice → Wall
    assert merged["semantic"][1, 4, 4] == 2   # untouched slice preserved
    # Slice 2's Cell mask was replaced (now empty there); Wall present on both.
    assert merged["class_vols"]["Cell"][0].sum() == 0
    assert merged["class_vols"]["Wall"][0].sum() > 0


def test_merge_fresh_when_no_existing():
    new = _volumes({"3": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 1, "y": 1, "w": 4, "h": 4}]})
    merged = merge_mask_volumes(None, new)
    assert merged["slice_indices"] == [3]
    assert merged["updated_indices"] == [3]
    assert merged["semantic"].shape == (1, H, W)


# ---------------------------------------------------------------------------
# _read_existing_masks / write_masks_to_tiled — fake Tiled container, no live
# server. Only mask_pyramid.register_mask_pyramid actually needs a live Tiled
# server (see mask_pyramid.py's own test file for that established boundary),
# so it's stubbed here; everything else about write_masks_to_tiled is real
# container navigation/merge logic that can run against a fake.
# ---------------------------------------------------------------------------

class FakeMaskContainer:
    def __init__(self, metadata=None, children=None):
        self.metadata = metadata or {}
        self._children: dict[str, Any] = dict(children or {})
        self.deleted = False
        self.written: dict[str, dict] = {}
        self.created: list[str] = []

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)

    def keys(self):
        return list(self._children.keys())

    def delete_contents(self, recursive=True, external_only=False):
        self.deleted = True
        self._children = {}

    def update_metadata(self, metadata):
        self.metadata = metadata

    def write_array(self, arr, key, dims=None, metadata=None):
        self.written[key] = {"arr": arr, "dims": dims, "metadata": metadata}

    def create_container(self, key, metadata):
        child = FakeMaskContainer(metadata=metadata)
        self._children[key] = child
        self.created.append(key)
        return child


@pytest.fixture(autouse=True)
def stub_register_mask_pyramid(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_register(semantic, key, container, cache_key):
        calls.append({"semantic": semantic, "key": key, "container": container, "cache_key": cache_key})
        return {"key": key}

    monkeypatch.setattr(tiled_mask_sync.mask_pyramid, "register_mask_pyramid", fake_register)
    return calls


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch):
    ds = FakeMaskContainer()
    root = FakeMaskContainer(children={"browse": FakeMaskContainer(children={"ds": ds})})
    monkeypatch.setattr(tiled_mask_sync, "get_tiled_client", lambda uri, key: root)
    monkeypatch.setattr(tiled_mask_sync, "api_key_for_uri", lambda uri: None)
    return ds  # the "browse/ds" container new mask containers get created under


class TestReadExistingMasks:
    def test_reads_semantic_via_mask_pyramid_and_class_arrays_directly(self, monkeypatch: pytest.MonkeyPatch):
        semantic = np.zeros((2, H, W), dtype=np.uint8)
        monkeypatch.setattr(tiled_mask_sync.mask_pyramid, "read_mask_scale0", lambda container, key: semantic)
        cell_arr = np.full((2, H, W), 255, dtype=np.uint8)
        container = FakeMaskContainer(
            metadata={
                "legend": [{"id": 1, "name": "Cell", "color": "#f00"}],
                "slice_indices": [1, 2],
                "slice_updated_at": {"1": "t1"},
            },
            children={"semantic": object(), "Cell": cell_arr},
        )
        result = tiled_mask_sync._read_existing_masks(container)
        assert result is not None
        assert result["slice_indices"] == [1, 2]
        assert np.array_equal(result["semantic"], semantic)
        assert np.array_equal(result["class_arrays"]["Cell"], cell_arr)
        assert result["slice_updated_at"] == {"1": "t1"}

    def test_unreadable_container_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tiled_mask_sync.mask_pyramid, "read_mask_scale0",
            lambda container, key: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        container = FakeMaskContainer(metadata={"legend": [], "slice_indices": []}, children={"semantic": object()})
        assert tiled_mask_sync._read_existing_masks(container) is None

    def test_unknown_class_key_is_skipped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tiled_mask_sync.mask_pyramid, "read_mask_scale0", lambda container, key: np.zeros((1, H, W)))
        container = FakeMaskContainer(
            metadata={"legend": [{"id": 1, "name": "Cell"}], "slice_indices": [0]},
            children={"semantic": object(), "orphan_key": np.zeros((1, H, W))},
        )
        result = tiled_mask_sync._read_existing_masks(container)
        assert result["class_arrays"] == {}


class TestWriteMasksToTiled:
    def _volumes_dict(self):
        return _volumes({"2": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 6, "h": 6}]})

    def test_creates_a_new_container_when_none_exists(self, fake_client, stub_register_mask_pyramid):
        info = tiled_mask_sync.write_masks_to_tiled("browse/ds/img", None, self._volumes_dict(), _classes())
        assert info["path"] == "browse/ds/img__masks"
        assert "img__masks" in fake_client.created
        new_container = fake_client["img__masks"]
        assert new_container.metadata["studio_type"] == "segmentation_masks"
        assert "Cell" in new_container.written

    def test_container_suffix_keeps_producers_independent(self, fake_client, stub_register_mask_pyramid):
        info = tiled_mask_sync.write_masks_to_tiled(
            "browse/ds/img", None, self._volumes_dict(), _classes(), container_suffix="_deep",
        )
        assert info["path"] == "browse/ds/img__masks_deep"
        assert stub_register_mask_pyramid[0]["cache_key"] == "img__masks_deep"

    def test_merges_into_an_existing_container(self, fake_client, monkeypatch, stub_register_mask_pyramid):
        existing_vols = self._volumes_dict()
        existing_container = FakeMaskContainer(
            metadata={
                "legend": existing_vols["legend"], "slice_indices": existing_vols["slice_indices"],
            },
            children={"semantic": object(), "Cell": existing_vols["class_vols"]["Cell"]},
        )
        monkeypatch.setattr(
            tiled_mask_sync.mask_pyramid, "read_mask_scale0",
            lambda container, key: existing_vols["semantic"],
        )
        fake_client._children["img__masks"] = existing_container

        new_vols = _volumes({"5": [{"id": "b", "kind": "rectangle", "classId": 20, "x": 1, "y": 1, "w": 3, "h": 3}]})
        info = tiled_mask_sync.write_masks_to_tiled("browse/ds/img", None, new_vols, _classes())

        assert existing_container.deleted is True
        assert info["n_slices"] == 2  # slice 2 (existing) + slice 5 (new)
        assert info["updated"] == 1

    def test_shape_mismatch_discards_existing_and_replaces(self, fake_client, monkeypatch, stub_register_mask_pyramid):
        mismatched_semantic = np.zeros((1, H * 2, W * 2), dtype=np.uint8)
        existing_container = FakeMaskContainer(
            metadata={"legend": [], "slice_indices": [9]},
            children={"semantic": object()},
        )
        monkeypatch.setattr(tiled_mask_sync.mask_pyramid, "read_mask_scale0", lambda container, key: mismatched_semantic)
        fake_client._children["img__masks"] = existing_container

        info = tiled_mask_sync.write_masks_to_tiled("browse/ds/img", None, self._volumes_dict(), _classes())
        # Old slice 9 is gone entirely — replaced, not merged, due to the H/W mismatch.
        assert info["n_slices"] == 1
        assert info["updated"] == 1

    def test_returns_correct_summary_counts(self, fake_client, stub_register_mask_pyramid):
        info = tiled_mask_sync.write_masks_to_tiled("browse/ds/img", None, self._volumes_dict(), _classes())
        assert info["n_classes"] == 2  # Cell + Wall
        assert info["n_slices"] == 1
        assert info["updated"] == 1


# ---------------------------------------------------------------------------
# run_mask_sync_job
# ---------------------------------------------------------------------------

class _Payload:
    def __init__(self, classes):
        self.classes = classes


class TestRunMaskSyncJob:
    def test_no_tiled_sources_reports_done_with_a_note(self):
        import export_jobs

        local_item = ExportSourceItem(kind="local", source="foo.tif", slices={"0": []})
        jid = export_jobs.new_job("x")
        tiled_mask_sync.run_mask_sync_job(jid, [local_item], _Payload(_classes()))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["written"] == []
        assert "skipped" in job["result"]["note"]

    def test_writes_masks_for_each_tiled_source(self, monkeypatch, fake_client, stub_register_mask_pyramid):
        import export_jobs

        slices = {"2": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 2, "y": 2, "w": 6, "h": 6}]}
        item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices=slices)
        monkeypatch.setattr(tiled_mask_sync.arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
        monkeypatch.setattr(tiled_mask_sync.arrays_mod, "array_shape_meta", lambda node: {"height": H, "width": W})

        jid = export_jobs.new_job("x")
        tiled_mask_sync.run_mask_sync_job(jid, [item], _Payload(_classes()))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert len(job["result"]["written"]) == 1
        assert job["result"]["written"][0]["source"] == "browse/ds/img"
        assert job["result"]["written"][0]["container"] == "browse/ds/img__masks"

    def test_item_with_no_annotated_slices_is_skipped_not_errored(self, monkeypatch, fake_client):
        import export_jobs

        item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices={})
        monkeypatch.setattr(tiled_mask_sync.arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
        monkeypatch.setattr(tiled_mask_sync.arrays_mod, "array_shape_meta", lambda node: {"height": H, "width": W})

        jid = export_jobs.new_job("x")
        tiled_mask_sync.run_mask_sync_job(jid, [item], _Payload(_classes()))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["written"] == []

    def test_exception_is_reported_as_a_job_error(self, monkeypatch):
        import export_jobs

        item = ExportSourceItem(kind="tiled", source="browse/ds/img", slices={"0": [{"id": "a", "kind": "rectangle", "classId": 10, "x": 0, "y": 0, "w": 2, "h": 2}]})
        monkeypatch.setattr(
            tiled_mask_sync.arrays_mod, "resolve_array",
            lambda source, kind, server_uri: (_ for _ in ()).throw(RuntimeError("resolve failed")),
        )
        jid = export_jobs.new_job("x")
        tiled_mask_sync.run_mask_sync_job(jid, [item], _Payload(_classes()))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "resolve failed" in job["error"]
