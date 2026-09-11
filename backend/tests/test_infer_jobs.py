"""Real (not mocked) round-trip tests for infer_jobs.run_infer_job, focused on
this session's rework: the per-slice worker pool, the GPU_FORWARD_LOCK/ML_LOCK
split, and #15's live-preview (incremental result/cache updates).

Skips automatically if the `ml` extra (torch/dlsia) isn't installed — see
test_train_e2e_real_ml.py for the sibling test proving the underlying
train/save/load contract. This file builds on the same real-model pattern but
drives it through infer_jobs.run_infer_job itself, with a fake array source
(no real Tiled/local dataset needed) so it stays fast and self-contained.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
pytest.importorskip("dlsia")

import arrays as arrays_mod  # noqa: E402
import export_jobs  # noqa: E402
import infer_jobs  # noqa: E402
import train_common  # noqa: E402
from schemas import DlsiaTunetConfig, InferRequest  # noqa: E402

IMAGE_SIZE = 64
N_SLICES = 6
N_CLASSES = 2


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DINO_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture()
def trained_run_id(runs_dir) -> str:
    """A real, tiny, trained dlsia_tunet run — same pattern as
    test_train_e2e_real_ml.py, sized down for test speed."""
    model_cfg = DlsiaTunetConfig(
        hyperparams={
            "epochs": 1, "depth": 2, "base_channels": 4, "growth_rate": 1.2,
            "batch_size": 2, "image_size": IMAGE_SIZE, "tiling": False,
        }
    )
    built = train_common.build_family(model_cfg, N_CLASSES, "cpu", lambda _msg: None)

    rng = np.random.default_rng(0)
    train_pairs = [
        (
            rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            rng.integers(0, N_CLASSES, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8),
        )
        for _ in range(4)
    ]
    train_common.run_training_loop(
        train_pairs=train_pairs, val_pairs=[], image_size=IMAGE_SIZE, n_classes=N_CLASSES,
        epochs=1, batch_size=2, seed=0, flip_augment=False,
        to_tensor_fn=built.to_tensor_fn, forward_fn=built.forward_fn,
        trainable_params=built.trainable_params, lr=1e-3, device="cpu",
        set_train_mode=built.set_train_mode,
    )

    run_id = "test-infer-job-run"
    train_common.save_run(
        run_id,
        model_family=model_cfg.model_family,
        model_config=built.model_config_snapshot,
        classes=[{"classId": 1, "label": "a", "color": "#f00"}, {"classId": 2, "label": "b", "color": "#0f0"}],
        render={},
        image_size=IMAGE_SIZE,
        hyperparams=model_cfg.hyperparams.model_dump(),
        source_keys=["local:fake.tif"],
        adapter_state=built.adapter_state_fn(),
        metrics={"epochs_completed": 1, "cancelled": False},
    )
    return run_id


@pytest.fixture()
def fake_array_source(monkeypatch: pytest.MonkeyPatch):
    """Serve a small synthetic in-memory volume in place of a real Tiled/
    local dataset — infer_jobs only ever calls these three arrays.py
    functions, so patching them is enough to drive run_infer_job for real
    without any actual dataset on disk.

    A short sleep per read_slice call gives concurrent slice-workers a real
    window to overlap in (proving the pool actually parallelizes I/O across
    slices, not just that it doesn't crash) and gives a background-thread
    poller time to observe partial (live-preview) results before the job
    finishes.
    """
    rng = np.random.default_rng(1)
    volume = rng.integers(0, 256, size=(N_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    def fake_resolve_array(source, kind, server_uri):
        return volume

    def fake_array_shape_meta(node, pyramid=None):
        return {"height": IMAGE_SIZE, "width": IMAGE_SIZE, "n_slices": N_SLICES}

    def fake_read_slice(node, meta, idx):
        time.sleep(0.05)
        return node[idx]

    monkeypatch.setattr(arrays_mod, "resolve_array", fake_resolve_array)
    monkeypatch.setattr(arrays_mod, "array_shape_meta", fake_array_shape_meta)
    monkeypatch.setattr(arrays_mod, "read_slice", fake_read_slice)
    return volume


def _infer_request(run_id: str) -> InferRequest:
    return InferRequest(
        run_id=run_id, kind="local", source="fake.tif",
        slice_indices=list(range(N_SLICES)), min_confidence=0.0,
    )


def test_run_infer_job_completes_with_correct_results_and_cache(trained_run_id, fake_array_source):
    jid = export_jobs.new_job(trained_run_id)
    infer_jobs.run_infer_job(jid, _infer_request(trained_run_id))

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    result = job["result"]
    assert result["cancelled"] is False
    assert sorted(int(k) for k in result["slices"]) == list(range(N_SLICES))
    assert result["preview_slices"] == list(range(N_SLICES))

    cached = infer_jobs._cache_get(jid)
    assert cached is not None
    assert sorted(cached["label_pngs"].keys()) == list(range(N_SLICES))

    # ML_LOCK must always be released, whether the job succeeds or not —
    # otherwise every subsequent train/infer/bake/probe job would wrongly
    # report "another job is already running" forever.
    assert not train_common.ML_LOCK.locked()


def test_run_infer_job_publishes_partial_results_before_completion(trained_run_id, fake_array_source):
    """Regression test for #15: the job's `result` must be readable and
    growing WHILE state is still "running", not only once "done" — this is
    what lets the frontend show a usable preview slider mid-job instead of
    the old "nothing until every slice finishes" behavior."""
    jid = export_jobs.new_job(trained_run_id)
    thread = threading.Thread(
        target=infer_jobs.run_infer_job, args=(jid, _infer_request(trained_run_id)), daemon=True,
    )
    thread.start()

    saw_partial_progress = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        job = export_jobs.get_job(jid)
        if job["state"] == "running" and job.get("result"):
            n = len(job["result"].get("preview_slices", []))
            if 0 < n < N_SLICES:
                saw_partial_progress = True
                break
        if job["state"] in ("done", "error"):
            break
        time.sleep(0.02)
    thread.join(timeout=30.0)

    assert saw_partial_progress, "expected to observe a partial (running) result with some but not all slices done"
    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    assert len(job["result"]["preview_slices"]) == N_SLICES


def test_run_infer_job_cancellation_stops_early_with_partial_results(trained_run_id, fake_array_source):
    jid = export_jobs.new_job(trained_run_id)
    thread = threading.Thread(
        target=infer_jobs.run_infer_job, args=(jid, _infer_request(trained_run_id)), daemon=True,
    )
    thread.start()
    time.sleep(0.06)  # let at least one slice start
    export_jobs.request_cancel(jid)
    thread.join(timeout=30.0)

    job = export_jobs.get_job(jid)
    assert job["state"] == "done"
    assert job["result"]["cancelled"] is True
    assert len(job["result"]["preview_slices"]) <= N_SLICES
    assert not train_common.ML_LOCK.locked()


def test_predict_one_slice_serializes_gpu_forward_calls(trained_run_id, fake_array_source, monkeypatch):
    """Direct test of the concurrency guarantee GPU_FORWARD_LOCK exists for:
    even with several slice-workers running at once, at most one is ever
    inside the model-forward critical section at a time."""
    import dlsia_runtime as fam

    config = train_common.load_run_config(trained_run_id)
    adapter_state = train_common.load_adapter_state(trained_run_id)
    model = fam.load_model(adapter_state, "cpu")
    model.eval()
    real_forward_fn = fam.make_forward_fn(model)
    to_tensor_fn = fam.make_to_tensor_fn()

    concurrent_count = 0
    max_concurrent = 0
    count_lock = threading.Lock()

    def spying_forward_fn(batch):
        nonlocal concurrent_count, max_concurrent
        with count_lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        try:
            time.sleep(0.05)  # widen the window a real race would need
            return real_forward_fn(batch)
        finally:
            with count_lock:
                concurrent_count -= 1

    node = fake_array_source
    meta = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    request = _infer_request(trained_run_id)
    run_classes = config["classes"]
    tiling_logged = threading.Event()

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                infer_jobs._predict_one_slice,
                i,
                jid="test-jid", node=node, meta=meta, h=IMAGE_SIZE, w=IMAGE_SIZE,
                render={}, global_range=None,
                render_slice_fn=train_common.denoising_render_slice_fn(None),
                tiled=False, image_size=IMAGE_SIZE, forward_fn=spying_forward_fn,
                to_tensor_fn=to_tensor_fn, device="cpu", request=request,
                run_classes=run_classes, tiling_logged=tiling_logged,
            )
            for i in range(N_SLICES)
        ]
        results = [f.result() for f in futures]

    for slice_idx, png_bytes, shapes, error in results:
        assert error is None, f"slice {slice_idx} failed: {error}"
        assert png_bytes is not None
        assert shapes is not None

    assert max_concurrent == 1, "GPU_FORWARD_LOCK failed to serialize concurrent forward calls"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHexToRgb:
    def test_full_hex(self):
        assert infer_jobs._hex_to_rgb("#00ff80") == (0, 255, 128)

    def test_short_hex_is_expanded(self):
        assert infer_jobs._hex_to_rgb("#0f8") == (0, 255, 136)

    def test_none_falls_back_to_red(self):
        assert infer_jobs._hex_to_rgb(None) == (255, 0, 0)

    def test_missing_hash_falls_back_to_red(self):
        assert infer_jobs._hex_to_rgb("00ff80") == (255, 0, 0)

    def test_invalid_hex_digits_fall_back_to_red(self):
        assert infer_jobs._hex_to_rgb("#zzzzzz") == (255, 0, 0)


class TestVectorizeLabelMap:
    def _run_classes(self):
        return [{"classId": 1, "label": "a"}, {"classId": 2, "label": "b"}]

    def test_component_below_min_area_is_dropped(self):
        label_map = np.zeros((20, 20), dtype=np.uint8)
        label_map[0:2, 0:2] = 1  # 4px, tiny
        shapes = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=50, simplify_tol=0.0, run_id="run12345", slice_idx=0)
        assert shapes == []

    def test_ring_shaped_component_gets_a_hole(self):
        label_map = np.ones((30, 30), dtype=np.uint8)  # all class 1
        label_map[10:20, 10:20] = 2  # a class-2 hole inside class 1
        shapes = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=1, simplify_tol=0.0, run_id="run12345", slice_idx=0)
        class1_shape = next(s for s in shapes if s["classId"] == 1)
        assert "holes" in class1_shape
        assert len(class1_shape["holes"]) == 1

    def test_simplify_tol_reduces_point_count(self):
        label_map = np.zeros((40, 40), dtype=np.uint8)
        label_map[5:35, 5:35] = 1  # a large, simple square
        unsimplified = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=1, simplify_tol=0.0, run_id="run12345", slice_idx=0)
        simplified = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=1, simplify_tol=5.0, run_id="run12345", slice_idx=0)
        assert len(simplified[0]["points"]) <= len(unsimplified[0]["points"])

    def test_shape_ids_are_unique_per_component(self):
        label_map = np.zeros((30, 30), dtype=np.uint8)
        label_map[2:6, 2:6] = 1
        label_map[20:26, 20:26] = 1
        shapes = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=1, simplify_tol=0.0, run_id="run12345", slice_idx=3)
        ids = [s["id"] for s in shapes]
        assert len(ids) == len(set(ids))

    def test_class_with_no_pixels_is_skipped(self):
        label_map = np.zeros((10, 10), dtype=np.uint8)
        shapes = infer_jobs._vectorize_label_map(label_map, self._run_classes(), min_area=1, simplify_tol=0.0, run_id="run12345", slice_idx=0)
        assert shapes == []


# ---------------------------------------------------------------------------
# run_infer_job — guard paths that don't need a real trained run
# ---------------------------------------------------------------------------

class TestRunInferJobGuards:
    def test_busy_ml_lock_reports_error_not_a_crash(self):
        train_common.ML_LOCK.acquire()
        try:
            jid = export_jobs.new_job("x")
            infer_jobs.run_infer_job(jid, _infer_request("whatever"))
            job = export_jobs.get_job(jid)
            assert job["state"] == "error"
            assert "already running" in job["error"]
        finally:
            train_common.ML_LOCK.release()

    def test_no_device_reports_error_and_releases_lock(self, monkeypatch):
        monkeypatch.setattr(train_common, "pick_device", lambda: None)
        jid = export_jobs.new_job("x")
        infer_jobs.run_infer_job(jid, _infer_request("whatever"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "torch is not installed" in job["error"]
        assert train_common.ML_LOCK.locked() is False

    def test_denoiser_run_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            train_common, "load_run_config",
            lambda run_id: {"model_family": "dlsia_denoiser", "classes": [], "image_size": 64, "render": {}},
        )
        monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {})
        jid = export_jobs.new_job("x")
        infer_jobs.run_infer_job(jid, _infer_request("whatever"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "denoiser run" in job["error"]

    def test_unsupported_model_family_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            train_common, "load_run_config",
            lambda run_id: {"model_family": "something_weird", "classes": [], "image_size": 64, "render": {}},
        )
        monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {})
        jid = export_jobs.new_job("x")
        infer_jobs.run_infer_job(jid, _infer_request("whatever"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "Unsupported model family" in job["error"]

    def test_tiled_run_without_qlty_is_refused(self, monkeypatch):
        import tiling

        monkeypatch.setattr(
            train_common, "load_run_config",
            lambda run_id: {
                "model_family": "dlsia_tunet", "classes": [], "image_size": 64, "render": {},
                "hyperparams": {"tiling": True},
            },
        )
        monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {})
        monkeypatch.setattr(tiling, "qlty_available", lambda: False)
        jid = export_jobs.new_job("x")
        infer_jobs.run_infer_job(jid, _infer_request("whatever"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "qlty" in job["error"]


# ---------------------------------------------------------------------------
# preview_png
# ---------------------------------------------------------------------------

class TestPreviewPng:
    def test_missing_job_is_404(self):
        with pytest.raises(Exception) as exc:
            infer_jobs.preview_png("no-such-job", 0)
        assert exc.value.status_code == 404

    def test_missing_slice_is_404(self):
        infer_jobs._cache_put("job-with-no-slices", {"classes": [], "label_pngs": {}})
        with pytest.raises(Exception) as exc:
            infer_jobs.preview_png("job-with-no-slices", 0)
        assert exc.value.status_code == 404

    def test_colorizes_predicted_classes(self):
        from PIL import Image as PILImage

        label = np.zeros((8, 8), dtype=np.uint8)
        label[0:4, 0:4] = 1
        label[4:8, 4:8] = 2
        buf = io.BytesIO()
        PILImage.fromarray(label, mode="L").save(buf, format="PNG")

        infer_jobs._cache_put(
            "job-colorize",
            {
                "classes": [{"classId": 1, "color": "#ff0000"}, {"classId": 2, "color": "#00ff00"}],
                "label_pngs": {0: buf.getvalue()},
            },
        )
        png = infer_jobs.preview_png("job-colorize", 0)
        rgba = np.asarray(PILImage.open(io.BytesIO(png)))
        assert tuple(rgba[0, 0]) == (255, 0, 0, 180)
        assert tuple(rgba[4, 4]) == (0, 255, 0, 180)
        assert rgba[7, 0][3] == 0  # background stays transparent


# ---------------------------------------------------------------------------
# run_write_tiled_job
# ---------------------------------------------------------------------------

class TestRunWriteTiledJob:
    def test_missing_cache_entry_reports_error(self):
        jid = export_jobs.new_job("x")
        infer_jobs.run_write_tiled_job(jid, "no-such-infer-job")
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "No cached inference results" in job["error"]

    def test_non_tiled_source_is_refused(self):
        infer_jobs._cache_put("infer-local", {"kind": "local", "classes": [], "label_pngs": {}})
        jid = export_jobs.new_job("x")
        infer_jobs.run_write_tiled_job(jid, "infer-local")
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "not a Tiled array" in job["error"]

    def test_writes_semantic_and_class_volumes(self, monkeypatch):
        from PIL import Image as PILImage

        label0 = np.zeros((4, 4), dtype=np.uint8)
        label0[0:2, 0:2] = 1
        label1 = np.zeros((4, 4), dtype=np.uint8)
        label1[2:4, 2:4] = 2

        def _png(arr):
            buf = io.BytesIO()
            PILImage.fromarray(arr, mode="L").save(buf, format="PNG")
            return buf.getvalue()

        infer_jobs._cache_put(
            "infer-tiled",
            {
                "kind": "tiled", "source": "browse/sample", "server_uri": "http://x",
                "classes": [{"classId": 1, "label": "a", "color": "#f00"}, {"classId": 2, "label": "b", "color": "#0f0"}],
                "label_pngs": {0: _png(label0), 1: _png(label1)},
            },
        )

        captured = {}

        def fake_write_masks_to_tiled(source, server_uri, volumes, classes, container_suffix=""):
            captured.update(source=source, server_uri=server_uri, volumes=volumes, container_suffix=container_suffix)
            return {"path": "browse/sample__masks_deep"}

        import tiled_mask_sync

        monkeypatch.setattr(tiled_mask_sync, "write_masks_to_tiled", fake_write_masks_to_tiled)

        jid = export_jobs.new_job("x")
        infer_jobs.run_write_tiled_job(jid, "infer-tiled")
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"] == {"path": "browse/sample__masks_deep"}
        assert captured["container_suffix"] == "_deep"
        assert captured["volumes"]["semantic"].shape == (2, 4, 4)
        assert np.array_equal(captured["volumes"]["class_vols"]["a"][0], (label0 == 1) * 255)

    def test_write_failure_is_reported_as_a_job_error(self, monkeypatch):
        infer_jobs._cache_put(
            "infer-tiled-fail",
            {
                "kind": "tiled", "source": "browse/sample", "server_uri": None,
                "classes": [{"classId": 1, "label": "a", "color": "#f00"}],
                "label_pngs": {0: _blank_label_png()},
            },
        )
        import tiled_mask_sync

        def boom(*a, **k):
            raise RuntimeError("tiled write blew up")

        monkeypatch.setattr(tiled_mask_sync, "write_masks_to_tiled", boom)
        jid = export_jobs.new_job("x")
        infer_jobs.run_write_tiled_job(jid, "infer-tiled-fail")
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "tiled write blew up" in job["error"]


def _blank_label_png() -> bytes:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()
