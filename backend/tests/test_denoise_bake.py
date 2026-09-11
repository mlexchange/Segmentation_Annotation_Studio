"""Tests for denoise_bake.py — real job-registry (export_jobs), fake array
source (same monkeypatch pattern as test_infer_jobs.py/test_denoise_train.py)
and a lightweight duck-typed fake Tiled container supporting compound
slash-separated __getitem__ (matching how client[target_path] is used
directly in run_denoise_bake_job)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import arrays as arrays_mod  # noqa: E402
import denoise as denoise_mod  # noqa: E402
import denoise_bake  # noqa: E402
import export_jobs  # noqa: E402
import schemas  # noqa: E402
import train_common  # noqa: E402


class FakeTiledContainer:
    def __init__(self):
        self._children: dict[str, "FakeTiledContainer"] = {}
        self.metadata: dict = {}
        self.written: list[dict] = []
        self.updated_metadata: dict | None = None

    def __getitem__(self, key):
        node = self
        for part in str(key).split("/"):
            node = node._children[part]
        return node

    def create_container(self, key, metadata):
        child = FakeTiledContainer()
        child.metadata = metadata
        self._children[key] = child
        return child

    def write_array(self, arr, key, dims=None, metadata=None):
        self.written.append({"key": key, "arr": arr, "dims": dims, "metadata": metadata})

    def update_metadata(self, metadata):
        self.updated_metadata = metadata


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch):
    client = FakeTiledContainer()
    client.create_container("browse", {})
    monkeypatch.setattr(denoise_bake, "get_tiled_client", lambda uri: client)
    return client


@pytest.fixture()
def fake_array_source(monkeypatch: pytest.MonkeyPatch):
    """A 4-slice volume of small 2D frames, distinguishable per slice."""
    frames = [np.full((6, 6), fill_value=float(i * 10), dtype=np.float32) for i in range(4)]

    monkeypatch.setattr(arrays_mod, "resolve_array", lambda source, kind, server_uri: "node")
    monkeypatch.setattr(arrays_mod, "array_shape_meta", lambda node: {"n_slices": len(frames)})
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: frames[idx])
    return frames


def _request(**overrides):
    defaults = dict(
        source="browse/sample", server_uri=None, method="median",
        strength=0.5, target_path="browse/sample_denoised", description="",
    )
    defaults.update(overrides)
    return schemas.DenoiseBakeRequest(**defaults)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestDefaultTargetPath:
    def test_appends_suffix_to_last_segment(self):
        assert denoise_bake.default_target_path("browse/sample") == "browse/sample_denoised"

    def test_custom_suffix(self):
        assert denoise_bake.default_target_path("browse/sample", suffix="clean") == "browse/sample_clean"

    def test_strips_whitespace_and_slashes(self):
        assert denoise_bake.default_target_path("  /browse/sample/  ") == "browse/sample_denoised"

    def test_empty_path_raises(self):
        with pytest.raises(ValueError):
            denoise_bake.default_target_path("   /// ")


class TestSliceKey:
    def test_zero_padded(self):
        assert denoise_bake._slice_key(7) == "slice_0007"

    def test_large_index(self):
        assert denoise_bake._slice_key(12345) == "slice_12345"


# ---------------------------------------------------------------------------
# run_denoise_bake_job — guard/validation paths
# ---------------------------------------------------------------------------

class TestGuardPaths:
    def test_method_none_reports_error(self):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="none"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "method" in job["error"].lower()

    def test_model_method_without_run_id_reports_error(self):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="model", run_id=None))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "run_id" in job["error"]

    def test_unavailable_method_reports_error(self):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="not_a_real_method"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "unavailable" in job["error"]

    def test_invalid_target_path_reports_error(self, fake_client):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(target_path="../escape"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"

    def test_existing_target_path_reports_error(self, fake_client):
        fake_client["browse"].create_container("sample_denoised", {})
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "already exists" in job["error"]

    def test_unexpected_exception_is_reported_not_raised(self, fake_client, monkeypatch):
        monkeypatch.setattr(
            arrays_mod, "resolve_array",
            lambda source, kind, server_uri: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request())
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "boom" in job["error"]


# ---------------------------------------------------------------------------
# run_denoise_bake_job — real classical-filter round trips
# ---------------------------------------------------------------------------

class TestClassicalBakeRoundTrip:
    def test_2d_method_writes_every_slice(self, fake_client, fake_array_source):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["n_slices"] == 4
        target = fake_client["browse"]["sample_denoised"]
        assert len(target.written) == 4
        assert target.updated_metadata["n_images"] == 4
        assert target.updated_metadata["denoise_method"] == "median"

    def test_3d_method_uses_z_window_and_stacks(self, fake_client, fake_array_source, monkeypatch):
        calls = []
        real_stack = denoise_mod.denoise_stack

        def spy_stack(stack, method, strength):
            calls.append(stack.shape[0])
            return real_stack(stack, method, strength)

        monkeypatch.setattr(denoise_mod, "denoise_stack", spy_stack)
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median3d"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        # radius=1: middle slices see a 3-frame window, edges see 2.
        assert calls == [2, 3, 3, 2]

    def test_per_slice_failure_falls_back_to_unfiltered_copy(self, fake_client, fake_array_source, monkeypatch):
        real_denoise_slice = denoise_mod.denoise_slice

        def flaky_denoise_slice(arr, method, strength):
            if arr[0, 0] == 10.0:  # slice index 1
                raise RuntimeError("filter blew up")
            return real_denoise_slice(arr, method, strength)

        monkeypatch.setattr(denoise_mod, "denoise_slice", flaky_denoise_slice)
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["n_slices"] == 4
        assert len(job["result"]["errors"]) == 1
        assert job["result"]["errors"][0]["slice"] == 1
        target = fake_client["browse"]["sample_denoised"]
        # The failed slice was still written (as an unfiltered copy of source).
        written_slice1 = next(w for w in target.written if w["key"] == "slice_0001")
        assert np.array_equal(written_slice1["arr"], fake_array_source[1])

    def test_totally_unreadable_slice_is_skipped_not_written(self, fake_client, fake_array_source, monkeypatch):
        monkeypatch.setattr(denoise_mod, "denoise_slice", lambda arr, m, s: (_ for _ in ()).throw(RuntimeError("f")))

        def flaky_read_slice(node, meta, idx):
            if idx == 2:
                raise RuntimeError("source also unreadable")
            return fake_array_source[idx]

        monkeypatch.setattr(arrays_mod, "read_slice", flaky_read_slice)
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "done"
        assert job["result"]["n_slices"] == 3  # slice 2 was skipped entirely
        slice2_error = next(e for e in job["result"]["errors"] if e["slice"] == 2)
        assert "also unreadable" in slice2_error["error"]

    def test_cancellation_mid_job_stops_early(self, fake_client, fake_array_source, monkeypatch):
        monkeypatch.setattr(export_jobs, "cancel_requested", lambda jid: True)
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "nothing was written" in job["error"]

    def test_all_slices_failing_reports_error(self, fake_client, fake_array_source, monkeypatch):
        monkeypatch.setattr(denoise_mod, "denoise_slice", lambda arr, m, s: (_ for _ in ()).throw(RuntimeError("f")))
        monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: (_ for _ in ()).throw(RuntimeError("g")))
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median"))
        job = export_jobs.get_job(jid)
        assert job["state"] == "error"
        assert "nothing was written" in job["error"]

    def test_description_and_keywords_propagate_to_metadata(self, fake_client, fake_array_source):
        jid = export_jobs.new_job("x")
        denoise_bake.run_denoise_bake_job(jid, _request(method="median", description="alpha, beta"))
        target = fake_client["browse"]["sample_denoised"]
        assert target.updated_metadata["description"] == "alpha, beta"
        assert set(target.updated_metadata["keywords"]) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# _ModelDenoiser — real forward pass via a minimal nn.Conv2d stand-in
# ---------------------------------------------------------------------------

class _FakeDenoiserRuntime:
    """Stand-in for denoise_runtime/autoencoder_runtime — generic over
    forward_fn/to_tensor_fn like the real modules, using a real nn.Conv2d."""

    @staticmethod
    def load_model(state, device):
        return torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    @staticmethod
    def make_forward_fn(model):
        def forward_fn(batch):
            return model(batch)
        return forward_fn

    @staticmethod
    def make_to_tensor_fn():
        def to_tensor_fn(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0) / 255.0
        return to_tensor_fn


@pytest.fixture()
def fake_model_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        train_common, "load_run_config",
        lambda run_id: {"model_family": "dlsia_denoiser", "task": "denoising", "image_size": 16, "render": {}},
    )
    monkeypatch.setattr(train_common, "denoiser_runtime_for", lambda config: _FakeDenoiserRuntime)
    monkeypatch.setattr(train_common, "denoiser_needs_dlsia", lambda config: False)
    monkeypatch.setattr(train_common, "dlsia_available", lambda: True)
    monkeypatch.setattr(train_common, "pick_device", lambda: "cpu")
    monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: {})
    # _ModelDenoiser.__init__ always computes global stats via arrays.read_slice
    # regardless of which guard path a test is exercising.
    monkeypatch.setattr(arrays_mod, "read_slice", lambda node, meta, idx: np.zeros((4, 4)))


class TestModelDenoiser:
    def test_rejects_run_that_is_not_a_denoiser(self, monkeypatch):
        monkeypatch.setattr(train_common, "load_run_config", lambda run_id: {"model_family": "dlsia", "task": "segmentation"})
        with pytest.raises(ValueError, match="not a denoiser"):
            denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})

    def test_missing_dlsia_when_needed_raises(self, fake_model_run, monkeypatch):
        monkeypatch.setattr(train_common, "denoiser_needs_dlsia", lambda config: True)
        monkeypatch.setattr(train_common, "dlsia_available", lambda: False)
        with pytest.raises(ValueError, match="dlsia is not installed"):
            denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})

    def test_missing_device_raises(self, fake_model_run, monkeypatch):
        monkeypatch.setattr(train_common, "pick_device", lambda: None)
        with pytest.raises(ValueError, match="torch is not installed"):
            denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})

    def test_busy_lock_raises_and_does_not_deadlock(self, fake_model_run):
        train_common.ML_LOCK.acquire()
        try:
            with pytest.raises(ValueError, match="busy"):
                denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})
        finally:
            train_common.ML_LOCK.release()

    def test_real_forward_pass_and_lock_lifecycle(self, fake_model_run, monkeypatch):
        gray = np.full((16, 16), 128, dtype=np.uint8)
        monkeypatch.setattr("denoise_train._slice_to_gray_uint8", lambda node, meta, idx, opts, gr: gray)
        monkeypatch.setattr("images._sample_global_stats", lambda node, meta: (0.0, 255.0))

        assert train_common.ML_LOCK.locked() is False
        md = denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})
        assert train_common.ML_LOCK.locked() is True
        try:
            out = md.denoise(0)
            assert out.shape == (16, 16)
            assert out.dtype == np.uint8
        finally:
            md.close()
        assert train_common.ML_LOCK.locked() is False
        # Idempotent: closing again must not raise or double-release.
        md.close()

    def test_load_failure_releases_lock(self, fake_model_run, monkeypatch):
        monkeypatch.setattr(train_common, "load_adapter_state", lambda run_id: (_ for _ in ()).throw(RuntimeError("nope")))
        with pytest.raises(RuntimeError, match="nope"):
            denoise_bake._ModelDenoiser("r1", node=object(), meta={"n_slices": 1})
        assert train_common.ML_LOCK.locked() is False
