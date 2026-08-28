"""Multi-slice training pools labeled pixels across slices into one model."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from ipred import feature_setups, preprocess, train_infer
from ipred.catalog import Catalog


@pytest.fixture()
def catalog(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Catalog:
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return Catalog(tmp_path / "ipred" / "catalog.db")


def _make_stack(tmp_path, n=3, size=48):
    stack = np.zeros((n, size, size), dtype=np.uint8)
    for i in range(n):
        stack[i, :, size // 2:] = 200
    tifffile.imwrite(tmp_path / "stack.tif", stack, photometric="minisblack")
    return "stack.tif"


def test_train_multi_slice_pools_across_slices(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    source = _make_stack(tmp_path, n=3, size=48)
    session = catalog.open_session(kind="local", source=source, root=str(tmp_path))

    per_slice_shapes: dict[int, list[dict]] = {}
    feature_ids: dict[int, str] = {}
    for slice_index in (0, 1, 2):
        bank = preprocess.run_preprocess(
            catalog,
            session_id=session.session_id,
            feature_setup_id="default-skimage",
            slice_index=slice_index,
        )
        feature_ids[slice_index] = bank["feature_id"]
        per_slice_shapes[slice_index] = [
            {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 18, "h": 40},
            {"kind": "rectangle", "classId": 2, "x": 28, "y": 2, "w": 18, "h": 40},
        ]

    trained = train_infer.run_train_multi_slice(
        catalog,
        session_id=session.session_id,
        per_slice_shapes=per_slice_shapes,
        feature_ids=feature_ids,
        trainer_id="catboost",
        config={"iterations": 20, "depth": 4, "random_seed": 0},
    )
    assert trained["model_id"]
    assert trained["class_ids"] == [1, 2]
    assert trained["trained_slice_indices"] == [0, 1, 2]
    # Pooled across 3 slices should see roughly 3x the pixels of one slice alone.
    assert trained["n_samples"] > 1000

    # The model is usable for ordinary single-slice infer against any one slice.
    run = train_infer.run_infer(
        catalog,
        session_id=session.session_id,
        model_id=trained["model_id"],
        feature_id=feature_ids[1],
        alpha=0.2,
    )
    assert run["run_id"]
    assert run["class_ids"] == [1, 2]


def test_train_multi_slice_requires_two_classes_pooled(catalog: Catalog, tmp_path) -> None:
    feature_setups.ensure_default_setups(catalog)
    source = _make_stack(tmp_path, n=2, size=48)
    session = catalog.open_session(kind="local", source=source, root=str(tmp_path))

    per_slice_shapes: dict[int, list[dict]] = {}
    feature_ids: dict[int, str] = {}
    for slice_index in (0, 1):
        bank = preprocess.run_preprocess(
            catalog,
            session_id=session.session_id,
            feature_setup_id="default-skimage",
            slice_index=slice_index,
        )
        feature_ids[slice_index] = bank["feature_id"]
        # Only class 1 labeled on every slice — never two classes, even pooled.
        per_slice_shapes[slice_index] = [
            {"kind": "rectangle", "classId": 1, "x": 2, "y": 2, "w": 18, "h": 40},
        ]

    with pytest.raises(ValueError, match="need at least two classes"):
        train_infer.run_train_multi_slice(
            catalog,
            session_id=session.session_id,
            per_slice_shapes=per_slice_shapes,
            feature_ids=feature_ids,
            trainer_id="catboost",
            config={},
        )
