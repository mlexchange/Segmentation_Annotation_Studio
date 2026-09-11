"""FastAPI app for the ipred iterative prediction backend."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ipred import compositions, feature_setups, manifold_jobs, preprocess, train_infer
from ipred.catalog import Catalog
from ipred.modules import list_module_catalog
from ipred.trainers import list_trainers

logger = logging.getLogger(__name__)

_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    """Return process-wide catalog."""
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
        feature_setups.ensure_default_setups(_catalog)
        compositions.ensure_default_compositions(_catalog)
    return _catalog


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Ensure defaults on startup."""
    get_catalog()
    yield


app = FastAPI(title="ipred", version="0.1.0", lifespan=lifespan)


class ProjectIdentity(BaseModel):
    """Source identity for a project / session."""

    kind: str
    source: str
    server_uri: Optional[str] = None
    root: Optional[str] = None


class PreprocessBody(BaseModel):
    """Featurize request."""

    session_id: str
    feature_setup_id: Optional[str] = None
    composition_id: Optional[str] = None
    slice_index: int = 0
    array_ref: Optional[str] = None


class CompositionUpsertBody(BaseModel):
    """Create/update a composition document."""

    name: str
    nodes: list[dict[str, Any]]
    outputs: list[str]
    composition_id: Optional[str] = None
    builtin: bool = False


class ArrayUploadBody(BaseModel):
    """Upload a float array (base64 raw float32) for remote preprocess."""

    session_id: str
    shape: list[int]
    dtype: str = "float32"
    data_b64: str
    array_ref: Optional[str] = None


class TrainBody(BaseModel):
    """Train request."""

    session_id: str
    shapes: list[dict[str, Any]]
    feature_id: Optional[str] = None
    trainer_id: str = "catboost"
    config: dict[str, Any] = Field(default_factory=dict)


class TrainMultiBody(BaseModel):
    """Multi-slice train request — pools labeled pixels across slices."""

    session_id: str
    slices: dict[str, list[dict[str, Any]]]
    feature_ids: dict[str, str]
    trainer_id: str = "catboost"
    config: dict[str, Any] = Field(default_factory=dict)


class InferBody(BaseModel):
    """Infer request."""

    session_id: str
    model_id: Optional[str] = None
    feature_id: Optional[str] = None
    alpha: float = 0.05
    store_probabilities: bool = True


class RethresholdBody(BaseModel):
    """Rethreshold request."""

    session_id: str
    alpha: float
    run_id: Optional[str] = None


class SetupUpsertBody(BaseModel):
    """Create/update a Feature Setup."""

    name: str
    kind: str
    procedure_id: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    encoder_setup_id: Optional[str] = None
    weights_path: Optional[str] = None
    weights_format: Optional[str] = None
    inference: Optional[dict[str, Any]] = None
    setup_id: Optional[str] = None


class ManifoldSampleBody(BaseModel):
    """Suggest Labels sample request."""

    feature_id: str
    k: int = 24
    box_size: Optional[int] = None
    stride: Optional[int] = None
    pca_dims: int = 16
    shapes: Optional[list[dict[str, Any]]] = None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "ipred"}


@app.post("/sessions")
def open_session(body: ProjectIdentity) -> dict[str, Any]:
    """Open a session for a project (creates project if needed)."""
    cat = get_catalog()
    session = cat.open_session(
        kind=body.kind,
        source=body.source,
        server_uri=body.server_uri,
        root=body.root,
    )
    return {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "current_feature_id": session.current_feature_id,
        "current_model_id": session.current_model_id,
        "current_run_id": session.current_run_id,
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Fetch session state."""
    session = get_catalog().get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    project = get_catalog().get_project(session.project_id)
    return {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "project": {
            "kind": project.kind if project else None,
            "source": project.source if project else None,
            "server_uri": project.server_uri if project else None,
            "root": project.root if project else None,
        },
        "current_feature_id": session.current_feature_id,
        "current_model_id": session.current_model_id,
        "current_run_id": session.current_run_id,
    }


@app.get("/setups")
def api_list_setups() -> dict[str, Any]:
    """List Feature Setups (legacy) + ensure compositions exist."""
    feature_setups.ensure_default_setups(get_catalog())
    compositions.ensure_default_compositions(get_catalog())
    return {"setups": feature_setups.list_setups()}


@app.get("/modules")
def api_list_modules() -> dict[str, Any]:
    """List composable feature modules."""
    return {"modules": list_module_catalog()}


@app.get("/compositions")
def api_list_compositions() -> dict[str, Any]:
    """List composition documents."""
    compositions.ensure_default_compositions(get_catalog())
    return {"compositions": compositions.list_compositions()}


@app.get("/compositions/{composition_id}")
def api_get_composition(composition_id: str) -> dict[str, Any]:
    """Get one composition (+ concat preview labels)."""
    try:
        doc = compositions.resolve_composition(composition_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    doc = dict(doc)
    doc["preview_labels"] = compositions.preview_concat_labels(doc)
    return doc


@app.post("/compositions")
def api_upsert_composition(body: CompositionUpsertBody) -> dict[str, Any]:
    """Create or update a composition."""
    try:
        return compositions.save_composition(
            name=body.name,
            nodes=body.nodes,
            outputs=body.outputs,
            composition_id=body.composition_id,
            builtin=body.builtin,
            catalog=get_catalog(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/compositions/preview")
def api_preview_composition(body: CompositionUpsertBody) -> dict[str, Any]:
    """Return concat labels without saving."""
    try:
        labels = compositions.preview_concat_labels(
            {"nodes": body.nodes, "outputs": body.outputs}
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"preview_labels": labels}


@app.post("/sessions/{session_id}/arrays")
def api_upload_array(session_id: str, body: ArrayUploadBody) -> dict[str, Any]:
    """Upload a content-addressed array blob for remote preprocess."""
    import base64

    import numpy as np

    from ipred import array_blobs

    session = get_catalog().get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if body.session_id and body.session_id != session_id:
        raise HTTPException(422, "session_id mismatch")
    try:
        raw = base64.b64decode(body.data_b64)
        shape = tuple(int(x) for x in body.shape)
        arr = np.frombuffer(raw, dtype=np.dtype(body.dtype)).reshape(shape)
    except Exception as exc:
        raise HTTPException(422, f"invalid array payload: {exc}") from exc
    return array_blobs.save_array_blob(
        session.project_id,
        arr,
        array_ref=body.array_ref,
    )


@app.get("/setups/{setup_id}")
def api_get_setup(setup_id: str) -> dict[str, Any]:
    """Get one Feature Setup."""
    try:
        return feature_setups.resolve_setup(setup_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/setups")
def api_upsert_setup(body: SetupUpsertBody) -> dict[str, Any]:
    """Create or update a Feature Setup."""
    try:
        return feature_setups.save_setup(
            name=body.name,
            kind=body.kind,
            procedure_id=body.procedure_id,
            params=body.params,
            encoder_setup_id=body.encoder_setup_id,
            weights_path=body.weights_path,
            weights_format=body.weights_format,
            inference=body.inference,
            setup_id=body.setup_id,
            catalog=get_catalog(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/trainers")
def api_list_trainers() -> dict[str, Any]:
    """List trainer plugin ids."""
    return {"trainers": list_trainers()}


@app.post("/preprocess")
def api_preprocess(body: PreprocessBody) -> dict[str, Any]:
    """Cache-aware featurize (composition_id preferred; setup_id still accepted)."""
    try:
        return preprocess.run_preprocess(
            get_catalog(),
            session_id=body.session_id,
            feature_setup_id=body.feature_setup_id,
            composition_id=body.composition_id,
            slice_index=body.slice_index,
            array_ref=body.array_ref,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("preprocess failed")
        raise HTTPException(500, f"preprocess failed: {exc}") from exc


@app.get("/features/{feature_id}/channels/{index}")
def api_feature_channel(feature_id: str, index: int) -> Response:
    """Return a channel PNG for a feature bank."""
    row = get_catalog().get_feature_bank(feature_id)
    if row is None:
        raise HTTPException(404, "feature bank not found")
    path = preprocess.channel_png_path(row["blob_dir"], index)
    if not path.is_file():
        raise HTTPException(404, "channel not found")
    return FileResponse(path, media_type="image/png")


@app.delete("/features/{feature_id}")
def api_delete_feature_bank(feature_id: str) -> dict[str, Any]:
    """Delete a feature bank (DB row + its on-disk blob dir).

    For batch-apply jobs (Phase 4.5's "Apply across volume") to release each
    slice's feature bank immediately after it's used — see
    `Catalog.delete_feature_bank`'s doc for why this exists. Deleting an
    already-gone or unknown feature_id is a no-op, not an error: cleanup
    calls are best-effort and must never fail the job that triggered them.
    """
    blob_dir = get_catalog().delete_feature_bank(feature_id)
    if blob_dir:
        shutil.rmtree(blob_dir, ignore_errors=True)
    return {"deleted": blob_dir is not None}


@app.post("/train")
def api_train(body: TrainBody) -> dict[str, Any]:
    """Train via trainer plugin."""
    try:
        return train_infer.run_train(
            get_catalog(),
            session_id=body.session_id,
            shapes=body.shapes,
            feature_id=body.feature_id,
            trainer_id=body.trainer_id,
            config=body.config,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("train failed")
        raise HTTPException(500, f"train failed: {exc}") from exc


@app.post("/train/multi")
def api_train_multi(body: TrainMultiBody) -> dict[str, Any]:
    """Train one model pooling labeled pixels across multiple slices."""
    try:
        return train_infer.run_train_multi_slice(
            get_catalog(),
            session_id=body.session_id,
            per_slice_shapes={int(k): v for k, v in body.slices.items()},
            feature_ids={int(k): v for k, v in body.feature_ids.items()},
            trainer_id=body.trainer_id,
            config=body.config,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("multi-slice train failed")
        raise HTTPException(500, f"multi-slice train failed: {exc}") from exc


@app.post("/infer")
def api_infer(body: InferBody) -> dict[str, Any]:
    """Infer + conformal products."""
    try:
        return train_infer.run_infer(
            get_catalog(),
            session_id=body.session_id,
            model_id=body.model_id,
            feature_id=body.feature_id,
            alpha=body.alpha,
            store_probabilities=body.store_probabilities,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("infer failed")
        raise HTTPException(500, f"infer failed: {exc}") from exc


@app.post("/rethreshold")
def api_rethreshold(body: RethresholdBody) -> dict[str, Any]:
    """Rethreshold from cached proba."""
    try:
        return train_infer.run_rethreshold(
            get_catalog(),
            session_id=body.session_id,
            run_id=body.run_id,
            alpha=body.alpha,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("rethreshold failed")
        raise HTTPException(500, f"rethreshold failed: {exc}") from exc


@app.get("/runs/{run_id}/commit.png")
def api_run_commit(run_id: str) -> Response:
    """Commit map PNG."""
    run = get_catalog().get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    path = Path(run["blob_dir"]) / "commit.png"
    if not path.is_file():
        raise HTTPException(404, "commit.png missing")
    return FileResponse(path, media_type="image/png")


@app.get("/runs/{run_id}/status.png")
def api_run_status(run_id: str) -> Response:
    """Status map PNG."""
    run = get_catalog().get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    path = Path(run["blob_dir"]) / "status.png"
    if not path.is_file():
        raise HTTPException(404, "status.png missing")
    return FileResponse(path, media_type="image/png")


@app.get("/runs/{run_id}/proba/{class_index}.png")
def api_run_proba_channel(run_id: str, class_index: int) -> Response:
    """Softmax probability heatmap PNG for one class column (0-based index)."""
    try:
        data = train_infer.proba_heatmap_png(get_catalog(), run_id, class_index)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(content=data, media_type="image/png")


class ThresholdClassBody(BaseModel):
    class_id: int
    threshold: float = 0.5


@app.post("/runs/{run_id}/threshold-class")
def api_threshold_class(run_id: str, body: ThresholdClassBody) -> dict[str, Any]:
    """Threshold one softmax class into a dense label map (for mask-set caches)."""
    try:
        return train_infer.threshold_class_label_map(
            get_catalog(),
            run_id,
            class_id=body.class_id,
            threshold=body.threshold,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/runs/{run_id}")
def api_get_run(run_id: str) -> dict[str, Any]:
    """Run metadata."""
    run = get_catalog().get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    import json

    meta = json.loads(run["meta_json"])
    return {**meta, "blob_dir": run["blob_dir"], "alpha": run["alpha"]}


@app.post("/manifold/sample")
def api_manifold_sample(body: ManifoldSampleBody) -> dict[str, Any]:
    """Suggest Labels: variance boxes + residual heatmap on a feature bank."""
    try:
        return manifold_jobs.run_manifold_sample(
            get_catalog(),
            feature_id=body.feature_id,
            k=body.k,
            box_size=body.box_size,
            stride=body.stride,
            pca_dims=body.pca_dims,
            shapes=body.shapes,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("manifold sample failed")
        raise HTTPException(500, f"manifold sample failed: {exc}") from exc


@app.get("/manifold/{sample_id}/heatmap.png")
def api_manifold_heatmap(sample_id: str) -> Response:
    """Residual heatmap PNG for a manifold sample."""
    try:
        png = manifold_jobs.heatmap_png(sample_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )
