"""Proxy routes for the disjoint ipred service (GUI never calls port 8003 directly).

All routes live under ``/api/ipred/*`` and are thin pass-throughs into
``ipred_client``, which talks to the standalone iPred FastAPI service over
HTTP. A connection failure surfaces as 503 rather than a 500/toast, so the
frontend can show a clear "not running" state.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import ipred_client as ipred_client_mod

router = APIRouter(prefix="/api/ipred")


class IpredSessionRequest(BaseModel):
    """Open an ipred session for one image/stack project."""

    kind: str
    source: str
    server_uri: Optional[str] = None
    root: Optional[str] = None


class IpredPreprocessRequest(BaseModel):
    """Proxy featurize request."""

    session_id: str
    feature_setup_id: Optional[str] = None
    composition_id: Optional[str] = None
    slice_index: int = 0
    array_ref: Optional[str] = None


class IpredCompositionUpsertRequest(BaseModel):
    """Create/update composition on ipred."""

    name: str
    nodes: list[dict[str, Any]]
    outputs: list[str]
    composition_id: Optional[str] = None
    builtin: bool = False


class IpredArrayUploadRequest(BaseModel):
    """Upload float array for remote ipred preprocess."""

    session_id: str
    shape: list[int]
    dtype: str = "float32"
    data_b64: str
    array_ref: Optional[str] = None


class IpredTrainRequest(BaseModel):
    """Proxy train request."""

    session_id: str
    shapes: list[dict[str, Any]]
    feature_id: Optional[str] = None
    trainer_id: str = "catboost"
    config: Optional[dict[str, Any]] = None


class IpredInferRequest(BaseModel):
    """Proxy infer request."""

    session_id: str
    model_id: Optional[str] = None
    feature_id: Optional[str] = None
    alpha: float = 0.05


class IpredRethresholdRequest(BaseModel):
    """Proxy rethreshold request."""

    session_id: str
    alpha: float
    run_id: Optional[str] = None


class IpredSetupUpsertRequest(BaseModel):
    """Create or update a Feature Setup on ipred."""

    name: str
    kind: str
    procedure_id: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    encoder_setup_id: Optional[str] = None
    weights_path: Optional[str] = None
    weights_format: Optional[str] = None
    inference: Optional[dict[str, Any]] = None
    setup_id: Optional[str] = None


class IpredManifoldSampleRequest(BaseModel):
    """Suggest Labels via ipred feature bank."""

    feature_id: str
    k: int = 24
    box_size: Optional[int] = None
    stride: Optional[int] = None
    pca_dims: int = 16
    shapes: Optional[list[dict[str, Any]]] = None


class IpredThresholdClassRequest(BaseModel):
    class_id: int
    threshold: float = 0.5


def _ipred_http_error(exc: Exception) -> HTTPException:
    """Translate an ipred_client exception into the right HTTP response."""
    if isinstance(exc, httpx.HTTPStatusError):
        detail: Any = exc.response.text
        try:
            detail = exc.response.json()
        except Exception:
            pass
        return HTTPException(exc.response.status_code, detail)
    if isinstance(exc, httpx.ConnectError):
        return HTTPException(503, f"ipred unreachable at {ipred_client_mod.ipred_url()}")
    return HTTPException(500, str(exc))


@router.get("/health")
async def ipred_health() -> dict:
    """Liveness of the disjoint ipred."""
    try:
        return ipred_client_mod.health()
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/sessions")
async def ipred_open_session(body: IpredSessionRequest) -> dict:
    """Open/create an engine session for a project."""
    try:
        return ipred_client_mod.open_session(
            kind=body.kind,
            source=body.source,
            server_uri=body.server_uri,
            root=body.root,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/setups")
async def ipred_list_setups() -> dict:
    """List Feature Setups from the ipred."""
    try:
        return {"setups": ipred_client_mod.list_setups()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/setups/{setup_id}")
async def ipred_get_setup(setup_id: str) -> dict:
    """Get one Feature Setup from ipred."""
    try:
        return ipred_client_mod.get_setup(setup_id)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/setups")
async def ipred_upsert_setup(body: IpredSetupUpsertRequest) -> dict:
    """Create or update a Feature Setup on ipred."""
    try:
        return ipred_client_mod.upsert_setup(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/trainers")
async def ipred_list_trainers() -> dict:
    """List trainer plugin ids from ipred."""
    try:
        return {"trainers": ipred_client_mod.list_trainers()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/modules")
async def ipred_list_modules() -> dict:
    """List composable feature modules (each reports `runtime` + `ready`)."""
    try:
        return {"modules": ipred_client_mod.list_modules()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/compositions")
async def ipred_list_compositions() -> dict:
    """List feature compositions."""
    try:
        return {"compositions": ipred_client_mod.list_compositions()}
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/compositions/{composition_id}")
async def ipred_get_composition(composition_id: str) -> dict:
    """Get one composition."""
    try:
        return ipred_client_mod.get_composition(composition_id)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/compositions")
async def ipred_upsert_composition(body: IpredCompositionUpsertRequest) -> dict:
    """Create or update a composition."""
    try:
        return ipred_client_mod.upsert_composition(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/compositions/preview")
async def ipred_preview_composition(body: IpredCompositionUpsertRequest) -> dict:
    """Preview concat labels for a composition draft."""
    try:
        return ipred_client_mod.preview_composition(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/sessions/{session_id}/arrays")
async def ipred_upload_array(session_id: str, body: IpredArrayUploadRequest) -> dict:
    """Upload an array blob to ipred for remote preprocess."""
    try:
        payload = body.model_dump(exclude_none=True)
        payload["session_id"] = session_id
        return ipred_client_mod.upload_session_array(session_id, payload)
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/manifold/sample")
async def ipred_manifold_sample(body: IpredManifoldSampleRequest) -> dict:
    """Suggest Labels on an ipred feature bank."""
    try:
        return ipred_client_mod.manifold_sample(body.model_dump(exclude_none=True))
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/manifold/{sample_id}/heatmap.png")
async def ipred_manifold_heatmap(sample_id: str) -> Response:
    """Proxy manifold residual heatmap PNG."""
    try:
        return Response(
            content=ipred_client_mod.manifold_heatmap_png(sample_id),
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/preprocess")
async def ipred_preprocess(body: IpredPreprocessRequest) -> dict:
    """Cache-aware featurize via ipred."""
    try:
        return ipred_client_mod.preprocess(
            session_id=body.session_id,
            feature_setup_id=body.feature_setup_id,
            composition_id=body.composition_id,
            slice_index=body.slice_index,
            array_ref=body.array_ref,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/features/{feature_id}/channels/{index}")
async def ipred_feature_channel(feature_id: str, index: int) -> Response:
    """Proxy a feature-channel PNG from the ipred."""
    try:
        data = ipred_client_mod.feature_channel_bytes(feature_id, index)
        return Response(content=data, media_type="image/png")
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/train")
async def ipred_train(body: IpredTrainRequest) -> dict:
    """Train via ipred trainer plugin."""
    try:
        return ipred_client_mod.train(
            session_id=body.session_id,
            shapes=body.shapes,
            feature_id=body.feature_id,
            trainer_id=body.trainer_id,
            config=body.config,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/infer")
async def ipred_infer(body: IpredInferRequest) -> dict:
    """Infer + conformal products via ipred."""
    try:
        return ipred_client_mod.infer(
            session_id=body.session_id,
            model_id=body.model_id,
            feature_id=body.feature_id,
            alpha=body.alpha,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/rethreshold")
async def ipred_rethreshold(body: IpredRethresholdRequest) -> dict:
    """Rethreshold from cached proba via ipred."""
    try:
        return ipred_client_mod.rethreshold(
            session_id=body.session_id,
            alpha=body.alpha,
            run_id=body.run_id,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/runs/{run_id}/commit.png")
async def ipred_run_commit(run_id: str) -> Response:
    """Proxy commit map PNG."""
    try:
        return Response(
            content=ipred_client_mod.run_commit_png(run_id),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/runs/{run_id}/status.png")
async def ipred_run_status(run_id: str) -> Response:
    """Proxy status map PNG."""
    try:
        return Response(
            content=ipred_client_mod.run_status_png(run_id),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.get("/runs/{run_id}/proba/{class_index}.png")
async def ipred_run_proba(run_id: str, class_index: int) -> Response:
    """Proxy softmax class heatmap PNG from ipred."""
    try:
        return Response(
            content=ipred_client_mod.run_proba_png(run_id, class_index),
            media_type="image/png",
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc


@router.post("/runs/{run_id}/threshold-class")
async def ipred_threshold_class(run_id: str, body: IpredThresholdClassRequest) -> dict:
    """Threshold one softmax class into a dense label map via ipred."""
    try:
        return ipred_client_mod.threshold_class_map(
            run_id,
            class_id=body.class_id,
            threshold=body.threshold,
        )
    except Exception as exc:
        raise _ipred_http_error(exc) from exc
