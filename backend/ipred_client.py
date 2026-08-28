"""HTTP client for the disjoint ipred backend (no Python imports of ``ipred``)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_IPRED_URL = "http://127.0.0.1:8003"


def ipred_url() -> str:
    """Base URL for the iterative prediction backend."""
    # Prefer IPRED_URL; accept legacy CLF_ENGINE_URL during transition.
    return os.getenv(
        "IPRED_URL",
        os.getenv("CLF_ENGINE_URL", DEFAULT_IPRED_URL),
    ).rstrip("/")


def _client(timeout: float = 300.0) -> httpx.Client:
    return httpx.Client(base_url=ipred_url(), timeout=timeout)


def health() -> dict[str, Any]:
    """GET /health."""
    with _client(timeout=5.0) as client:
        r = client.get("/health")
        r.raise_for_status()
        return r.json()


def open_session(
    *,
    kind: str,
    source: str,
    server_uri: str | None = None,
    root: str | None = None,
) -> dict[str, Any]:
    """POST /sessions."""
    with _client() as client:
        r = client.post(
            "/sessions",
            json={
                "kind": kind,
                "source": source,
                "server_uri": server_uri,
                "root": root,
            },
        )
        r.raise_for_status()
        return r.json()


def list_setups() -> list[dict[str, Any]]:
    """GET /setups."""
    with _client() as client:
        r = client.get("/setups")
        r.raise_for_status()
        return list(r.json().get("setups") or [])


def get_setup(setup_id: str) -> dict[str, Any]:
    """GET /setups/{id}."""
    with _client() as client:
        r = client.get(f"/setups/{setup_id}")
        r.raise_for_status()
        return r.json()


def upsert_setup(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /setups."""
    with _client() as client:
        r = client.post("/setups", json=payload)
        r.raise_for_status()
        return r.json()


def list_trainers() -> list[str]:
    """GET /trainers."""
    with _client() as client:
        r = client.get("/trainers")
        r.raise_for_status()
        return list(r.json().get("trainers") or [])


def list_modules() -> list[dict[str, Any]]:
    """GET /modules."""
    with _client() as client:
        r = client.get("/modules")
        r.raise_for_status()
        return list(r.json().get("modules") or [])


def list_compositions() -> list[dict[str, Any]]:
    """GET /compositions."""
    with _client() as client:
        r = client.get("/compositions")
        r.raise_for_status()
        return list(r.json().get("compositions") or [])


def get_composition(composition_id: str) -> dict[str, Any]:
    """GET /compositions/{id}."""
    with _client() as client:
        r = client.get(f"/compositions/{composition_id}")
        r.raise_for_status()
        return r.json()


def upsert_composition(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /compositions."""
    with _client() as client:
        r = client.post("/compositions", json=payload)
        r.raise_for_status()
        return r.json()


def preview_composition(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /compositions/preview."""
    with _client() as client:
        r = client.post("/compositions/preview", json=payload)
        r.raise_for_status()
        return r.json()


def upload_session_array(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /sessions/{id}/arrays."""
    with _client() as client:
        r = client.post(f"/sessions/{session_id}/arrays", json=payload)
        r.raise_for_status()
        return r.json()


def preprocess(
    *,
    session_id: str,
    feature_setup_id: str | None = None,
    composition_id: str | None = None,
    slice_index: int = 0,
    array_ref: str | None = None,
) -> dict[str, Any]:
    """POST /preprocess."""
    body: dict[str, Any] = {
        "session_id": session_id,
        "slice_index": slice_index,
    }
    if composition_id:
        body["composition_id"] = composition_id
    if feature_setup_id:
        body["feature_setup_id"] = feature_setup_id
    if array_ref:
        body["array_ref"] = array_ref
    with _client() as client:
        r = client.post("/preprocess", json=body)
        r.raise_for_status()
        return r.json()


def feature_channel_bytes(feature_id: str, index: int) -> bytes:
    """GET /features/{id}/channels/{index}."""
    with _client() as client:
        r = client.get(f"/features/{feature_id}/channels/{index}")
        r.raise_for_status()
        return r.content


def train(
    *,
    session_id: str,
    shapes: list[dict[str, Any]],
    feature_id: str | None = None,
    trainer_id: str = "catboost",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /train."""
    with _client() as client:
        r = client.post(
            "/train",
            json={
                "session_id": session_id,
                "shapes": shapes,
                "feature_id": feature_id,
                "trainer_id": trainer_id,
                "config": config or {},
            },
        )
        r.raise_for_status()
        return r.json()


def train_multi(
    *,
    session_id: str,
    slices: dict[str, list[dict[str, Any]]],
    feature_ids: dict[str, str],
    trainer_id: str = "catboost",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /train/multi — pool labeled pixels across multiple slices into one model."""
    with _client() as client:
        r = client.post(
            "/train/multi",
            json={
                "session_id": session_id,
                "slices": slices,
                "feature_ids": feature_ids,
                "trainer_id": trainer_id,
                "config": config or {},
            },
        )
        r.raise_for_status()
        return r.json()


def infer(
    *,
    session_id: str,
    model_id: str | None = None,
    feature_id: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """POST /infer."""
    with _client() as client:
        r = client.post(
            "/infer",
            json={
                "session_id": session_id,
                "model_id": model_id,
                "feature_id": feature_id,
                "alpha": alpha,
            },
        )
        r.raise_for_status()
        return r.json()


def rethreshold(
    *,
    session_id: str,
    alpha: float,
    run_id: str | None = None,
) -> dict[str, Any]:
    """POST /rethreshold."""
    with _client() as client:
        r = client.post(
            "/rethreshold",
            json={
                "session_id": session_id,
                "alpha": alpha,
                "run_id": run_id,
            },
        )
        r.raise_for_status()
        return r.json()


def run_commit_png(run_id: str) -> bytes:
    """GET /runs/{id}/commit.png."""
    with _client() as client:
        r = client.get(f"/runs/{run_id}/commit.png")
        r.raise_for_status()
        return r.content


def run_status_png(run_id: str) -> bytes:
    """GET /runs/{id}/status.png."""
    with _client() as client:
        r = client.get(f"/runs/{run_id}/status.png")
        r.raise_for_status()
        return r.content


def run_proba_png(run_id: str, class_index: int) -> bytes:
    """GET /runs/{id}/proba/{class_index}.png."""
    with _client() as client:
        r = client.get(f"/runs/{run_id}/proba/{int(class_index)}.png")
        r.raise_for_status()
        return r.content


def threshold_class_map(
    run_id: str,
    *,
    class_id: int,
    threshold: float,
) -> dict[str, Any]:
    """POST /runs/{id}/threshold-class."""
    with _client() as client:
        r = client.post(
            f"/runs/{run_id}/threshold-class",
            json={"class_id": int(class_id), "threshold": float(threshold)},
        )
        r.raise_for_status()
        return r.json()


def manifold_sample(payload: dict[str, Any]) -> dict[str, Any]:
    """POST /manifold/sample."""
    with _client() as client:
        r = client.post("/manifold/sample", json=payload)
        r.raise_for_status()
        return r.json()


def manifold_heatmap_png(sample_id: str) -> bytes:
    """GET /manifold/{id}/heatmap.png."""
    with _client() as client:
        r = client.get(f"/manifold/{sample_id}/heatmap.png")
        r.raise_for_status()
        return r.content
