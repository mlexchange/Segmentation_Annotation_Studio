"""HTTP client for the disjoint ipred backend (no Python imports of ``ipred``)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

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


def new_shared_client(timeout: float = 300.0) -> httpx.Client:
    """A `httpx.Client` a caller owns and reuses across many calls (e.g. one
    per slice in a volume-wide batch-apply job), instead of the fresh
    open/close-per-call `_client()` every other function here defaults to.
    `httpx.Client` is documented safe for concurrent use from multiple
    threads, so this is also what a thread-pooled batch job should share.
    Caller is responsible for closing it (a `with` block or `.close()`).
    """
    return httpx.Client(base_url=ipred_url(), timeout=timeout)


@contextmanager
def _use_client(client: httpx.Client | None, timeout: float = 300.0) -> Iterator[httpx.Client]:
    """Yield `client` if given (never closing it — the caller owns its
    lifecycle), else open-and-close a fresh one exactly like every call site
    here did before `client=` params existed. Keeps every function's
    single-call default behavior unchanged for callers that don't pass one."""
    if client is not None:
        yield client
        return
    with _client(timeout=timeout) as owned:
        yield owned


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
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST /preprocess. Pass `client` (see `new_shared_client`) to reuse a
    connection across many calls instead of opening a fresh one each time."""
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
    with _use_client(client) as c:
        r = c.post("/preprocess", json=body)
        r.raise_for_status()
        return r.json()


def delete_feature_bank(feature_id: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """DELETE /features/{id} — release a feature bank's disk blob once nothing
    later in the current job needs it (see ipred_batch_jobs.py's volume-apply
    job, the only caller)."""
    with _use_client(client) as c:
        r = c.delete(f"/features/{feature_id}")
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
    store_probabilities: bool = True,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST /infer. `store_probabilities=False` skips persisting the run's
    full probability array — see `train_infer.run_infer`'s doc for why a
    volume-wide batch-apply job (the only caller that passes `False`) never
    needs it. Pass `client` (see `new_shared_client`) to reuse a connection
    across many calls instead of opening a fresh one each time."""
    with _use_client(client) as c:
        r = c.post(
            "/infer",
            json={
                "session_id": session_id,
                "model_id": model_id,
                "feature_id": feature_id,
                "alpha": alpha,
                "store_probabilities": store_probabilities,
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
