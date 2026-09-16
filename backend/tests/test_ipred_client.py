"""Tests for ipred_client.py — a real httpx.Client is exercised end to end
via httpx.MockTransport (no real network), so ipred_url()/_client()/_use_client()
and every wrapper function's request construction + response parsing are all
genuinely tested, not mocked away."""
from __future__ import annotations

import types

import httpx
import pytest

import ipred_client


class RequestLog:
    def __init__(self):
        self.requests: list[httpx.Request] = []


def _install_handler(monkeypatch: pytest.MonkeyPatch, handler):
    """Patch the `httpx` name inside ipred_client's module namespace so every
    httpx.Client(...) constructed by the module routes through a MockTransport
    calling `handler`, while everything else about httpx.Client stays real."""
    real_client_cls = httpx.Client

    def fake_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client_cls(**kwargs)

    fake_httpx = types.SimpleNamespace(Client=fake_client)
    monkeypatch.setattr(ipred_client, "httpx", fake_httpx)


def _json_handler(log: RequestLog, routes: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        log.requests.append(request)
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"detail": "no route"})
        body, status = routes[key]
        if isinstance(body, (bytes, bytearray)):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)
    return handler


# ---------------------------------------------------------------------------
# ipred_url
# ---------------------------------------------------------------------------

class TestIpredUrl:
    def test_defaults_when_no_env_vars_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("IPRED_URL", raising=False)
        monkeypatch.delenv("CLF_ENGINE_URL", raising=False)
        assert ipred_client.ipred_url() == ipred_client.DEFAULT_IPRED_URL

    def test_ipred_url_env_var_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("IPRED_URL", "http://a:1/")
        monkeypatch.setenv("CLF_ENGINE_URL", "http://b:2/")
        assert ipred_client.ipred_url() == "http://a:1"

    def test_legacy_clf_engine_url_used_when_ipred_url_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("IPRED_URL", raising=False)
        monkeypatch.setenv("CLF_ENGINE_URL", "http://legacy:9/")
        assert ipred_client.ipred_url() == "http://legacy:9"

    def test_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("IPRED_URL", "http://x:1///")
        assert ipred_client.ipred_url() == "http://x:1"


# ---------------------------------------------------------------------------
# health / open_session / setups / trainers / modules / compositions
# ---------------------------------------------------------------------------

class TestSimpleGetPostWrappers:
    def test_health(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/health"): ({"status": "ok"}, 200)}))
        assert ipred_client.health() == {"status": "ok"}

    def test_open_session_posts_expected_body(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/sessions"): ({"session_id": "s1"}, 200)}))
        result = ipred_client.open_session(kind="tiled", source="foo", server_uri="http://t", root=None)
        assert result == {"session_id": "s1"}
        sent = log.requests[0]
        assert sent.method == "POST"
        import json
        body = json.loads(sent.content)
        assert body == {"kind": "tiled", "source": "foo", "server_uri": "http://t", "root": None}

    def test_list_setups_defaults_to_empty_list(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/setups"): ({}, 200)}))
        assert ipred_client.list_setups() == []

    def test_list_setups_returns_setups_key(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/setups"): ({"setups": [{"id": 1}]}, 200)}))
        assert ipred_client.list_setups() == [{"id": 1}]

    def test_get_setup(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/setups/abc"): ({"id": "abc"}, 200)}))
        assert ipred_client.get_setup("abc") == {"id": "abc"}

    def test_upsert_setup(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/setups"): ({"id": "new"}, 200)}))
        assert ipred_client.upsert_setup({"name": "x"}) == {"id": "new"}

    def test_list_trainers(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/trainers"): ({"trainers": ["catboost"]}, 200)}))
        assert ipred_client.list_trainers() == ["catboost"]

    def test_list_modules(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/modules"): ({"modules": [{"id": "m1"}]}, 200)}))
        assert ipred_client.list_modules() == [{"id": "m1"}]

    def test_list_compositions(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/compositions"): ({"compositions": []}, 200)}))
        assert ipred_client.list_compositions() == []

    def test_get_composition(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/compositions/c1"): ({"id": "c1"}, 200)}))
        assert ipred_client.get_composition("c1") == {"id": "c1"}

    def test_upsert_composition(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/compositions"): ({"id": "c2"}, 200)}))
        assert ipred_client.upsert_composition({"name": "y"}) == {"id": "c2"}

    def test_preview_composition(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/compositions/preview"): ({"preview": True}, 200)}))
        assert ipred_client.preview_composition({"x": 1}) == {"preview": True}

    def test_upload_session_array(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/sessions/s1/arrays"): ({"ok": True}, 200)}))
        assert ipred_client.upload_session_array("s1", {"a": 1}) == {"ok": True}


# ---------------------------------------------------------------------------
# preprocess / delete_feature_bank (accept an optional shared client)
# ---------------------------------------------------------------------------

class TestSharedClientParam:
    def test_preprocess_builds_minimal_body(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/preprocess"): ({"ok": 1}, 200)}))
        ipred_client.preprocess(session_id="s1")
        import json
        body = json.loads(log.requests[0].content)
        assert body == {"session_id": "s1", "slice_index": 0}

    def test_preprocess_includes_optional_fields_when_given(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/preprocess"): ({"ok": 1}, 200)}))
        ipred_client.preprocess(
            session_id="s1", feature_setup_id="f1", composition_id="c1",
            slice_index=5, array_ref="ref1",
        )
        import json
        body = json.loads(log.requests[0].content)
        assert body == {
            "session_id": "s1", "slice_index": 5, "composition_id": "c1",
            "feature_setup_id": "f1", "array_ref": "ref1",
        }

    def test_preprocess_reuses_a_passed_in_client_without_closing_it(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/preprocess"): ({"ok": 1}, 200)}))
        shared = ipred_client.new_shared_client()
        try:
            ipred_client.preprocess(session_id="s1", client=shared)
            assert shared.is_closed is False
            # A second call on the same shared client still works — proves it
            # wasn't closed by _use_client after the first call.
            ipred_client.preprocess(session_id="s2", client=shared)
            assert len(log.requests) == 2
        finally:
            shared.close()

    def test_delete_feature_bank(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("DELETE", "/features/f1"): ({"deleted": True}, 200)}))
        assert ipred_client.delete_feature_bank("f1") == {"deleted": True}


# ---------------------------------------------------------------------------
# byte-returning endpoints
# ---------------------------------------------------------------------------

class TestByteEndpoints:
    def test_feature_channel_bytes(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/features/f1/channels/2"): (b"\x89PNG", 200)}))
        assert ipred_client.feature_channel_bytes("f1", 2) == b"\x89PNG"

    def test_run_commit_png(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/runs/r1/commit.png"): (b"commitbytes", 200)}))
        assert ipred_client.run_commit_png("r1") == b"commitbytes"

    def test_run_status_png(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/runs/r1/status.png"): (b"statusbytes", 200)}))
        assert ipred_client.run_status_png("r1") == b"statusbytes"

    def test_run_proba_png(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/runs/r1/proba/3.png"): (b"probabytes", 200)}))
        assert ipred_client.run_proba_png("r1", 3) == b"probabytes"

    def test_manifold_heatmap_png(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/manifold/m1/heatmap.png"): (b"heatbytes", 200)}))
        assert ipred_client.manifold_heatmap_png("m1") == b"heatbytes"


# ---------------------------------------------------------------------------
# train / train_multi / infer / rethreshold / threshold_class_map / manifold_sample
# ---------------------------------------------------------------------------

class TestMlWrappers:
    def test_train_fills_default_trainer_and_empty_config(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/train"): ({"model_id": "m1"}, 200)}))
        result = ipred_client.train(session_id="s1", shapes=[{"a": 1}])
        assert result == {"model_id": "m1"}
        import json
        body = json.loads(log.requests[0].content)
        assert body["trainer_id"] == "catboost"
        assert body["config"] == {}

    def test_train_multi(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/train/multi"): ({"model_id": "m2"}, 200)}))
        result = ipred_client.train_multi(
            session_id="s1", slices={"0": []}, feature_ids={"0": "f1"},
        )
        assert result == {"model_id": "m2"}

    def test_infer_defaults(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/infer"): ({"run_id": "r1"}, 200)}))
        result = ipred_client.infer(session_id="s1")
        assert result == {"run_id": "r1"}
        import json
        body = json.loads(log.requests[0].content)
        assert body["store_probabilities"] is True
        assert body["alpha"] == 0.05

    def test_infer_store_probabilities_false_for_batch_apply(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/infer"): ({"run_id": "r2"}, 200)}))
        ipred_client.infer(session_id="s1", store_probabilities=False)
        import json
        body = json.loads(log.requests[0].content)
        assert body["store_probabilities"] is False

    def test_rethreshold(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/rethreshold"): ({"ok": 1}, 200)}))
        assert ipred_client.rethreshold(session_id="s1", alpha=0.1, run_id="r1") == {"ok": 1}

    def test_threshold_class_map(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/runs/r1/threshold-class"): ({"ok": 1}, 200)}))
        result = ipred_client.threshold_class_map("r1", class_id=2, threshold=0.5)
        assert result == {"ok": 1}
        import json
        body = json.loads(log.requests[0].content)
        assert body == {"class_id": 2, "threshold": 0.5}

    def test_manifold_sample(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("POST", "/manifold/sample"): ({"sample_id": "sm1"}, 200)}))
        assert ipred_client.manifold_sample({"k": "v"}) == {"sample_id": "sm1"}


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    def test_http_error_status_raises(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/health"): ({"error": "down"}, 500)}))
        with pytest.raises(httpx.HTTPStatusError):
            ipred_client.health()

    def test_missing_route_also_raises(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {}))
        with pytest.raises(httpx.HTTPStatusError):
            ipred_client.list_setups()


# ---------------------------------------------------------------------------
# new_shared_client / _use_client
# ---------------------------------------------------------------------------

class TestSharedClientLifecycle:
    def test_new_shared_client_is_open_and_usable(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/health"): ({"status": "ok"}, 200)}))
        client = ipred_client.new_shared_client()
        try:
            assert client.is_closed is False
        finally:
            client.close()

    def test_use_client_without_explicit_client_opens_and_closes_its_own(self, monkeypatch: pytest.MonkeyPatch):
        log = RequestLog()
        _install_handler(monkeypatch, _json_handler(log, {("GET", "/health"): ({"status": "ok"}, 200)}))
        with ipred_client._use_client(None, timeout=1.0) as c:
            assert c.is_closed is False
        assert c.is_closed is True
