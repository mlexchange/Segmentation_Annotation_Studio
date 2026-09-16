"""channel_png_path validates feature_id against its known-safe shape before
ever constructing a filesystem path from it (CodeQL py/path-injection)."""

from __future__ import annotations

import uuid

import pytest

from ipred import preprocess


@pytest.fixture()
def local_data_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_valid_feature_id_resolves_under_project_blob_dir(local_data_root) -> None:
    from ipred.paths import project_blob_dir

    feature_id = uuid.uuid4().hex
    path = preprocess.channel_png_path("abc123", feature_id, 0)
    assert path == project_blob_dir("abc123") / "features" / feature_id / "channels" / "0000.png"


def test_traversal_payload_as_feature_id_is_rejected(local_data_root) -> None:
    with pytest.raises(ValueError, match="invalid feature_id"):
        preprocess.channel_png_path("abc123", "../../../../etc/passwd", 0)


def test_wrong_length_feature_id_is_rejected(local_data_root) -> None:
    with pytest.raises(ValueError, match="invalid feature_id"):
        preprocess.channel_png_path("abc123", "not-a-real-uuid", 0)


def test_uppercase_hex_feature_id_is_rejected(local_data_root) -> None:
    # uuid.uuid4().hex is always lowercase — anything else can't be a real one.
    with pytest.raises(ValueError, match="invalid feature_id"):
        preprocess.channel_png_path("abc123", uuid.uuid4().hex.upper(), 0)
