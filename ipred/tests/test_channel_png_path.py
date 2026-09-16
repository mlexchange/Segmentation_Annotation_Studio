"""channel_png_path's engine-root containment guard (CodeQL py/path-injection)."""

from __future__ import annotations

import pytest

from ipred import preprocess


@pytest.fixture()
def local_data_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_normal_blob_dir_resolves_under_engine_root(local_data_root) -> None:
    from ipred.paths import engine_root

    blob_dir = engine_root() / "projects" / "abc123" / "features" / "def456"
    path = preprocess.channel_png_path(str(blob_dir), 0)
    assert path == (blob_dir / "channels" / "0000.png").resolve()


def test_traversal_in_blob_dir_is_rejected(local_data_root) -> None:
    from ipred.paths import engine_root

    escaping = engine_root() / "projects" / ".." / ".." / "etc"
    with pytest.raises(ValueError, match="escapes engine root"):
        preprocess.channel_png_path(str(escaping), 0)


def test_absolute_blob_dir_outside_root_is_rejected(local_data_root, tmp_path) -> None:
    outside = tmp_path.parent / "somewhere-else"
    with pytest.raises(ValueError, match="escapes engine root"):
        preprocess.channel_png_path(str(outside), 0)
