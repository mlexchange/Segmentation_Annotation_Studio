"""Tests for concurrent, versioned annotation-guide persistence."""

from __future__ import annotations

import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


def test_guide_document_has_schema_discriminator(tmp_path: Path) -> None:
    """Guide files identify their document type and migration version."""
    with patch.dict(os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}):
        import guides

        importlib.reload(guides)
        result = guides.save_guide("source", {"classes": [], "notes": "hello"})
        document = json.loads(Path(result["path"]).read_text())

    assert document["schema_version"] == 1
    assert document["document_type"] == "guide"


def test_concurrent_guide_writes_use_unique_temporary_files(tmp_path: Path) -> None:
    """Overlapping guide autosaves remain valid and do not race on one temp path."""
    with patch.dict(os.environ, {"LOCAL_DATA_ROOT": str(tmp_path)}):
        import guides

        importlib.reload(guides)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: guides.save_guide(
                        "shared-guide",
                        {"classes": [], "notes": f"note-{i}"},
                    ),
                    range(20),
                )
            )
        loaded = guides.load_guide("shared-guide")
        guide_dir = Path(results[0]["path"]).parent

    assert loaded is not None
    assert loaded["guide"]["notes"] in {f"note-{i}" for i in range(20)}
    assert list(guide_dir.glob("*.tmp")) == []
