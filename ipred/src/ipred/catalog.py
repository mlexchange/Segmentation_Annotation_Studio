"""SQLite catalog for projects, sessions, feature banks, models, and runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ipred.paths import catalog_db_path, project_blob_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_id_for(
    *,
    kind: str,
    source: str,
    server_uri: str | None = None,
    root: str | None = None,
) -> str:
    """Stable project id from source identity."""
    payload = json.dumps(
        {
            "kind": kind,
            "source": source,
            "server_uri": server_uri or "",
            "root": root or "",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProjectRow:
    """Project catalog row."""

    project_id: str
    kind: str
    source: str
    server_uri: str | None
    root: str | None
    created_at: str


@dataclass(frozen=True)
class SessionRow:
    """Session catalog row."""

    session_id: str
    project_id: str
    current_feature_id: str | None
    current_model_id: str | None
    current_run_id: str | None
    created_at: str
    updated_at: str


class Catalog:
    """SQLite-backed engine catalog."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else catalog_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                  project_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  source TEXT NOT NULL,
                  server_uri TEXT,
                  root TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL REFERENCES projects(project_id),
                  current_feature_id TEXT,
                  current_model_id TEXT,
                  current_run_id TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feature_setups (
                  setup_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  meta_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feature_banks (
                  feature_id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  setup_id TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  slice_index INTEGER NOT NULL,
                  n_channels INTEGER NOT NULL,
                  height INTEGER NOT NULL,
                  width INTEGER NOT NULL,
                  blob_dir TEXT NOT NULL,
                  setup_snapshot TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(project_id, setup_id, content_hash, slice_index)
                );

                CREATE TABLE IF NOT EXISTS models (
                  model_id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  feature_id TEXT NOT NULL,
                  trainer_id TEXT NOT NULL,
                  blob_dir TEXT NOT NULL,
                  meta_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  project_id TEXT NOT NULL,
                  model_id TEXT NOT NULL,
                  feature_id TEXT NOT NULL,
                  alpha REAL NOT NULL,
                  blob_dir TEXT NOT NULL,
                  meta_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    def ensure_project(
        self,
        *,
        kind: str,
        source: str,
        server_uri: str | None = None,
        root: str | None = None,
    ) -> ProjectRow:
        """Create or return the project for this source identity."""
        pid = project_id_for(
            kind=kind, source=source, server_uri=server_uri, root=root
        )
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE project_id = ?", (pid,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO projects
                      (project_id, kind, source, server_uri, root, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (pid, kind, source, server_uri, root, now),
                )
                project_blob_dir(pid)
                row = conn.execute(
                    "SELECT * FROM projects WHERE project_id = ?", (pid,)
                ).fetchone()
        return self._project_from_row(row)

    def open_session(
        self,
        *,
        kind: str,
        source: str,
        server_uri: str | None = None,
        root: str | None = None,
    ) -> SessionRow:
        """Ensure project exists and open a new session on it."""
        project = self.ensure_project(
            kind=kind, source=source, server_uri=server_uri, root=root
        )
        sid = uuid.uuid4().hex
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                  (session_id, project_id, current_feature_id, current_model_id,
                   current_run_id, created_at, updated_at)
                VALUES (?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (sid, project.project_id, now, now),
            )
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
        return self._session_from_row(row)

    def get_session(self, session_id: str) -> SessionRow | None:
        """Look up a session."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._session_from_row(row) if row else None

    def get_project(self, project_id: str) -> ProjectRow | None:
        """Look up a project."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        return self._project_from_row(row) if row else None

    def set_session_currents(
        self,
        session_id: str,
        *,
        feature_id: str | None = None,
        model_id: str | None = None,
        run_id: str | None = None,
    ) -> SessionRow:
        """Update session current_* pointers (only provided fields)."""
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id}")
        feat = feature_id if feature_id is not None else session.current_feature_id
        model = model_id if model_id is not None else session.current_model_id
        run = run_id if run_id is not None else session.current_run_id
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET current_feature_id = ?, current_model_id = ?,
                    current_run_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (feat, model, run, now, session_id),
            )
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._session_from_row(row)

    def upsert_feature_setup(
        self,
        *,
        setup_id: str,
        name: str,
        kind: str,
        content_hash: str,
        meta: dict[str, Any],
    ) -> None:
        """Insert or update a feature setup index row."""
        now = _utc_now()
        payload = json.dumps(meta, sort_keys=True)
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT setup_id FROM feature_setups WHERE setup_id = ?",
                (setup_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO feature_setups
                      (setup_id, name, kind, content_hash, meta_json,
                       created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (setup_id, name, kind, content_hash, payload, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE feature_setups
                    SET name = ?, kind = ?, content_hash = ?, meta_json = ?,
                        updated_at = ?
                    WHERE setup_id = ?
                    """,
                    (name, kind, content_hash, payload, now, setup_id),
                )

    def find_feature_bank(
        self,
        *,
        project_id: str,
        setup_id: str,
        content_hash: str,
        slice_index: int,
    ) -> dict[str, Any] | None:
        """Return a ready feature-bank row dict or None."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM feature_banks
                WHERE project_id = ? AND setup_id = ? AND content_hash = ?
                  AND slice_index = ? AND status = 'ready'
                """,
                (project_id, setup_id, content_hash, slice_index),
            ).fetchone()
        return dict(row) if row else None

    def insert_feature_bank(self, record: dict[str, Any]) -> None:
        """Insert a feature-bank row."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feature_banks
                  (feature_id, project_id, setup_id, content_hash, slice_index,
                   n_channels, height, width, blob_dir, setup_snapshot, status,
                   created_at)
                VALUES
                  (:feature_id, :project_id, :setup_id, :content_hash,
                   :slice_index, :n_channels, :height, :width, :blob_dir,
                   :setup_snapshot, :status, :created_at)
                """,
                record,
            )

    def get_feature_bank(self, feature_id: str) -> dict[str, Any] | None:
        """Look up a feature bank by id."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM feature_banks WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_feature_bank(self, feature_id: str) -> str | None:
        """Delete a feature-bank row and return its `blob_dir` for the caller
        to remove from disk (this method does not touch the filesystem — it
        only owns the DB row, matching every other method here).

        Feature banks bake 32-64 full-resolution float32 PCA channels
        (~1 GB per 2000x2000 slice, see `sam_embed.emb_grid_to_pca_channels`)
        and were never evicted: a single interactive slice benefits from the
        cache (re-predicting after a param tweak), but a volume-wide batch
        job (Phase 4.5's "Apply across volume") touches every slice exactly
        once, so there is nothing later in that job to reuse the cache for —
        it just accumulates forever. 690 slices at ~1 GB each is enough to
        fill a laptop's disk mid-job. Batch-apply calls this right after each
        slice's inference completes; ordinary single-slice interactive use
        (train/preprocess/infer outside a batch job) is untouched.

        Returns None if the feature_id doesn't exist (already deleted, or a
        bad id) — the caller should treat that as a no-op, not an error.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT blob_dir FROM feature_banks WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM feature_banks WHERE feature_id = ?", (feature_id,))
        return row["blob_dir"]

    def insert_model(self, record: dict[str, Any]) -> None:
        """Insert a model row."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO models
                  (model_id, project_id, feature_id, trainer_id, blob_dir,
                   meta_json, created_at)
                VALUES
                  (:model_id, :project_id, :feature_id, :trainer_id, :blob_dir,
                   :meta_json, :created_at)
                """,
                record,
            )

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Look up a model row."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        return dict(row) if row else None

    def insert_run(self, record: dict[str, Any]) -> None:
        """Insert a run row."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                  (run_id, project_id, model_id, feature_id, alpha, blob_dir,
                   meta_json, created_at, updated_at)
                VALUES
                  (:run_id, :project_id, :model_id, :feature_id, :alpha,
                   :blob_dir, :meta_json, :created_at, :updated_at)
                """,
                record,
            )

    def update_run(
        self,
        run_id: str,
        *,
        alpha: float,
        meta_json: str,
    ) -> None:
        """Update run alpha/meta after rethreshold."""
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET alpha = ?, meta_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (alpha, meta_json, now, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Look up a run row."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ProjectRow:
        return ProjectRow(
            project_id=row["project_id"],
            kind=row["kind"],
            source=row["source"],
            server_uri=row["server_uri"],
            root=row["root"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionRow:
        return SessionRow(
            session_id=row["session_id"],
            project_id=row["project_id"],
            current_feature_id=row["current_feature_id"],
            current_model_id=row["current_model_id"],
            current_run_id=row["current_run_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
