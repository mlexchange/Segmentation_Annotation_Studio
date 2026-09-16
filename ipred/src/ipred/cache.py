"""Tiny TTL cache for short-lived ipred artifacts (manifold heatmaps)."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Thread-safe TTL + max-entries cache."""

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self.ttl = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._data: dict[str, tuple[float, T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, val = item
            if now - ts > self.ttl:
                del self._data[key]
                return None
            return val

    def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            self._data[key] = (now, value)
            if len(self._data) > self.max_entries:
                # Drop oldest
                oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                del self._data[oldest]
