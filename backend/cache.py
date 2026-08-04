"""Small TTL + LRU cache used by the Browse API.

Kept intentionally tiny: the Browse API only needs to memoize a handful of
results per endpoint, so a full LRU (functools) or a third-party cache would
be overkill. Entries are evicted when they expire *or* when the cache exceeds
a max size (oldest-inserted first).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Hashable


class TTLCache:
    """Thread-safe cache with per-entry expiry and an insertion-order size cap."""

    __slots__ = ("_ttl", "_max", "_store", "_lock")

    def __init__(self, ttl_seconds: float, max_entries: int = 128) -> None:
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._store: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
