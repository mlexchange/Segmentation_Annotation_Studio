"""Recognising sidecar containers written alongside an annotated array.

Saving annotations creates helper containers *next to* the array they belong to,
named after it: ``<name>__v_thumbs`` (version thumbnails, see
:mod:`annotation_thumbnails`) and ``<name>__masks`` (synced masks, see
:mod:`tiled_mask_sync`). They are bookkeeping, not image data.

Anything that enumerates a container's children — reading a stack's slices,
listing samples, counting slices for the Browse UI — must skip them. Otherwise
they surface as phantom samples/slices, and because callers map a row's
*position* to a slice index, a phantom row shifts every slice after it.

Lives in its own module so :mod:`arrays` and :mod:`browse_helpers` can share one
definition without depending on each other.
"""

from __future__ import annotations

from typing import Any

SIDECAR_SUFFIXES = ("__v_thumbs", "__masks")


def is_sidecar_key(key: Any) -> bool:
    """True if *key* names a sidecar container rather than real image data."""
    return str(key).endswith(SIDECAR_SUFFIXES)
