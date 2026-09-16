"""Find the Tiled node that holds a dataset's renderable 3-D volume.

The 3-D viewer needs an OME-NGFF multiscale group. Which node that is depends on
how the dataset got into the catalog, and the answer is not something the
frontend can work out from a path:

* **Registered Zarr volumes** already are one (:mod:`zarr_source`), so the open
  node — or an ancestor of it, when a specific pyramid level is open — is the
  volume.
* **TIFF stacks** are stored as a container of per-slice 2-D arrays, which is not
  a volume at all. Their 3-D view lives in the ``__volume`` sidecar built by
  :mod:`tiff_stack_source`, and it only exists once someone has built it.

Without this the viewer is pointed straight at whatever is open and fails with
``openOmeZarr: missing multiscales in root .zattrs`` — technically accurate and
useless to the person reading it, since the real answer is either "look at the
sidecar" or "no volume has been built for this dataset yet".

Detection is on ``metadata["attributes"]["multiscales"]``: Tiled's ``.zattrs``
route returns ``metadata["attributes"]`` verbatim, so that key is exactly what
the viewer will see.
"""

from __future__ import annotations

import logging
from typing import Any

from tiff_stack_source import VOLUME_SUFFIX
from tiled_clients import api_key_for_uri, get_tiled_client

logger = logging.getLogger("volume_nodes")

#: How far to walk up from an open node looking for the multiscale root. A
#: pyramid level sits at most two levels down (``<volume>/scale0/image``).
_MAX_ASCENT = 3


def has_multiscales(node: Any) -> bool:
    """True if *node* carries OME-NGFF ``multiscales`` where Tiled will serve it."""
    try:
        attributes = (dict(getattr(node, "metadata", {}) or {})).get("attributes") or {}
    except Exception:  # noqa: BLE001 — a node we cannot describe is not a volume
        return False
    multiscales = attributes.get("multiscales")
    return isinstance(multiscales, list) and bool(multiscales)


def _node_at(client: Any, parts: list[str]) -> Any | None:
    node = client
    for part in parts:
        try:
            node = node[part]
        except Exception:  # noqa: BLE001 — missing child, or not a container
            return None
    return node


def source_dir_of(node: Any) -> str | None:
    """Filesystem path a volume was registered from, if it records one."""
    try:
        meta = dict(getattr(node, "metadata", {}) or {})
    except Exception:  # noqa: BLE001
        return None
    for key in ("tiff_dir", "zarr_path"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_volume(server_uri: str | None, source: str) -> dict[str, Any]:
    """Locate the renderable volume for the dataset open at *source*.

    Args:
        server_uri: Connected Tiled server URI; ``None`` uses the default.
        source: Tiled path of the open dataset.

    Returns:
        ``mode`` is one of:

        * ``"self"`` — the open node is the volume.
        * ``"ancestor"`` — a specific level was open; ``path`` is its root.
        * ``"sidecar"`` — the ``__volume`` node built for a TIFF stack.
        * ``"none"`` — no volume exists yet; ``message`` says so in plain terms
          and ``source_dir`` carries a build candidate when one is known.
    """
    parts = [p for p in (source or "").strip("/").split("/") if p]
    if not parts:
        return {"path": None, "mode": "none", "source_dir": None,
                "message": "Open a dataset from Browse to view it in 3D."}

    client = get_tiled_client(server_uri, api_key_for_uri(server_uri))

    # 1. The open node itself.
    node = _node_at(client, parts)
    if node is not None and has_multiscales(node):
        return {"path": "/".join(parts), "mode": "self",
                "source_dir": source_dir_of(node), "message": ""}

    # 2. An ancestor — the user may have opened one pyramid level directly, e.g.
    #    `<volume>/scale0/image`, which is an array and has no multiscales itself.
    for depth in range(1, min(_MAX_ASCENT, len(parts)) + 1):
        ancestor_parts = parts[:-depth]
        if not ancestor_parts:
            break
        ancestor = _node_at(client, ancestor_parts)
        if ancestor is not None and has_multiscales(ancestor):
            return {"path": "/".join(ancestor_parts), "mode": "ancestor",
                    "source_dir": source_dir_of(ancestor), "message": ""}

    # 3. The sidecar built for a per-slice TIFF stack.
    sidecar_parts = [*parts[:-1], f"{parts[-1]}{VOLUME_SUFFIX}"]
    sidecar = _node_at(client, sidecar_parts)
    if sidecar is not None and has_multiscales(sidecar):
        return {"path": "/".join(sidecar_parts), "mode": "sidecar",
                "source_dir": source_dir_of(sidecar), "message": ""}

    return {
        "path": None,
        "mode": "none",
        "source_dir": None,
        "message": (
            "No 3-D volume has been built for this dataset yet. It is stored as "
            "individual 2-D slices, which the 3-D view cannot stream — build one "
            "from the source images to enable it."
        ),
    }
