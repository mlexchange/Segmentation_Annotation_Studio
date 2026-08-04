"""Browse helper functions used by the Browse API routes.

Exports
-------
* :class:`FieldMapping`        — typed metadata field mapping
* :func:`build_field_mapping`  — introspect a Tiled container for metadata keys
* :func:`tiled_distinct_values` — distinct values + counts for one field
* :func:`tiled_search_items`   — filtered item list

The Tiled-specific imports (``tiled.queries``) are done lazily so this module
can be imported in unit tests without pulling in the full Tiled stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from sidecars import is_sidecar_key

logger = logging.getLogger("browse-server")

# Display-name aliases applied to ``thinfilm_<name>`` raw keys and studio keys.
DISPLAY_KEY_ALIASES: dict[str, str] = {
    "AnnealingTemp": "Temp",
    "studio_annotated": "Annotated",
    "studio_shape_count": "Shape count",
    "studio_class_count": "Class count",
    "studio_updated_at": "Annotated at",
}

# Raw keys written by annotation autosave (always exposed in Browse).
_STUDIO_RAW_KEYS: tuple[str, ...] = (
    "studio_annotated",
    "studio_shape_count",
    "studio_class_count",
    "studio_updated_at",
)

# Ingest-time keys worth showing as Browse facets even when only ONE distinct
# value exists (e.g. a single dropped file, or a whole batch sharing one
# user-supplied description). Without this they'd be hidden by the >=2 rule.
_INGEST_FACET_RAW_KEYS: tuple[str, ...] = (
    "description",
    "keywords",
    "sample_name",
    "original_filename",
)

# Keys that qualify as a facet with a single distinct value (vs. the default >=2).
_SINGLE_VALUE_FACET_RAW_KEYS: frozenset[str] = frozenset(
    (*_STUDIO_RAW_KEYS, *_INGEST_FACET_RAW_KEYS),
)

# Metadata keys stored as a LIST of values (rather than a scalar). Each element
# is treated as its own distinct Browse value, and filtering matches membership
# (via Tiled's ``Contains`` query) rather than equality.
_LIST_VALUED_RAW_KEYS: frozenset[str] = frozenset({"keywords"})

# Raw keys that are stored at the array-node level rather than on the parent
# sample container. When a filter references one of these, we switch to a
# per-sample search path.
_ARRAY_ONLY_RAW_KEYS: frozenset[str] = frozenset({
    "angle_id", "incident_angle_deg", "bar", *_STUDIO_RAW_KEYS,
})

# Heuristics used when introspecting the first few samples to decide that we
# have "enough" metadata keys to lock in the mapping.
_SCAN_MAX_SAMPLES = 20
_KEY_RICHNESS_THRESHOLD = 10
_IMPORTANT_KEYS = frozenset({
    "PI", "sample_name", "incident_angle_deg", "technique", "scan_type",
    "studio_annotated", "studio_shape_count",
})
_IMPORTANT_MIN_COUNT = 4


@dataclass(frozen=True)
class FieldMapping:
    """Mapping between raw Tiled metadata keys and friendly display names."""

    display_to_raw: dict[str, str] = field(default_factory=dict)
    raw_to_display: dict[str, str] = field(default_factory=dict)
    all_display_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Field discovery
# ---------------------------------------------------------------------------

def build_field_mapping(container_node: Any) -> FieldMapping:
    """Introspect *container_node* to discover available metadata keys.

    Walks up to :data:`_SCAN_MAX_SAMPLES` children and picks the first whose
    combined (container + first-array-child) metadata is "rich enough" — i.e.
    has thinfilm keys, enough important keys, or at least
    :data:`_KEY_RICHNESS_THRESHOLD` keys. Falls back to the last scanned set.
    """
    raw_keys: list[str] = []
    last_combined: list[str] = []

    try:
        for idx, first_key in enumerate(container_node):
            if idx >= _SCAN_MAX_SAMPLES:
                break

            try:
                combined_keys = _combined_metadata_keys(container_node[first_key])
            except Exception:
                # Skip samples that error during lazy open (e.g. bad array payloads).
                continue
            last_combined = combined_keys

            if _is_rich_enough(combined_keys):
                raw_keys = combined_keys
                break
    except (StopIteration, KeyError, TypeError):
        pass

    if not raw_keys:
        raw_keys = last_combined

    display_to_raw: dict[str, str] = {}
    raw_to_display: dict[str, str] = {}

    for raw_key in raw_keys:
        display_key = _display_name(raw_key)
        display_to_raw.setdefault(display_key, raw_key)
        raw_to_display[raw_key] = display_key

    _inject_studio_keys(display_to_raw, raw_to_display)

    return FieldMapping(
        display_to_raw=display_to_raw,
        raw_to_display=raw_to_display,
        all_display_keys=sorted(display_to_raw.keys()),
    )


# Ingest writes these at the dataset-container level, so they're always worth
# offering as facets even when the scanned sample metadata didn't surface them.
_INJECTED_INGEST_RAW_KEYS: tuple[str, ...] = ("description", "keywords", "sample_name")


def _inject_studio_keys(
    display_to_raw: dict[str, str],
    raw_to_display: dict[str, str],
) -> None:
    """Ensure studio annotation + ingest description fields are always in Browse."""
    for raw_key in (*_STUDIO_RAW_KEYS, *_INJECTED_INGEST_RAW_KEYS):
        display_key = _display_name(raw_key)
        display_to_raw.setdefault(display_key, raw_key)
        raw_to_display.setdefault(raw_key, display_key)


def _combined_metadata_keys(first_item: Any) -> list[str]:
    """Return metadata keys present on the sample/container node.

    We intentionally do **not** open the first array child to merge keys: in
    Tiled, subscripting an array often issues a slice read that can fail with
    HTTP 500 and long client-side retries, which blocks facet discovery for
    tens of seconds. Array-only fields are still handled elsewhere via
    :data:`browse_helpers._ARRAY_ONLY_RAW_KEYS` and the items search path.
    """
    item_meta = getattr(first_item, "metadata", {}) or {}
    return list(getattr(item_meta, "keys", lambda: [])())


def _is_rich_enough(keys: list[str]) -> bool:
    has_thinfilm = any(str(k).startswith("thinfilm_") for k in keys)
    has_important = sum(1 for k in keys if k in _IMPORTANT_KEYS) >= 1
    return has_thinfilm or (has_important and len(keys) >= _IMPORTANT_MIN_COUNT) or len(keys) >= _KEY_RICHNESS_THRESHOLD


def _display_name(raw_key: str) -> str:
    if raw_key.startswith("thinfilm_"):
        stripped = raw_key[len("thinfilm_"):]
        return DISPLAY_KEY_ALIASES.get(stripped, stripped)
    return DISPLAY_KEY_ALIASES.get(raw_key, raw_key)


# ---------------------------------------------------------------------------
# Distinct values
# ---------------------------------------------------------------------------

def scoped_metadata_rows(node: Any, limit: int = 5000) -> list[dict]:
    """Read each child's metadata once for a *specific* container.

    Tiled's ``container.distinct()`` aggregates across the whole catalog, not
    just the node it's called on, so it can't be used to compute values scoped
    to a chosen sub-container. Iterating children gives correctly-scoped data
    at the cost of one metadata read per child.
    """
    rows: list[dict] = []
    try:
        for key in list(node)[:limit]:
            try:
                meta = node[key].metadata
                rows.append(dict(meta) if meta else {})
            except Exception:  # noqa: BLE001 — skip children that fail to open
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("scoped_metadata_rows iteration failed: %s", exc)
    return rows


def distinct_from_rows(rows: list[dict], raw_key: str) -> list[dict]:
    """Tally distinct values of *raw_key* across pre-read metadata *rows*.

    List-valued metadata (e.g. ``keywords``) is exploded so each element is
    counted as its own distinct value, making every tag individually filterable.
    """
    counts: dict[Any, int] = {}
    for meta in rows:
        val = meta.get(raw_key)
        if val is None:
            continue
        values = val if isinstance(val, (list, tuple)) else [val]
        for item in values:
            if item is None:
                continue
            counts[item] = counts.get(item, 0) + 1
    return [{"value": v, "count": c} for v, c in counts.items()]


def tiled_distinct_values(
    container_node: Any,
    raw_key: str,
    filters: Optional[dict] = None,
    field_mapping: Optional[FieldMapping] = None,
    limit: int = 500,
    scoped: bool = False,
) -> dict:
    """Return distinct values (+ counts) for *raw_key* under optional filters.

    When *scoped* is True, values are computed by iterating the (filtered)
    container's children so they reflect only that container — use this when
    browsing a specific ``container_path``. Otherwise the faster but
    catalog-global ``container.distinct()`` is used.

    Returns::

        {
            "values": [{"value": str, "count": int, "sample_paths": []}, ...],
            "total":  int,
            "field":  str,
        }
    """
    from tiled.queries import Key

    filters = filters or {}
    display_to_raw = field_mapping.display_to_raw if field_mapping else {}

    node = container_node
    for disp_key, val in filters.items():
        if val is None:
            continue
        r_key = display_to_raw.get(disp_key, disp_key)
        try:
            node = node.search(Key(r_key) == val)
        except Exception:  # noqa: BLE001 — Tiled raises a range of errors here
            pass

    if scoped:
        raw_values = distinct_from_rows(scoped_metadata_rows(node), raw_key)
    else:
        try:
            result = node.distinct(raw_key, counts=True)
            raw_values = result.get("metadata", {}).get(raw_key, [])
        except Exception as exc:
            logger.warning("distinct() failed for key %r: %s", raw_key, exc)
            raw_values = []

    non_null = [entry for entry in raw_values if _is_valid_value(entry.get("value"))]
    sorted_vals = sorted(non_null, key=lambda e: (-e.get("count", 0), str(e["value"])))[:limit]
    display_key = field_mapping.raw_to_display.get(raw_key, raw_key) if field_mapping else raw_key

    return {
        "values": [
            {"value": str(e["value"]).strip(), "count": e.get("count", 1), "sample_paths": []}
            for e in sorted_vals
        ],
        "total": len(non_null),
        "field": display_key,
    }


# ---------------------------------------------------------------------------
# Item search
# ---------------------------------------------------------------------------

def tiled_search_items(
    container_node: Any,
    filters: Optional[dict] = None,
    field_mapping: Optional[FieldMapping] = None,
    limit: int = 500,
    container_path_prefix: str = "",
) -> dict:
    """Apply *filters* via ``search(Key())`` and return matching container entries.

    Dispatches to one of three strategies:

    * No array-only filters → straight container search.
    * Only array-only filters → search, then map array paths back to parents.
    * Mixed filters → per-sample container match + array-level subsearch.
    """
    filters = filters or {}
    display_to_raw = field_mapping.display_to_raw if field_mapping else {}
    raw_filters = _raw_filters(filters, display_to_raw)

    has_array_only = any(rk in _ARRAY_ONLY_RAW_KEYS for rk, _ in raw_filters)
    has_container_level = any(rk not in _ARRAY_ONLY_RAW_KEYS for rk, _ in raw_filters)

    if not has_array_only:
        return _search_container_only(container_node, raw_filters, container_path_prefix, limit)

    if not has_container_level:
        return _search_array_only(container_node, raw_filters, container_path_prefix, limit)

    return _search_mixed(container_node, raw_filters, container_path_prefix, limit)


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------


# A group nested this many levels deep still gets expanded; beyond it, a
# grouping container is described as one opaque sample rather than recursed
# into forever. 3 covers "ingest batch / sample" (1) plus one extra level of
# manual re-organisation — deeper than that is not a shape ingest produces.
_MAX_GROUP_EXPANSION_DEPTH = 3


def _search_container_only(
    container_node: Any,
    raw_filters: list[tuple[str, str]],
    prefix: str,
    limit: int,
) -> dict:
    node = _apply_filters(container_node, raw_filters)

    items: list[dict] = []
    try:
        for k in _visible_keys(node):
            if len(items) >= limit:
                break
            try:
                entry = node[k]
                group_children = _fetch_if_all_containers(entry)
                if group_children is not None:
                    # A grouping container: ingest wrote browse/<batch>/<sample>/<arrays>
                    # when splitting one upload into several samples. The samples are a
                    # level deeper than Browse's "child of the root is a sample" model,
                    # so surface them (recursing through further nested groups) instead
                    # of one opaque row.
                    items.extend(
                        _expand_sample_group(group_children, _join(prefix, k), limit - len(items))
                    )
                else:
                    items.append(_describe_sample(entry, _join(prefix, k), k))
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("tiled_search_items iteration failed: %s", exc)

    truncated = items[:limit]
    return {"items": truncated, "total": len(truncated)}


def _fetch_if_all_containers(entry: Any) -> dict[str, Any] | None:
    """If every (non-sidecar) child of *entry* is itself a container, return the
    already-fetched ``{key: node}`` mapping; otherwise ``None``.

    Distinguishes a grouping parent (containers of samples) from a stack
    (arrays). An empty container is not a group — nothing to expand. Returning
    the fetched children (rather than a bool) means ``_expand_sample_group``
    doesn't re-list and re-fetch the same children a second time.

    The container check MUST come before anything that iterates *entry* — an
    array node has no "keys" in this sense, but `for k in entry` doesn't raise
    on one; it falls back to the sequence protocol and yields the array's rows
    one at a time (a real network fetch per row for a Tiled array client). On
    a several-thousand-row image that silently turns one "is this a group?"
    check into thousands of row fetches instead of failing fast, and it does
    this once per sibling array in the container — this is what made Browse
    hang on a freshly re-ingested full-resolution dataset. `_is_container` is
    a single metadata attribute read, not a fetch, so checking it first is
    what actually keeps this function cheap on a leaf.
    """
    if not _is_container(entry):
        return None
    keys = _visible_keys(entry)
    if not keys:
        return None
    try:
        children = {k: entry[k] for k in keys}
    except Exception:  # noqa: BLE001
        return None
    return children if all(_is_container(child) for child in children.values()) else None


def _expand_sample_group(
    children: dict[str, Any], group_path: str, limit: int, depth: int = 1
) -> list[dict]:
    """Describe each child container of a grouping parent as its own sample.

    A grouping container's children are occasionally themselves all-container
    groups (an upload grouped, then reorganised into a further sub-folder) —
    recurses up to :data:`_MAX_GROUP_EXPANSION_DEPTH` so those samples are found
    instead of the outer group being mis-described as one opaque image (whose
    "first child" would be a container, not image data).
    """
    out: list[dict] = []
    for key, child in children.items():
        if len(out) >= limit:
            break
        try:
            path = _join(group_path, key)
            if depth < _MAX_GROUP_EXPANSION_DEPTH:
                nested = _fetch_if_all_containers(child)
                if nested is not None:
                    out.extend(_expand_sample_group(nested, path, limit - len(out), depth + 1))
                    continue
            out.append(_describe_sample(child, path, key))
        except Exception:  # noqa: BLE001
            continue
    return out


def _describe_sample(entry: Any, path: str, key: str) -> dict:
    """Build one sample record, counting array children as its slices.

    Drag-and-drop ingest nests as ``browse/<dataset>/<array(s)>``: the dataset is
    a container of array slices, with the descriptive metadata on the children. So
    report the slice count (letting the UI offer drill-in) and, when the container
    itself is metadata-less, borrow the first child's metadata (a cheap
    metadata-only read).
    """
    meta = dict(entry.metadata) if hasattr(entry, "metadata") and entry.metadata else {}
    n_slices = 1
    if _is_container(entry):
        # Sidecars excluded so the count matches what /api/browse/slices lists and
        # arrays._stack_keys reads. Counting them would report a lone annotated
        # image as a 2-slice volume and send the UI down the drill-in path instead
        # of opening it directly.
        child_keys = _visible_keys(entry)
        if child_keys and not _is_container(entry[child_keys[0]]):
            n_slices = len(child_keys)  # flat stack of array slices
            if not meta:
                cmeta = entry[child_keys[0]].metadata
                meta = dict(cmeta) if cmeta else {}
    return {"path": path, "sample": key, "metadata": meta, "n_slices": n_slices}


def _visible_keys(node: Any) -> list[str]:
    """Child keys of *node* excluding sidecar containers (see :mod:`sidecars`),
    which are annotation bookkeeping rather than browsable image data."""
    return [k for k in node if not is_sidecar_key(k)]


def _is_container(entry: Any) -> bool:
    """True if *entry* is a Tiled container (vs. an array/leaf) node."""
    sf = getattr(entry, "structure_family", None)
    return str(getattr(sf, "value", sf)) == "container"


def _search_array_only(
    container_node: Any,
    raw_filters: list[tuple[str, str]],
    prefix: str,
    limit: int,
) -> dict:
    node = _apply_filters(container_node, raw_filters)
    array_only_vals = {rk: v for rk, v in raw_filters if rk in _ARRAY_ONLY_RAW_KEYS}

    sample_keys: list[str] = []
    seen: set[str] = set()
    try:
        for k in list(node)[: max(limit * 200, 1000)]:
            sample_key = str(k).split("/", 1)[0]
            if is_sidecar_key(sample_key):
                continue
            if sample_key and sample_key not in seen:
                seen.add(sample_key)
                sample_keys.append(sample_key)
            if len(sample_keys) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("tiled_search_items array-only iteration failed: %s", exc)

    items: list[dict] = []
    for sample_key in sample_keys:
        try:
            sample_node = container_node[sample_key]
        except Exception:  # noqa: BLE001
            continue
        meta = dict(sample_node.metadata) if hasattr(sample_node, "metadata") else {}
        meta.update(array_only_vals)
        items.append({"path": _join(prefix, sample_key), "sample": sample_key, "metadata": meta})

    return {"items": items, "total": len(items)}


def _search_mixed(
    container_node: Any,
    raw_filters: list[tuple[str, str]],
    prefix: str,
    limit: int,
) -> dict:
    container_filters = [(rk, v) for rk, v in raw_filters if rk not in _ARRAY_ONLY_RAW_KEYS]
    array_only_vals = {rk: v for rk, v in raw_filters if rk in _ARRAY_ONLY_RAW_KEYS}

    items: list[dict] = []
    sample_scan_budget = max(500, limit * 5)

    for sample_key in _visible_keys(container_node)[:sample_scan_budget]:
        try:
            sample_node = container_node[sample_key]
        except Exception:  # noqa: BLE001
            continue

        meta = dict(sample_node.metadata) if hasattr(sample_node, "metadata") else {}

        if not _matches_container_filters(meta, container_filters):
            continue

        sub = _apply_filters(sample_node, [(rk, v) for rk, v in raw_filters if rk in _ARRAY_ONLY_RAW_KEYS])
        try:
            if not list(sub)[:1]:
                continue
        except Exception:  # noqa: BLE001
            continue

        meta.update(array_only_vals)
        items.append({"path": _join(prefix, sample_key), "sample": sample_key, "metadata": meta})
        if len(items) >= limit:
            break

    return {"items": items, "total": len(items)}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _raw_filters(
    filters: dict,
    display_to_raw: dict[str, str],
) -> list[tuple[str, str]]:
    """Convert a display-key filter dict to (raw_key, string-value) pairs."""
    out: list[tuple[str, str]] = []
    for disp_key, val in filters.items():
        if val is None:
            continue
        out.append((display_to_raw.get(disp_key, disp_key), str(val)))
    return out


class FilterApplicationError(RuntimeError):
    """Raised when a requested filter cannot be enforced against Tiled.

    Distinct from other Tiled/browse errors so callers can tell "a restriction
    the caller asked for could not be applied" apart from an unrelated backend
    failure — but both are still surfaced as an error, never silently dropped.
    """


def _apply_filters(node: Any, raw_filters: Iterable[tuple[str, str]]) -> Any:
    """Apply each filter to *node*.

    List-valued keys (e.g. ``keywords``) match membership via ``Contains``;
    scalar keys match equality via ``Key(...) == value``. A filter that Tiled
    rejects or fails to apply raises rather than being silently skipped — a
    dropped filter would silently widen the result set past what the caller
    explicitly requested, which is a data-boundary bug, not a UX nicety.
    """
    from tiled.queries import Key

    try:
        from tiled.queries import Contains
    except ImportError:  # pragma: no cover — older Tiled without Contains
        Contains = None

    for r_key, val in raw_filters:
        try:
            if r_key in _LIST_VALUED_RAW_KEYS and Contains is not None:
                node = node.search(Contains(r_key, val))
            else:
                node = node.search(Key(r_key) == _typed_query_value(val))
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise FilterApplicationError(f"could not apply filter {r_key!r}") from exc
    return node


def _matches_container_filters(meta: dict, container_filters: list[tuple[str, str]]) -> bool:
    """Case-insensitive match over container-level metadata.

    Scalar values match by equality; list-valued metadata matches if the filter
    value is one of the list's members.
    """
    for r_key, val in container_filters:
        sample_val = meta.get(r_key)
        if sample_val is None:
            return False
        target = val.strip().lower()
        if isinstance(sample_val, (list, tuple)):
            if target not in {str(item).strip().lower() for item in sample_val}:
                return False
        elif str(sample_val).strip().lower() != target:
            return False
    return True


def _typed_query_value(raw: str) -> Any:
    """Coerce a filter string to int / float when it round-trips exactly.

    Tiled stores many fields as numbers; submitting them as strings silently
    fails to match. We only downcast when the conversion is unambiguous.
    """
    try:
        iv = int(raw)
        if str(iv) == raw:
            return iv
    except (ValueError, TypeError):
        pass
    try:
        fv = float(raw)
        if str(fv) == raw:
            return fv
    except (ValueError, TypeError):
        pass
    return raw


def _is_valid_value(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() not in ("", "None", "NaN", "nan")


def _join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key
