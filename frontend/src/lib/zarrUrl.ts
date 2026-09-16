/**
 * zarrUrl — address a Tiled node as a Zarr store for the WebGPU volume viewer.
 *
 * Tiled 0.2.12 mounts a Zarr v2 router at `/zarr/v2`, so every array already in
 * the catalog is *already* readable as a Zarr store — `.zattrs`/`.zarray` are
 * synthesized from the Tiled structure and `/{i.j.k}` serves a chunk. Nothing
 * has to be exported or copied: the 3D view reads the same catalog Browse and
 * Annotate read.
 *
 * Two properties this module exists to preserve:
 *
 *   1. **The port is never hardcoded.** `start_all.sh` picks a free port for
 *      Tiled (default 8010, reassigned when busy) and the resolved URI reaches
 *      the frontend through `GET /api/config/servers` / `datasetStore.serverUri`.
 *      A literal `8010` anywhere here breaks the moment that fallback fires.
 *   2. **No API key goes to the browser.** `tiled/config.yml` sets
 *      `allow_anonymous_access: true` and anonymous access is read-only; the
 *      generated key authorizes *writes* and stays server-side by design. Zarr
 *      chunk reads are reads, so the renderer's plain `fetch()` needs no
 *      credentials — and must not be given any.
 *
 * Pure and dependency-free so the URL algebra is unit-testable; the React side
 * lives in `useZarrUrl`.
 */

/** Why a source cannot be opened as a Zarr volume, when it cannot. */
export type ZarrUnavailable =
  /** Local files are read off disk by the backend; they are not in the catalog. */
  | 'local-source'
  /** Nothing is open yet. */
  | 'no-source'
  /** Kind is 'tiled' but no server URI has been resolved yet. */
  | 'no-server'
  /** Still asking the backend which node holds the volume. */
  | 'resolving'
  /** The dataset exists but has no multiscale volume to render. */
  | 'no-volume';

export interface ZarrUrlResult {
  /** Zarr store root, or `null` when unavailable. */
  url: string | null;
  reason: ZarrUnavailable | null;
}

/** Human-readable explanation for a `ZarrUnavailable`, for the empty state. */
export function describeUnavailable(reason: ZarrUnavailable): string {
  switch (reason) {
    case 'no-source':
      return 'Open a dataset from Browse to view it in 3D.';
    case 'local-source':
      return 'The 3D view reads from the Tiled catalog. Ingest this file first, then reopen it.';
    case 'no-server':
      return 'Still resolving the Tiled server address — one moment.';
    case 'resolving':
      return 'Looking for this dataset’s 3D volume…';
    case 'no-volume':
      return 'No 3D volume has been built for this dataset yet.';
  }
}

/**
 * The `/zarr/v2` root for a Tiled server URI.
 *
 * Accepts a URI with or without a trailing `/api/v1`: our backend reports the
 * bare origin (`http://127.0.0.1:8010`), but a Tiled URI copied from elsewhere
 * often carries the REST prefix, and silently producing
 * `.../api/v1/zarr/v2/...` would 404 in a way that looks like missing data
 * rather than a malformed URL.
 */
export function zarrRootFor(serverUri: string): string {
  const trimmed = serverUri.trim().replace(/\/+$/, '');
  const base = trimmed.replace(/\/api\/v\d+$/, '');
  return `${base}/zarr/v2`;
}

/**
 * Zarr store URL for a dataset's volume.
 *
 * @param kind      `datasetStore.kind` — 'tiled' or 'local'.
 * @param source    Tiled path of the node that **holds the volume** — not
 *   necessarily the open dataset. A per-slice TIFF stack keeps its volume in a
 *   `__volume` sidecar, and a pyramid level's volume is its parent, so this must
 *   be the path `GET /api/volume/resolve` returned. Pointing the viewer at
 *   whatever happens to be open is what produces
 *   `openOmeZarr: missing multiscales in root .zattrs`.
 * @param serverUri Resolved Tiled server URI (see the port note above).
 */
export function buildZarrUrl(
  kind: string | null,
  source: string | null,
  serverUri: string | null,
): ZarrUrlResult {
  if (!kind || !source) return { url: null, reason: 'no-source' };
  if (kind !== 'tiled') return { url: null, reason: 'local-source' };
  if (!serverUri) return { url: null, reason: 'no-server' };

  // Encode each segment separately: the path is a `/`-joined chain of Tiled
  // keys, and encodeURIComponent on the whole thing would escape the separators.
  const path = source
    .split('/')
    .filter(Boolean)
    .map(encodeURIComponent)
    .join('/');

  return { url: `${zarrRootFor(serverUri)}/${path}`, reason: null };
}

/**
 * Zarr store URL for a sample's mask/annotation volume — the
 * `<stem>__masks<suffix>` sibling container `tiled_mask_sync.write_masks_to_tiled`
 * writes, registered as a real OME-NGFF multiscale node by
 * `mask_pyramid.register_mask_pyramid` so this is directly loadable via the
 * volume viewer's mask layer (`loadMask(slot, url)`).
 *
 * @param source Tiled path of the annotated dataset — same `source` passed to
 *   `buildZarrUrl` for the primary volume, NOT the `__masks` container itself;
 *   the `__masks`/`semantic` suffix is appended here.
 * @param suffix Distinguishes independent mask producers for the same source
 *   that must not merge into one container — `''` for the manual "sync masks
 *   to Tiled" action (iPred's fast results), `'_deep'` for a dlsia run's
 *   "Write masks to Tiled" (`infer_jobs.py`'s `container_suffix="_deep"`).
 *   Must match the backend suffix exactly or this points at an empty/missing
 *   container.
 */
export function buildMaskZarrUrl(
  kind: string | null,
  source: string | null,
  serverUri: string | null,
  suffix: '' | '_deep' = '',
): ZarrUrlResult {
  if (!kind || !source) return { url: null, reason: 'no-source' };
  if (kind !== 'tiled') return { url: null, reason: 'local-source' };
  if (!serverUri) return { url: null, reason: 'no-server' };

  const parts = source.split('/').filter(Boolean);
  const stem = parts.pop();
  if (!stem) return { url: null, reason: 'no-source' };
  const path = [...parts, `${stem}__masks${suffix}`, 'semantic']
    .map(encodeURIComponent)
    .join('/');

  return { url: `${zarrRootFor(serverUri)}/${path}`, reason: null };
}
