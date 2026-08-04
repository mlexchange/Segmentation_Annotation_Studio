/**
 * useVolume — TanStack Query hook to fetch a downsampled uint8 intensity
 * volume from `/api/image/volume` for the 3D tab.
 *
 * Mirrors `useImageSlice.ts`'s shape (a `buildXUrl` helper + a thin query
 * hook), but the response is raw octet-stream bytes rather than a PNG blob,
 * and dimensions travel out-of-band in the `X-Volume-Meta` response header
 * (see `backend/annotation_server.py`'s `image_volume` route) instead of
 * being encoded in the body. A volume has no per-slice index and no color
 * map — it's the whole downsampled stack, rendered client-side — so those
 * two `RenderOpts` fields are intentionally absent from both the URL and the
 * query key.
 */
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { formatApiError } from '@/lib/apiError';
import { volumeDimsFor, type VolumeDims } from '@/lib/volume/volumeDims';

export interface VolumeResult {
  data: Uint8Array;
  dims: VolumeDims;
  skippedZ: number[];
}

interface VolumeRenderOpts {
  norm: string;
  scale: string;
  vminPct: number;
  vmaxPct: number;
}

interface VolumeMetaHeader {
  nz: number;
  ny: number;
  nx: number;
  sxy: number;
  sz: number;
  n_slices: number;
  height: number;
  width: number;
  skipped_z?: number[];
}

/** Build the /api/image/volume URL. Mirrors `buildSliceUrl` in
 *  `useImageSlice.ts`, minus `slice_index`/`cmap` — a volume has no color map
 *  or single slice, it's the whole downsampled stack. */
export function buildVolumeUrl(
  source: string,
  kind: string,
  maxDim: number,
  renderOpts: VolumeRenderOpts,
  serverUri: string | null
): string {
  const params = new URLSearchParams({
    source,
    kind,
    max_dim: String(maxDim),
    norm: renderOpts.norm,
    scale: renderOpts.scale,
    vmin_pct: String(renderOpts.vminPct),
    vmax_pct: String(renderOpts.vmaxPct),
  });
  if (serverUri) params.set('server_uri', serverUri);
  return `${API_BASE}/api/image/volume?${params.toString()}`;
}

/**
 * Fetches the downsampled raw intensity volume for the currently-open
 * dataset (5min stale/gc time — these payloads are tens of MB, so they
 * should not linger in the cache longer than that). Disabled until both
 * `source` and `kind` are set.
 */
export function useVolume(
  source: string | null,
  kind: string | null,
  serverUri: string | null,
  maxDim: number,
  renderOpts: VolumeRenderOpts
): UseQueryResult<VolumeResult> {
  const enabled = Boolean(source && kind);

  return useQuery({
    queryKey: ['imageVolume', source, kind, serverUri, maxDim, renderOpts.norm, renderOpts.scale, renderOpts.vminPct, renderOpts.vmaxPct],
    queryFn: async ({ signal }): Promise<VolumeResult> => {
      const url = buildVolumeUrl(source!, kind!, maxDim, renderOpts, serverUri);
      const res = await fetch(url, { signal });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(formatApiError(body, `Volume fetch failed: ${res.status}`));
      }

      const metaHeader = res.headers.get('X-Volume-Meta');
      if (!metaHeader) {
        throw new Error(
          'Volume fetch succeeded but the X-Volume-Meta response header was missing — cannot ' +
          'determine volume dimensions without it (a proxy may have stripped it).'
        );
      }

      let meta: VolumeMetaHeader;
      try {
        meta = JSON.parse(metaHeader);
      } catch {
        throw new Error('Volume fetch succeeded but X-Volume-Meta was not valid JSON.');
      }

      const dims: VolumeDims = { nz: meta.nz, ny: meta.ny, nx: meta.nx, sxy: meta.sxy, sz: meta.sz };

      // Cross-check against the locally-computed formula purely as a
      // trust-but-verify sanity check; the header's own values always win
      // since they describe what the body actually contains.
      const expected = volumeDimsFor(meta.n_slices, meta.height, meta.width, maxDim);
      if (
        expected.nz !== dims.nz || expected.ny !== dims.ny || expected.nx !== dims.nx ||
        expected.sxy !== dims.sxy || expected.sz !== dims.sz
      ) {
        console.warn(
          'useVolume: X-Volume-Meta dims disagree with volumeDimsFor — trusting the header.',
          { header: dims, computed: expected }
        );
      }

      const buffer = await res.arrayBuffer();
      const data = new Uint8Array(buffer);
      const expectedBytes = dims.nz * dims.ny * dims.nx;
      if (data.byteLength !== expectedBytes) {
        throw new Error(
          `Volume payload size mismatch: got ${data.byteLength} bytes, expected ` +
          `${expectedBytes} (nz*ny*nx = ${dims.nz}*${dims.ny}*${dims.nx}) — dims-contract bug ` +
          'between the backend volume route and this hook.'
        );
      }

      return { data, dims, skippedZ: meta.skipped_z ?? [] };
    },
    enabled,
    staleTime: 300_000,
    gcTime: 300_000,
  });
}
