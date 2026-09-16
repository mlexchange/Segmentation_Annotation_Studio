/**
 * useImageSlice — TanStack Query hook to fetch a PNG slice from the backend.
 *
 * Slices are cached as blob object URLs. Those are NOT garbage collected when the
 * query holding them is evicted — an object URL pins its blob until explicitly
 * revoked — so `installImageSliceGc` must be wired up once at startup, or every
 * slice the user visits stays resident for the lifetime of the tab.
 */
import { useQuery, type QueryClient } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import { RenderOpts, type DenoiseOpts } from '@/stores/datasetStore';

const SLICE_QUERY_KEY = 'imageSlice';

export interface SliceResult {
  url: string;
}

/** Build the /api/image/slice URL encoding source, slice index, and render options.
 *
 * `denoise` is optional and omitted from the URL entirely when off, so the
 * un-denoised request stays byte-identical to what it has always been — the
 * backend only pays (and caches) the filter cost when a method is actually set.
 *
 * `crop`, when set, asks the backend to filter and return only a centred square
 * of that size at 1:1. Filtering a full slice costs seconds for NLM and TV;
 * cropping keeps slider-dragging interactive (measured: 7.3s -> 0.49s for
 * bilateral on a 2560² slice).
 */
export function buildSliceUrl(
  source: string,
  kind: string,
  sliceIndex: number,
  renderOpts: RenderOpts,
  serverUri: string | null,
  denoise?: DenoiseOpts | null,
  crop?: number
): string {
  const params = new URLSearchParams({
    source,
    kind,
    slice_index: String(sliceIndex),
    norm: renderOpts.norm,
    scale: renderOpts.scale,
    vmin_pct: String(renderOpts.vminPct),
    vmax_pct: String(renderOpts.vmaxPct),
    cmap: renderOpts.cmap,
  });
  if (serverUri) params.set('server_uri', serverUri);
  if (denoise && denoise.method !== 'none') {
    params.set('denoise_method', denoise.method);
    params.set('denoise_strength', String(denoise.strength));
    if (crop && crop > 0) params.set('denoise_crop', String(crop));
  }
  return `${API_BASE}/api/image/slice?${params.toString()}`;
}

/**
 * Fetches a PNG slice as a blob object URL (5min stale time). Disabled until
 * both source and kind are set. Use for displaying a rendered image slice.
 */
export function useImageSlice(
  source: string | null,
  kind: string | null,
  sliceIndex: number,
  renderOpts: RenderOpts,
  serverUri: string | null,
  denoise?: DenoiseOpts | null
) {
  const enabled = Boolean(source && kind);
  const url = enabled
    ? buildSliceUrl(source!, kind!, sliceIndex, renderOpts, serverUri, denoise)
    : null;

  // Denoise is keyed only when active, so switching it off returns to the exact
  // cache entry the un-denoised view already had rather than refetching.
  const denoiseKey = denoise && denoise.method !== 'none'
    ? [denoise.method, denoise.strength]
    : null;

  return useQuery({
    queryKey: [SLICE_QUERY_KEY, source, kind, sliceIndex, renderOpts, serverUri, denoiseKey],
    queryFn: async () => {
      const res = await fetch(url!);
      if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`);
      const blob = await res.blob();
      return URL.createObjectURL(blob);
    },
    enabled,
    staleTime: 1000 * 300,
  });
}

/**
 * Revoke a slice's object URL when its query leaves the cache, so paging through a
 * volume doesn't retain every PNG for the session. Without this, the browser holds
 * each blob alive indefinitely — a few hundred slices is easily hundreds of MB, and
 * the resulting GC pressure shows up as the whole tab getting slower the longer it
 * is used.
 *
 * Also revokes the previous URL when a query's data is replaced (a refetch of the
 * same slice), which would otherwise orphan the old blob.
 *
 * Call once, next to the QueryClient. Returns the unsubscribe function.
 */
export function installImageSliceGc(queryClient: QueryClient): () => void {
  const isSliceQuery = (key: readonly unknown[]) => key[0] === SLICE_QUERY_KEY;
  const revoke = (value: unknown) => {
    if (typeof value === 'string' && value.startsWith('blob:')) URL.revokeObjectURL(value);
  };
  // Track the last-seen URL per query so an 'updated' event can revoke the one
  // being replaced (the event carries the new state, not the old).
  const seen = new Map<string, string>();

  return queryClient.getQueryCache().subscribe((event) => {
    const { query } = event;
    if (!isSliceQuery(query.queryKey)) return;
    const hash = query.queryHash;

    if (event.type === 'removed') {
      revoke(query.state.data);
      seen.delete(hash);
      return;
    }
    const data = query.state.data;
    if (typeof data === 'string') {
      const prev = seen.get(hash);
      if (prev && prev !== data) revoke(prev);
      seen.set(hash, data);
    }
  });
}
