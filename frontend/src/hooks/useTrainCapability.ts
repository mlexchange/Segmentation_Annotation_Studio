/**
 * useTrainCapability — Train-tab readiness: torch/dlsia availability and device.
 *
 * No DINOv3 checkpoint discovery here — that model family is deferred (see
 * Phase 5.5 in the integration plan) and the backend's capability() probe
 * doesn't report it at all.
 */
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';

/** A classical denoise filter the server can actually run. `available` is
 *  per-method because some need an optional dependency (wavelet ⇢ PyWavelets),
 *  and `cost` drives whether the UI warns before a full-resolution preview. */
export interface DenoiseMethodInfo {
  method: string;
  label: string;
  cost: 'cheap' | 'moderate' | 'slow';
  description: string;
  available: boolean;
  z_radius: number;
}

export interface TrainCapability {
  torch_available: boolean;
  torch_version: string | null;
  device: string | null;
  dlsia: { available: boolean };
  denoise: { available: boolean; methods: DenoiseMethodInfo[] };
  runs_dir: string;
  busy: boolean;
  error?: string;
}

const FALLBACK: TrainCapability = {
  torch_available: false,
  torch_version: null,
  device: null,
  dlsia: { available: false },
  denoise: { available: false, methods: [] },
  runs_dir: '',
  busy: false,
};

/**
 * Merge a server response over FALLBACK so a key this build expects but the
 * server doesn't send resolves to a safe default instead of `undefined`.
 *
 * `data ?? FALLBACK` alone only covers "no response at all". It does NOT cover
 * a response that's missing a key — and consumers reach into nested keys
 * (`capability.denoise.methods`), so a single absent key throws a TypeError
 * mid-render, which unmounts the React tree and white-screens the whole app.
 * That's reachable whenever the frontend is newer than the running backend: a
 * dev server still running a process started before a capability key was
 * added, or a stale deploy. A missing capability should degrade to "that
 * feature is unavailable", never take down the page.
 */
function withDefaults(data: Partial<TrainCapability> | undefined): TrainCapability {
  if (!data) return FALLBACK;
  return {
    ...FALLBACK,
    ...data,
    dlsia: { ...FALLBACK.dlsia, ...(data.dlsia ?? {}) },
    denoise: { ...FALLBACK.denoise, ...(data.denoise ?? {}) },
  };
}

/** Polls /api/train/capability every 10s — cheap enough to keep the
 *  CapabilityBanner and "busy" state current while the Train tab is open. */
export function useTrainCapability() {
  const query = useQuery<TrainCapability>({
    queryKey: ['trainCapability'],
    queryFn: async ({ signal }) => {
      const res = await fetch(`${API_BASE}/api/train/capability`, { signal });
      if (!res.ok) throw new Error(`Capability check failed: ${res.status}`);
      return res.json();
    },
    staleTime: 5_000,
    refetchInterval: 10_000,
  });

  return {
    capability: withDefaults(query.data),
    isLoading: query.isLoading,
    refetch: query.refetch,
  };
}
