/**
 * useTrainCapability — Train-tab readiness: torch/dlsia availability, device,
 * and the DINOv3 checkpoints discovered on the server.
 */
import { useQuery } from '@tanstack/react-query';
import { API_BASE } from '@/config';

export interface DinoCheckpoint {
  file_name: string;
  arch: string;
  weights: string;
  embed_dim: number;
  size_bytes: number;
}

export interface TrainCapability {
  torch_available: boolean;
  torch_version: string | null;
  device: string | null;
  dinov3: { available: boolean; checkpoints: DinoCheckpoint[] };
  dlsia: { available: boolean };
  models_dir: string;
  runs_dir: string;
  busy: boolean;
  error?: string;
}

const FALLBACK: TrainCapability = {
  torch_available: false,
  torch_version: null,
  device: null,
  dinov3: { available: false, checkpoints: [] },
  dlsia: { available: false },
  models_dir: '',
  runs_dir: '',
  busy: false,
};

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

  return { capability: query.data ?? FALLBACK, isLoading: query.isLoading, refetch: query.refetch };
}
