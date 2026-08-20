/**
 * useTrainRuns — saved fine-tune runs (both model families), for the
 * inference run-picker. Invalidate ``['trainRuns']`` after a training job
 * completes so a freshly-saved run shows up without a manual refresh.
 */
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { API_BASE } from '@/config';
import type { AnnotationClass } from '@/stores/classStore';

export interface TrainRun {
  run_id: string;
  // Discriminator literals from schemas.py's ModelConfig union; a saved
  // Noise2Noise/Noise2Void run carries 'dlsia_denoiser' (see isDenoiserRun).
  model_family: 'dinov3_lora' | 'dlsia_tunet' | 'dlsia_denoiser';
  model_config: Record<string, unknown>;
  classes: AnnotationClass[];
  render: Record<string, unknown>;
  /**
   * Denoising baked into this run's INPUT pixels at training time (see
   * schemas.DenoiseTrainOpts). Null for a run trained on raw pixels, and
   * absent entirely on runs saved before the option existed — hence optional
   * as well as nullable, so `run.denoise &&` is the only safe test.
   *
   * Read-only from the frontend's side: inference reapplies it off the run
   * itself, never off the request, so nothing in the app should offer it as a
   * choice at predict time. It's surfaced (RunsPanel, ApplyModelPanel) purely
   * so two runs that expect different input can't look identical.
   */
  denoise?: { method: string; strength: number } | null;
  image_size: number;
  hyperparams: Record<string, unknown>;
  source_keys: string[];
  created_at: string;
  metrics: {
    epochs_completed?: number;
    final_train_loss?: number | null;
    final_val_loss?: number | null;
    val_miou?: number | null;
    cancelled?: boolean;
  };
}

export function useTrainRuns() {
  const queryClient = useQueryClient();

  const query = useQuery<TrainRun[]>({
    queryKey: ['trainRuns'],
    queryFn: async ({ signal }) => {
      const res = await fetch(`${API_BASE}/api/train/runs`, { signal });
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: 10_000,
  });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['trainRuns'] });

  /** Permanently deletes a saved run (config, metrics, and weights) and refreshes the list. */
  const deleteRun = async (runId: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/train/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
      if (!res.ok) return false;
      invalidate();
      return true;
    } catch {
      return false;
    }
  };

  return { runs: query.data ?? [], isLoading: query.isLoading, invalidate, deleteRun };
}
