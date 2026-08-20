/**
 * RunsPanel — saved fine-tune runs (both model families); pick one to run
 * inference with (see InferencePanel), or permanently delete one.
 */
import { Trash } from '@phosphor-icons/react';
import type { TrainRun } from '@/hooks/useTrainRuns';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { denoiseMethodLabel } from '@/lib/trainDenoiseOption';

interface RunsPanelProps {
  runs: TrainRun[];
  selectedRunId: string | null;
  onSelectRun: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
}

export const FAMILY_LABELS: Record<string, string> = {
  dinov3_lora: 'DINOv3 + LoRA',
  dlsia_tunet: 'dlsia TUNet',
  // Architecture-agnostic: this family now covers both a dlsia TUNet and a
  // plain convolutional autoencoder, and more than the two original schemes.
  dlsia_denoiser: 'Denoiser (self-supervised)',
};

export default function RunsPanel({ runs, selectedRunId, onSelectRun, onDeleteRun }: RunsPanelProps) {
  // Only to turn a stored method id ("tv") into its label ("Total variation");
  // the query is shared and cached across the whole Train tab.
  const { capability } = useTrainCapability();

  const handleDelete = (run: TrainRun) => {
    // Includes run_id, not just family + timestamp: two runs of the same family
    // trained close together can otherwise look identical in this dialog, with
    // nothing to tell the user which one they're actually about to delete.
    const label = `${FAMILY_LABELS[run.model_family] ?? run.model_family} — ${new Date(run.created_at).toLocaleString()} (${run.run_id})`;
    if (!window.confirm(`Permanently delete this run?\n\n${label}\n\nThis removes its saved weights and cannot be undone.`)) return;
    onDeleteRun(run.run_id);
  };

  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Saved runs</p>
      {runs.length === 0 ? (
        <p className="text-xs text-slate-400">No runs saved yet — train a model above to create one.</p>
      ) : (
        <div className="max-h-56 overflow-y-auto rounded-lg border border-slate-700 divide-y divide-slate-700">
          {runs.map((run) => (
            <label
              key={run.run_id}
              className={`flex items-start gap-3 px-3 py-2 cursor-pointer transition-colors ${
                selectedRunId === run.run_id ? 'bg-sky-900/30' : 'hover:bg-slate-700/40'
              }`}
            >
              <input
                type="radio" name="selected-run" className="mt-1 shrink-0 accent-sky-500"
                checked={selectedRunId === run.run_id} onChange={() => onSelectRun(run.run_id)}
              />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span className="text-sm text-slate-200">{FAMILY_LABELS[run.model_family] ?? run.model_family}</span>
                  {run.model_family === 'dinov3_lora' && (
                    <span className="text-xs text-slate-400">({String(run.model_config.arch)})</span>
                  )}
                  <span className="text-xs text-slate-500">{new Date(run.created_at).toLocaleString()}</span>
                  {/* Two runs are otherwise indistinguishable here while
                      expecting completely different input pixels. */}
                  {run.denoise && (
                    <span
                      className="rounded bg-sky-900/50 px-1.5 py-px text-[10px] font-medium text-sky-300"
                      title={`Trained on ${denoiseMethodLabel(run.denoise.method, capability.denoise.methods)}-denoised input at ${Math.round(run.denoise.strength * 100)}% strength. Inference reapplies this automatically.`}
                    >
                      denoised input · {denoiseMethodLabel(run.denoise.method, capability.denoise.methods)}
                    </span>
                  )}
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-slate-400">
                  {run.metrics.val_miou != null && <span>mIoU {run.metrics.val_miou.toFixed(3)}</span>}
                  {run.metrics.epochs_completed != null && <span>{run.metrics.epochs_completed} epochs</span>}
                  {run.metrics.cancelled && <span className="text-amber-400">cancelled (partial)</span>}
                  <span className="truncate">{run.classes.map((c) => c.label).join(', ')}</span>
                </div>
              </div>
              <button
                type="button"
                aria-label="Delete run"
                title="Delete run"
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(run); }}
                className="shrink-0 rounded p-1 text-slate-500 hover:bg-red-900/30 hover:text-red-400 transition-colors"
              >
                <Trash size={14} />
              </button>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
