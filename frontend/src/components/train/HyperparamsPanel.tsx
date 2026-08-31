/**
 * HyperparamsPanel — training hyperparameters for the dlsia TUNet family
 * (epochs, lr, batch size, image size, flip augment, depth/base_channels/
 * growth_rate). Collapsed behind <details> — sensible defaults are supplied
 * by the backend schema, so most users never open it.
 *
 * DINOv3 LoRA is deferred (see Phase 5.5), so there is only one family here —
 * no model-family switch, no LoRA rank/alpha fields.
 */
import { IMAGE_SIZE_CONSTRAINTS, validateImageSize } from '@/lib/trainConstraints';

export interface HyperparamsState {
  epochs: number;
  lr: number;
  batch_size: number;
  image_size: number;
  flip_augment: boolean;
  /** Cut native-resolution `image_size` windows out of each slice instead of
   *  shrinking whole slices to `image_size`. */
  tiling: boolean;
  depth: number;
  base_channels: number;
  growth_rate: number;
}

interface HyperparamsPanelProps {
  values: HyperparamsState;
  onChange: (updates: Partial<HyperparamsState>) => void;
  runName: string;
  onRunNameChange: (name: string) => void;
  /** Measure the largest batch size this config fits (see backend/batch_probe.py). */
  onEstimateBatch: () => void;
  /** Request cancellation of the running probe (cooperative — see /api/train/cancel). */
  onCancelEstimate: () => void;
  /** True while the probe job runs — it holds the device, so it can't overlap. */
  estimatingBatch: boolean;
  /** Progress/result line for the probe, shown under the field. */
  batchEstimateNote: string | null;
  /** Why Estimate can't run right now (shown as the button's tooltip), or null when it can. */
  estimateDisabledReason: string | null;
}

const inputClass =
  'w-full rounded-md border border-slate-600 bg-slate-900/60 px-2.5 py-1.5 text-sm text-slate-200 focus:border-sky-500 focus:outline-none';
const labelClass = 'block text-xs text-slate-400 mb-1';

export default function HyperparamsPanel({
  values, onChange, runName, onRunNameChange,
  onEstimateBatch, onCancelEstimate, estimatingBatch, batchEstimateNote, estimateDisabledReason,
}: HyperparamsPanelProps) {
  const sizeLimits = IMAGE_SIZE_CONSTRAINTS.dlsia_tunet;
  const sizeError = validateImageSize(
    'dlsia_tunet', values.image_size, values.tiling ? 'Patch size' : 'Image size',
  );

  return (
    <details className="rounded-lg border border-slate-700 bg-slate-800/40">
      <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-slate-300 uppercase tracking-wide select-none">
        Hyperparameters (advanced)
      </summary>
      <div className="grid grid-cols-2 gap-3 px-3 pb-3 pt-1 sm:grid-cols-3">
        <div>
          <label className={labelClass}>Run name (optional)</label>
          <input
            type="text" value={runName} onChange={(e) => onRunNameChange(e.target.value)}
            placeholder="auto-generated" className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>Epochs</label>
          <input
            type="number" min={1} value={values.epochs}
            onChange={(e) => onChange({ epochs: Number(e.target.value) })} className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>Learning rate</label>
          <input
            type="number" step="0.0001" min={0} value={values.lr}
            onChange={(e) => onChange({ lr: Number(e.target.value) })} className={inputClass}
          />
        </div>
        <div>
          <div className="flex items-baseline justify-between gap-2 mb-1">
            <label className="block text-xs text-slate-400">Batch size</label>
            {/* Measures the real ceiling by running actual training steps, so it
                needs the device to itself and can't run during another job. */}
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={onEstimateBatch}
                disabled={estimatingBatch || !!estimateDisabledReason}
                title={
                  estimateDisabledReason
                  ?? 'Run real training steps at increasing batch sizes to find the largest that fits'
                }
                className="text-[10px] px-1.5 py-0.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {estimatingBatch ? 'Estimating…' : 'Estimate max'}
              </button>
              {estimatingBatch && (
                <button
                  type="button"
                  onClick={onCancelEstimate}
                  className="text-[10px] px-1.5 py-0.5 rounded border border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
          <input
            type="number" min={1} value={values.batch_size}
            onChange={(e) => onChange({ batch_size: Number(e.target.value) })} className={inputClass}
          />
          {batchEstimateNote && (
            <p className="mt-1 text-[10px] leading-snug text-slate-400">{batchEstimateNote}</p>
          )}
        </div>
        <div>
          {/* Same number, two meanings: the tile window when tiling, or the square
              the whole slice is squashed into when not. */}
          <label className={labelClass}>{values.tiling ? 'Patch size (px)' : 'Image size (px)'}</label>
          <input
            type="number"
            step={sizeLimits.multipleOf}
            min={sizeLimits.min}
            max={sizeLimits.max}
            value={values.image_size}
            onChange={(e) => onChange({ image_size: Number(e.target.value) })}
            className={`${inputClass} ${sizeError ? 'border-red-500' : ''}`}
          />
          <p className={`mt-1 text-[10px] ${sizeError ? 'text-red-400' : 'text-slate-500'}`}>
            {sizeError ?? `${sizeLimits.min}–${sizeLimits.max}, in steps of ${sizeLimits.multipleOf}`}
          </p>
        </div>
        <div className="flex items-end pb-1.5">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox" checked={values.flip_augment} className="accent-sky-500"
              onChange={(e) => onChange({ flip_augment: e.target.checked })}
            />
            Random flip augmentation
          </label>
        </div>

        <div className="col-span-2 sm:col-span-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox" checked={values.tiling} className="accent-sky-500"
              onChange={(e) => onChange({ tiling: e.target.checked })}
            />
            Tile large images (train &amp; predict at native resolution)
          </label>
          <p className="mt-1 text-[11px] leading-snug text-slate-400">
            {values.tiling
              ? `Cuts ${values.image_size}px patches at full resolution with 25% overlap, then blends the
                 predictions back together. Keeps fine detail on images larger than the patch size.`
              : `Shrinks each whole image to ${values.image_size}px before training — faster, but detail on
                 large images is lost before the model sees it.`}
          </p>
        </div>

        <div>
          <label className={labelClass}>Depth</label>
          <input
            type="number" min={2} max={6} value={values.depth}
            onChange={(e) => onChange({ depth: Number(e.target.value) })} className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>Base channels</label>
          <input
            type="number" min={1} value={values.base_channels}
            onChange={(e) => onChange({ base_channels: Number(e.target.value) })} className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>Growth rate</label>
          <input
            type="number" step="0.1" min={0.1} value={values.growth_rate}
            onChange={(e) => onChange({ growth_rate: Number(e.target.value) })} className={inputClass}
          />
        </div>
      </div>
    </details>
  );
}
