/**
 * TrainDenoiseToggle — the "Train on denoised input" opt-in.
 *
 * It reads the setting straight from `datasetStore`'s denoise state — the same
 * one Phase 2's Annotate-tab `DenoisePanel` reads/writes — so the label names
 * the exact filter the user tuned there rather than offering a second,
 * independent copy of the controls that could disagree with what they were
 * actually looking at.
 *
 * Deliberately opt-in and default-off: unlike every other denoise control in
 * the app this one changes what the model LEARNS, and a run trained on
 * filtered pixels is not interchangeable with one trained on raw pixels.
 * Whether it's on is owned by the host (it's part of that form's submit
 * state); this component owns only the presentation and the "can it be used
 * at all" guard — the payload builder in `lib/trainDenoiseOption` re-checks
 * the guard, so a stale checked box can't leak an unusable method into a
 * request.
 */
import { useDatasetStore } from '@/stores/datasetStore';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { trainDenoiseBlockedReason, trainDenoiseSummary } from '@/lib/trainDenoiseOption';

interface TrainDenoiseToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  /** Host-level busy state (a job in flight); the guard disables independently. */
  disabled?: boolean;
}

export default function TrainDenoiseToggle({
  checked, onChange, disabled = false,
}: TrainDenoiseToggleProps) {
  const denoise = useDatasetStore((s) => s.denoise);
  const { capability } = useTrainCapability();

  const blocked = trainDenoiseBlockedReason(denoise.method);
  const summary = trainDenoiseSummary(denoise, capability.denoise.methods);

  return (
    <div className="flex flex-col gap-0.5">
      <label
        className={`flex items-start gap-1.5 text-xs text-slate-300 ${
          blocked ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'
        }`}
        title={
          blocked
            ?? 'Applies this filter to the model\'s input pixels during training. The setting is recorded on the run, and inference reapplies it automatically, so training and prediction cannot disagree.'
        }
      >
        <input
          type="checkbox"
          className="mt-0.5 shrink-0 accent-sky-500"
          checked={checked && !blocked}
          disabled={disabled || !!blocked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>
          Train on denoised input
          {summary && <span className="text-slate-400"> ({summary})</span>}
        </span>
      </label>
      {blocked ? (
        <p className="pl-5 text-[11px] leading-snug text-slate-400">{blocked}</p>
      ) : (
        checked && (
          <p className="pl-5 text-[11px] leading-snug text-slate-400">
            Recorded on the run — inference will reapply it automatically.
          </p>
        )
      )}
    </div>
  );
}
