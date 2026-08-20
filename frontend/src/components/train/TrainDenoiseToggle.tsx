/**
 * TrainDenoiseToggle — the "Train on denoised input" opt-in, shared by the two
 * places that start a segmentation job (TrainPage's "Start training" and
 * ApplyModelPanel's "Fine-tune & apply").
 *
 * It reads the setting straight from the Annotate tab's denoise store, so the
 * label names the exact filter the user tuned there rather than offering a
 * second, independent copy of the controls that could disagree with what they
 * were actually looking at.
 *
 * Deliberately opt-in and default-off in both hosts: unlike every other denoise
 * control in the app this one changes what the model LEARNS, and a run trained
 * on filtered pixels is not interchangeable with one trained on raw pixels.
 * Whether it's on is owned by the host (it's part of that form's submit state);
 * this component owns only the presentation and the "can it be used at all"
 * guard — the payload builder in `lib/trainDenoiseOption` re-checks the guard,
 * so a stale checked box can't leak an unusable method into a request.
 *
 * `variant` exists because the two hosts sit on opposite backgrounds: the Train
 * tab is dark slate, the Annotate sidebar is white.
 */
import { useDenoiseStore } from '@/stores/denoiseStore';
import { useTrainCapability } from '@/hooks/useTrainCapability';
import { trainDenoiseBlockedReason, trainDenoiseSummary } from '@/lib/trainDenoiseOption';

interface TrainDenoiseToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  /** Host-level busy state (a job in flight); the guard disables independently. */
  disabled?: boolean;
  variant?: 'dark' | 'light';
}

const STYLES = {
  dark: { label: 'text-slate-300', note: 'text-slate-400', accent: 'accent-sky-500' },
  light: { label: 'text-gray-600', note: 'text-gray-400', accent: 'accent-violet-600' },
} as const;

export default function TrainDenoiseToggle({
  checked, onChange, disabled = false, variant = 'dark',
}: TrainDenoiseToggleProps) {
  const denoise = useDenoiseStore((s) => s.denoise);
  const { capability } = useTrainCapability();

  const blocked = trainDenoiseBlockedReason(denoise.method);
  const summary = trainDenoiseSummary(denoise, capability.denoise.methods);
  const style = STYLES[variant];

  return (
    <div className="flex flex-col gap-0.5">
      <label
        className={`flex items-start gap-1.5 text-xs ${style.label} ${
          blocked ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'
        }`}
        title={
          blocked
            ?? 'Applies this filter to the model\'s input pixels during training. The setting is recorded on the run, and inference reapplies it automatically, so training and prediction cannot disagree.'
        }
      >
        <input
          type="checkbox"
          className={`mt-0.5 shrink-0 ${style.accent}`}
          checked={checked && !blocked}
          disabled={disabled || !!blocked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>
          Train on denoised input
          {summary && <span className={style.note}> ({summary})</span>}
        </span>
      </label>
      {blocked ? (
        <p className={`pl-5 text-[11px] leading-snug ${style.note}`}>{blocked}</p>
      ) : (
        checked && (
          <p className={`pl-5 text-[11px] leading-snug ${style.note}`}>
            Recorded on the run — inference will reapply it automatically.
          </p>
        )
      )}
    </div>
  );
}
