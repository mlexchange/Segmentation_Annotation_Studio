/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 *
 * Loud temporary markers for Finch-scaffold label/model pickers.
 * Delete this module once you wire your own taxonomy + model UX.
 */

/** Exact flag string — keep identical in comments + UI. */
export const REMOVE_THIS_AND_USE_YOUR_OWN_STUFF =
  '# REMOVE THIS AND USE YOUR OWN STUFF';

export interface KillSwitchBannerProps {
  /** One-line hint: what this block is (labels, models, …). */
  detail?: string;
  className?: string;
}

/** Big orange strip so scaffold panels are impossible to miss. */
export function KillSwitchBanner({ detail, className = '' }: KillSwitchBannerProps) {
  return (
    <div
      role="status"
      data-testid="kill-switch-banner"
      className={[
        'rounded-md border-2 border-amber-600 bg-amber-100 px-2.5 py-2 shadow-sm',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <p className="font-mono text-[12px] font-black leading-tight tracking-wide text-amber-950">
        {REMOVE_THIS_AND_USE_YOUR_OWN_STUFF}
      </p>
      {detail ? (
        <p className="mt-1 text-[10px] font-medium leading-snug text-amber-900/90">{detail}</p>
      ) : null}
    </div>
  );
}
