/**
 * JobProgressBar — phase/progress bar + scrolling log pane + error banner,
 * shared by the training and inference sections of the Train tab. Styling
 * matches DownloadModal's export-job progress display.
 */
import { WarningCircle } from '@phosphor-icons/react';
import type { ExportJobState } from '@/hooks/useExportJob';

interface JobProgressBarProps {
  job: ExportJobState;
  /** Unit label for the done/total counter (e.g. "batches", "slices"). */
  unit?: string;
}

export default function JobProgressBar({ job, unit = 'steps' }: JobProgressBarProps) {
  if (job.status === 'idle') return null;
  const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="space-y-2">
      {(job.status === 'running' || job.status === 'done') && (
        <>
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="capitalize">{job.phase || 'working'}…</span>
            {job.total > 0 && <span className="tabular-nums">{job.done}/{job.total} {unit}</span>}
          </div>
          <div className="h-1.5 w-full rounded bg-slate-700 overflow-hidden">
            <div
              className={`h-full transition-all ${job.status === 'done' ? 'bg-green-500' : 'bg-sky-500'}`}
              style={{ width: job.status === 'done' ? '100%' : `${Math.max(5, pct)}%` }}
            />
          </div>
          {job.log.length > 0 && (
            <div className="max-h-32 overflow-y-auto rounded bg-slate-900/70 border border-slate-700 p-2 text-[11px] font-mono text-slate-400 leading-relaxed">
              {job.log.slice(-16).map((line, i) => <div key={i}>{line}</div>)}
            </div>
          )}
        </>
      )}
      {job.status === 'error' && (
        <div className="flex items-start gap-2 text-sm text-red-400">
          <WarningCircle size={16} className="mt-0.5 shrink-0" />
          <span className="min-w-0 break-words">{job.error ?? 'Job failed.'}</span>
        </div>
      )}
    </div>
  );
}
