/**
 * TrainingDataPanel — checkbox list of this session's annotated samples to
 * train on (see gatherTrainingSources.ts for the session-scoping rationale).
 */
import type { TrainingCandidate } from '@/lib/gatherTrainingSources';

interface TrainingDataPanelProps {
  candidates: TrainingCandidate[];
  selected: Set<string>;
  onToggle: (sourceKey: string) => void;
}

export default function TrainingDataPanel({ candidates, selected, onToggle }: TrainingDataPanelProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Training data</p>
        <span className="text-xs text-slate-400">{selected.size} of {candidates.length} selected</span>
      </div>
      {candidates.length === 0 ? (
        <p className="text-xs text-slate-400">
          No annotated samples yet this session. Annotate a few slices in the Annotate tab, then come back here.
        </p>
      ) : (
        <div className="max-h-48 overflow-y-auto rounded-lg border border-slate-700 divide-y divide-slate-700">
          {candidates.map((c) => (
            <label key={c.sourceKey} className="flex items-center gap-2 px-3 py-2 text-sm text-slate-200 cursor-pointer hover:bg-slate-700/40">
              <input
                type="checkbox"
                className="shrink-0 accent-sky-500"
                checked={selected.has(c.sourceKey)}
                onChange={() => onToggle(c.sourceKey)}
              />
              <span className="min-w-0 flex-1 truncate font-mono text-xs">{c.sourceKey}</span>
              <span className="shrink-0 text-xs text-slate-400">{c.shapeCount} shape{c.shapeCount !== 1 ? 's' : ''}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
