/**
 * PerfOverlay — dev-only timing HUD for the annotation workspace.
 *
 * Rendered only when `?perf=1` (see lib/perf.ts). Shows rolling p50/p95 per
 * instrumented path plus the current shape count, so an optimization can be
 * confirmed on real data — the slices that matter are far larger than anything
 * reproducible in a test.
 */
import { useSyncExternalStore, useState } from 'react';
import { X, ArrowCounterClockwise } from '@phosphor-icons/react';
import { snapshot, subscribe, getVersion, resetPerf, type PerfStat } from '@/lib/perf';

interface PerfOverlayProps {
  /** Shapes on the current slice — the main driver of the costs below. */
  shapeCount: number;
}

/** Colour by how close a p95 is to a dropped frame (16.7ms). */
function severity(ms: number): string {
  if (ms >= 50) return 'text-red-400';
  if (ms >= 16.7) return 'text-amber-300';
  return 'text-emerald-300';
}

function Row({ stat }: { stat: PerfStat }) {
  return (
    <tr>
      <td className="pr-3 text-gray-300">{stat.label}</td>
      <td className={`pr-3 text-right tabular-nums ${severity(stat.p50)}`}>{stat.p50.toFixed(1)}</td>
      <td className={`pr-3 text-right tabular-nums ${severity(stat.p95)}`}>{stat.p95.toFixed(1)}</td>
      <td className="text-right tabular-nums text-gray-400">{stat.count}</td>
    </tr>
  );
}

export default function PerfOverlay({ shapeCount }: PerfOverlayProps) {
  const [hidden, setHidden] = useState(false);
  // Re-render when new samples land (the version counter is the store value).
  useSyncExternalStore(subscribe, getVersion, () => 0);
  const stats = snapshot();

  if (hidden) return null;

  return (
    <div className="absolute bottom-2 right-2 z-30 rounded-md bg-gray-900/90 text-[11px] text-gray-200 shadow-lg backdrop-blur px-3 py-2 font-mono pointer-events-auto">
      <div className="flex items-center justify-between gap-4 mb-1">
        <span className="font-semibold text-sky-300">perf</span>
        <span className="text-gray-400">{shapeCount} shapes</span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={resetPerf}
            title="Reset samples"
            className="p-0.5 rounded hover:bg-white/10"
          >
            <ArrowCounterClockwise size={12} />
          </button>
          <button
            type="button"
            onClick={() => setHidden(true)}
            title="Hide (reload to show again)"
            className="p-0.5 rounded hover:bg-white/10"
          >
            <X size={12} />
          </button>
        </div>
      </div>
      {stats.length === 0 ? (
        <div className="text-gray-500">no samples yet — draw or zoom</div>
      ) : (
        <table>
          <thead>
            <tr className="text-gray-500">
              <th className="pr-3 text-left font-normal">path</th>
              <th className="pr-3 text-right font-normal">p50</th>
              <th className="pr-3 text-right font-normal">p95</th>
              <th className="text-right font-normal">n</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => <Row key={s.label} stat={s} />)}
          </tbody>
        </table>
      )}
    </div>
  );
}
