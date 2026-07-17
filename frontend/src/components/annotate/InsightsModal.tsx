/**
 * InsightsModal — dataset health + QA for the open sample, as a popup (like the
 * download dialog). Class balance, slice coverage, and click-to-navigate QA
 * flags (slivers, self-intersections, cross-class overlaps, empty-unmarked
 * slices). Stats are computed off the render path (deferred) so the popup opens
 * instantly, then fills in.
 */
import { useEffect, useMemo, useState } from 'react';
import { ChartBar, Warning, CheckCircle, Stack, ArrowRight, X, CircleDashed } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { computeSampleStats, type QAFlag, type SampleStats } from '@/lib/datasetStats';

interface InsightsModalProps {
  sourceKey: string | null;
  onClose: () => void;
  /** Jump to a slice and (optionally) zoom to + highlight a region on the canvas. */
  onFocus: (slice: number, bbox?: { x: number; y: number; w: number; h: number }) => void;
}

const FLAG_LABEL: Record<QAFlag['kind'], string> = {
  sliver: 'Tiny regions',
  'self-intersection': 'Self-intersecting polygons',
  overlap: 'Cross-class overlaps',
  'empty-unmarked': 'Empty, not marked negative',
};

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
      <div className="text-xl font-semibold tabular-nums text-gray-900">{value}</div>
      <div className="text-xs text-gray-600">{label}</div>
      {hint && <div className="text-[11px] text-gray-400">{hint}</div>}
    </div>
  );
}

export default function InsightsModal({ sourceKey, onClose, onFocus }: InsightsModalProps) {
  const { meta } = useDatasetStore();
  const byImage = useAnnotationStore((s) => s.byImage);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const { classes } = useClassStore();

  const [stats, setStats] = useState<SampleStats | null>(null);

  // Cheap session totals (counts only).
  const session = useMemo(() => {
    let samples = 0, shapes = 0;
    for (const slices of Object.values(byImage)) {
      const n = Object.values(slices).reduce((a, s) => a + s.length, 0);
      if (n > 0) { samples++; shapes += n; }
    }
    return { samples, shapes };
  }, [byImage]);

  // Compute stats off the render/click path so the popup paints immediately.
  useEffect(() => {
    if (!sourceKey || !meta) { setStats(null); return; }
    setStats(null);
    let cancelled = false;
    const id = window.setTimeout(() => {
      if (cancelled) return;
      const s = computeSampleStats(
        byImage[sourceKey] ?? {}, classes, meta.nSlices,
        negativeSlices[sourceKey] ?? [], meta.width, meta.height,
      );
      if (!cancelled) setStats(s);
    }, 0);
    return () => { cancelled = true; window.clearTimeout(id); };
  }, [sourceKey, meta, byImage, classes, negativeSlices]);

  const goToFlag = (f: QAFlag) => { onFocus(f.slice, f.bbox); onClose(); };

  const maxArea = stats ? Math.max(1, ...stats.classStats.map((c) => c.pixelArea)) : 1;
  const flagsByKind = stats
    ? stats.flags.reduce<Record<string, QAFlag[]>>((acc, f) => { (acc[f.kind] ??= []).push(f); return acc; }, {})
    : {};
  const totalFlags = stats?.flags.length ?? 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-4">
          <div className="flex items-center gap-2 font-semibold text-gray-900">
            <ChartBar size={18} /> Dataset Insights
          </div>
          <button type="button" onClick={onClose} className="text-gray-400 transition-colors hover:text-gray-700">
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col gap-5 overflow-y-auto px-5 py-4">
          {!meta || !sourceKey ? (
            <p className="py-8 text-center text-sm text-gray-500">Open a sample to see its dataset health.</p>
          ) : !stats ? (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-gray-500">
              <CircleDashed size={18} className="animate-spin" /> Analyzing sample…
            </div>
          ) : (
            <>
              {/* Coverage + session tiles */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Annotated slices" value={stats.coverage.annotatedSlices} hint={`of ${stats.coverage.totalSlices}`} />
                <Stat label="Negative slices" value={stats.coverage.negativeSlices} />
                <Stat label="Empty, unmarked" value={stats.coverage.emptyUnmarked.length} />
                <Stat label="Samples this session" value={session.samples} hint={`${session.shapes} shapes`} />
              </div>

              {/* Class balance */}
              <section className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                  <ChartBar size={16} /> Class balance
                </div>
                {stats.classStats.length === 0 ? (
                  <p className="text-sm text-gray-500">No classes defined.</p>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    {stats.classStats.map((c) => (
                      <div key={c.classId} className="flex items-center gap-2">
                        <span className="h-3 w-3 flex-shrink-0 rounded-sm border border-black/10" style={{ backgroundColor: c.color }} />
                        <span className="w-24 flex-shrink-0 truncate text-sm text-gray-800">{c.label}</span>
                        <div className="relative h-4 flex-1 overflow-hidden rounded bg-gray-100">
                          <div className="h-full rounded" style={{ width: `${(c.pixelArea / maxArea) * 100}%`, backgroundColor: c.color, opacity: 0.55 }} />
                        </div>
                        <span className="w-28 flex-shrink-0 text-right text-xs tabular-nums text-gray-600">
                          {c.shapeCount} · {formatArea(c.pixelArea)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* QA flags */}
              <section className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                  {totalFlags === 0 ? <CheckCircle size={16} className="text-emerald-500" /> : <Warning size={16} className="text-amber-500" />}
                  Quality checks {totalFlags > 0 && <span className="text-gray-400">({totalFlags})</span>}
                </div>
                {totalFlags === 0 ? (
                  <p className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
                    <CheckCircle size={15} /> No issues found on this sample.
                  </p>
                ) : (
                  <div className="flex flex-col gap-3">
                    {(Object.keys(flagsByKind) as QAFlag['kind'][]).map((kind) => (
                      <div key={kind}>
                        <div className="mb-1 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-gray-500">
                          <Stack size={12} /> {FLAG_LABEL[kind]} ({flagsByKind[kind].length})
                        </div>
                        <div className="flex flex-col gap-1">
                          {flagsByKind[kind].slice(0, 50).map((f, i) => (
                            <button
                              key={i}
                              onClick={() => goToFlag(f)}
                              title={f.bbox ? 'Zoom to this region on the canvas' : 'Go to this slice'}
                              className="group flex items-center justify-between gap-2 rounded-md border border-gray-200 bg-white px-3 py-1.5 text-left text-sm text-gray-700 hover:border-sky-300 hover:bg-sky-50"
                            >
                              <span className="truncate">{f.message}</span>
                              <ArrowRight size={14} className="flex-shrink-0 text-gray-300 group-hover:text-sky-500" />
                            </button>
                          ))}
                          {flagsByKind[kind].length > 50 && (
                            <span className="px-1 text-xs text-gray-400">…and {flagsByKind[kind].length - 50} more</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function formatArea(px: number): string {
  if (px >= 1e6) return `${(px / 1e6).toFixed(1)}M px²`;
  if (px >= 1e3) return `${(px / 1e3).toFixed(1)}k px²`;
  return `${px.toFixed(0)} px²`;
}
