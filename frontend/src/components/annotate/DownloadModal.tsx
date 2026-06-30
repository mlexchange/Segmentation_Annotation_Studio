/**
 * DownloadModal — scope picker + optional star-rating filter, then COCO export.
 */
import { useMemo, useState } from 'react';
import { DownloadSimple, X, CheckCircle, WarningCircle } from '@phosphor-icons/react';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';
import { useRatingStore } from '@/stores/ratingStore';
import { buildSourceKey } from '@/lib/sourceKey';
import { useExportJob } from '@/hooks/useExportJob';

interface DownloadModalProps {
  onClose: () => void;
}

type Scope = 'current' | 'all' | 'stars1' | 'stars2' | 'stars3';

/** Parse a canonical sourceKey back to { kind, source, serverUri }. */
function parseSourceKey(sk: string) {
  if (sk.startsWith('tiled:')) {
    const rest = sk.slice('tiled:'.length);
    const sep = rest.indexOf(':');
    return { kind: 'tiled' as const, serverUri: rest.slice(0, sep) || null, source: rest.slice(sep + 1) };
  }
  return { kind: 'local' as const, serverUri: null, source: sk.slice('local:'.length) };
}

const SCOPE_OPTIONS: { value: Scope; label: string; desc: string; stars?: string }[] = [
  {
    value: 'current',
    label: 'Current sample only',
    desc: 'Export annotations for the sample currently open.',
  },
  {
    value: 'all',
    label: 'All annotated samples',
    desc: 'Every sample with at least one annotation in this session.',
  },
  {
    value: 'stars1',
    label: '★ and above',
    desc: 'Annotated samples rated 1 star or higher.',
    stars: '★☆☆',
  },
  {
    value: 'stars2',
    label: '★★ and above',
    desc: 'Annotated samples rated 2 stars or higher.',
    stars: '★★☆',
  },
  {
    value: 'stars3',
    label: '★★★ only',
    desc: 'Only the best — annotated samples rated 3 stars.',
    stars: '★★★',
  },
];

/** Renders the COCO download dialog and drives the export job for the chosen scope. */
export default function DownloadModal({ onClose }: DownloadModalProps) {
  const { source, kind, serverUri } = useDatasetStore();
  const { byImage, splitBySlice, negativeSlices } = useAnnotationStore();
  const { classes } = useClassStore();
  const ratings = useRatingStore((s) => s.ratings);

  const [scope, setScope] = useState<Scope>('current');
  const [includePolygons, setIncludePolygons] = useState(false);
  const { state: job, start, downloadUrl } = useExportJob();
  const status = job.status;

  /** Preview count of samples that would be exported for the selected scope. */
  const previewCount = useMemo(() => {
    if (scope === 'current') return source ? 1 : 0;
    const minStars = scope === 'stars1' ? 1 : scope === 'stars2' ? 2 : scope === 'stars3' ? 3 : 0;
    return Object.keys(byImage).filter((sk) => {
      const hasShapes = Object.values(byImage[sk]).some((s) => s.length > 0);
      if (!hasShapes) return false;
      if (minStars > 0) return (ratings[sk] ?? 0) >= minStars;
      return true;
    }).length;
  }, [scope, source, byImage, ratings]);

  /** Builds the per-sample export source list for the selected scope; throws if no current sample. */
  const buildSources = () => {
    if (scope === 'current') {
      if (!source || !kind) throw new Error('No active sample loaded.');
      const sk = buildSourceKey(kind as 'tiled' | 'local', source, serverUri);
      return [{
        kind,
        source,
        server_uri: serverUri ?? null,
        slices: byImage[sk] ?? {},
        split_by_slice: splitBySlice[sk] ?? {},
        negative_slices: negativeSlices[sk] ?? [],
      }];
    }

    const minStars = scope === 'stars1' ? 1 : scope === 'stars2' ? 2 : scope === 'stars3' ? 3 : 0;

    return Object.keys(byImage)
      .filter((sk) => {
        const hasShapes = Object.values(byImage[sk]).some((s) => s.length > 0);
        if (!hasShapes) return false;
        if (minStars > 0) return (ratings[sk] ?? 0) >= minStars;
        return true;
      })
      .map((sk) => {
        const { kind: k, source: src, serverUri: srv } = parseSourceKey(sk);
        return {
          kind: k,
          source: src,
          server_uri: srv,
          slices: byImage[sk],
          split_by_slice: splitBySlice[sk] ?? {},
          negative_slices: negativeSlices[sk] ?? [],
        };
      });
  };

  /** Builds the sources (alerting on error/empty) and starts the COCO export job. */
  const handleExport = () => {
    let sources;
    try {
      sources = buildSources();
    } catch (e) {
      // buildSources throws only for an empty/invalid scope; surface inline.
      window.alert(String(e));
      return;
    }
    if (sources.length === 0) { window.alert('No samples match the selected scope.'); return; }
    start({ sources, classes, mode: 'merge', include_polygons: includePolygons });
  };

  const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md space-y-4 p-6">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-white">
            <DownloadSimple size={20} />
            <span className="text-base font-semibold">Download COCO Dataset</span>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Scope options */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Export scope</p>
          <div className="flex flex-col gap-1.5">
            {SCOPE_OPTIONS.map(({ value, label, desc, stars }) => (
              <label
                key={value}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border cursor-pointer transition-colors ${
                  scope === value
                    ? 'border-sky-500 bg-sky-900/30'
                    : 'border-slate-600 hover:border-slate-500'
                }`}
              >
                <input
                  type="radio"
                  name="scope"
                  value={value}
                  checked={scope === value}
                  onChange={() => setScope(value)}
                  className="shrink-0 accent-sky-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-slate-200">{label}</p>
                    {stars && (
                      <span className="text-amber-400 text-sm leading-none">{stars}</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Options */}
        {status === 'idle' && (
          <label className="flex items-start gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={includePolygons}
              onChange={(e) => setIncludePolygons(e.target.checked)}
              className="mt-0.5 accent-sky-500"
            />
            <span>
              Include polygon copy in COCO <span className="text-slate-500">(slower; RLE masks are always exact — only needed for some external viewers)</span>
            </span>
          </label>
        )}

        {/* Preview count */}
        {status === 'idle' && (
          <p className="text-xs text-slate-400 text-right">
            {previewCount === 0
              ? 'No samples match.'
              : `${previewCount} sample${previewCount !== 1 ? 's' : ''} will be exported.`}
          </p>
        )}

        {/* Live progress: phase + bar + scrolling backend log */}
        {(status === 'running' || status === 'done') && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="capitalize">{job.phase || 'working'}…</span>
              {job.total > 0 && <span className="tabular-nums">{job.done}/{job.total} slices</span>}
            </div>
            <div className="h-1.5 w-full rounded bg-slate-700 overflow-hidden">
              <div
                className={`h-full transition-all ${status === 'done' ? 'bg-green-500' : 'bg-sky-500'}`}
                style={{ width: status === 'done' ? '100%' : `${Math.max(5, pct)}%` }}
              />
            </div>
            {job.log.length > 0 && (
              <div className="max-h-28 overflow-y-auto rounded bg-slate-900/70 border border-slate-700 p-2 text-[11px] font-mono text-slate-400 leading-relaxed">
                {job.log.slice(-12).map((line, i) => <div key={i}>{line}</div>)}
              </div>
            )}
          </div>
        )}

        {status === 'done' && (
          <div className="flex items-start gap-2 text-sm text-green-300">
            <CheckCircle size={16} className="mt-0.5 shrink-0" />
            <span>
              Saved to <span className="font-mono">{String(job.result?.dataset_path ?? 'server')}</span> (Tiled).
              {' '}Use <b>Download .zip</b> to save images + masks to your computer.
            </span>
          </div>
        )}
        {status === 'error' && (
          <div className="flex items-start gap-2 text-sm text-red-400">
            <WarningCircle size={16} className="mt-0.5 shrink-0" />
            <span>{job.error ?? 'Export failed.'}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm rounded-md border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors"
          >
            {status === 'done' ? 'Close' : 'Cancel'}
          </button>
          {status === 'done' && downloadUrl ? (
            <a
              href={downloadUrl}
              download
              className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-500 transition-colors flex items-center gap-2"
            >
              <DownloadSimple size={15} />
              Download .zip
            </a>
          ) : status !== 'done' && (
            <button
              type="button"
              onClick={handleExport}
              disabled={status === 'running' || previewCount === 0}
              className="px-4 py-2 text-sm rounded-md bg-sky-600 text-white hover:bg-sky-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <DownloadSimple size={15} />
              {status === 'running' ? 'Exporting…' : 'Export'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
