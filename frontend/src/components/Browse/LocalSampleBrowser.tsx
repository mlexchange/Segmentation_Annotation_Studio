/**
 * LocalSampleBrowser — flat image-file list for a local folder.
 * Features: star ratings (persisted), annotation badge, star filter.
 */
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { File, PencilSimple, PencilLine } from '@phosphor-icons/react';
import StarRating from './StarRating';
import { useRatingStore, type StarRating as StarRatingValue } from '@/stores/ratingStore';
import { useAnnotatedSourceKeys } from '@/hooks/useAnnotatedSourceKeys';
import { buildSourceKey } from '@/lib/sourceKey';
import { ANNOTATION_FILTER_OPTIONS, type AnnotationFilter } from '@/types/annotationFilter';
import { API_BASE } from '@/config';

interface LocalSample {
  name: string;
  path: string;
}

interface LocalSampleBrowserProps {
  /** Granted absolute browse root. */
  root: string;
  /** Chosen subfolder relative to root. */
  rel: string;
  /** Called with the ABSOLUTE file path to open in Annotate. */
  onOpenInAnnotate: (absPath: string) => void;
  annotationFilter: AnnotationFilter;
  onAnnotationFilterChange: (filter: AnnotationFilter) => void;
}

/** Join an absolute root with a relative subpath into an absolute path. */
export function joinPath(root: string, rel: string): string {
  if (!rel) return root;
  return `${root.replace(/\/$/, '')}/${rel.replace(/^\//, '')}`;
}

export default function LocalSampleBrowser({
  root,
  rel,
  onOpenInAnnotate,
  annotationFilter,
  onAnnotationFilterChange,
}: LocalSampleBrowserProps) {
  const [minStars, setMinStars] = useState<0 | 1 | 2 | 3>(0);

  const ratings = useRatingStore((s) => s.ratings);
  const annotatedKeys = useAnnotatedSourceKeys();

  const { data, isLoading, error } = useQuery<{ items: LocalSample[]; total: number }>({
    queryKey: ['localSamples', root, rel],
    queryFn: async () => {
      const res = await fetch(
        `${API_BASE}/api/local/samples?root=${encodeURIComponent(root)}&rel=${encodeURIComponent(rel)}`,
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },
  });

  const allSamples = data?.items ?? [];

  const filtered = allSamples.filter((s) => {
    const sk = buildSourceKey('local', joinPath(root, s.path));
    const isAnnotated = annotatedKeys.has(sk);

    if (annotationFilter === 'annotated' && !isAnnotated) return false;
    if (annotationFilter === 'unannotated' && isAnnotated) return false;
    if (minStars > 0 && (ratings[sk] ?? 0) < minStars) return false;
    return true;
  });

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sky-300 text-sm">
        Loading samples…
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center text-red-400 text-sm">
        {String((error as Error).message)}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-200">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
        <File size={15} className="text-sky-500" />
        <span className="text-sm font-semibold text-slate-200">Local Samples</span>
        <span className="text-xs text-slate-500 font-mono truncate">{joinPath(root, rel) || 'root'}</span>
        <label className="flex items-center gap-2 text-xs text-slate-400 ml-auto">
          Annotation
          <select
            value={annotationFilter}
            onChange={(e) => onAnnotationFilterChange(e.target.value as AnnotationFilter)}
            className="text-xs rounded px-2 py-1 bg-slate-900 text-slate-200 border border-slate-600 focus:outline-none focus:ring-1 focus:ring-sky-500"
          >
            {ANNOTATION_FILTER_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full font-mono bg-slate-700 text-slate-400">
          {filtered.length !== allSamples.length ? `${filtered.length} / ${allSamples.length}` : allSamples.length}
        </span>
      </div>

      {/* Star filter */}
      <div className="flex items-center gap-1.5 px-4 py-1.5 border-b border-slate-700 bg-slate-900/60 shrink-0">
        <span className="text-[10px] text-slate-500 shrink-0">Min rating</span>
        <div className="flex items-center gap-1">
          {([0, 1, 2, 3] as const).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setMinStars(n)}
              className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                minStars === n
                  ? 'bg-amber-500 text-white font-medium'
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
              }`}
            >
              {n === 0 ? 'All' : '★'.repeat(n)}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {filtered.length === 0 && (
          <div className="flex items-center justify-center h-full text-slate-500 text-sm">
            {annotationFilter !== 'all' || minStars > 0
              ? 'No samples match the current filters.'
              : 'No image files found in this folder.'}
          </div>
        )}
        {filtered.map((s) => (
          <LocalRow
            key={s.path}
            sample={s}
            absPath={joinPath(root, s.path)}
            annotatedKeys={annotatedKeys}
            ratings={ratings}
            onOpenInAnnotate={onOpenInAnnotate}
          />
        ))}
      </div>
    </div>
  );
}

function LocalRow({
  sample,
  absPath,
  annotatedKeys,
  ratings,
  onOpenInAnnotate,
}: {
  sample: LocalSample;
  absPath: string;
  annotatedKeys: Set<string>;
  ratings: Record<string, StarRatingValue>;
  onOpenInAnnotate: (path: string) => void;
}) {
  const sourceKey = buildSourceKey('local', absPath);
  const setRating = useRatingStore((s) => s.setRating);
  const rating = (ratings[sourceKey] ?? 0) as StarRatingValue;

  const isAnnotated = annotatedKeys.has(sourceKey);

  return (
    <div className="flex items-center px-4 py-2 border-b border-slate-800 hover:bg-slate-800 group gap-2">
      {/* Stars */}
      <StarRating value={rating} onChange={(r) => setRating(sourceKey, r)} size={12} />

      {/* Name + badge */}
      <div className="flex-1 flex items-center gap-2 min-w-0">
        <span className="font-mono text-xs text-slate-300 truncate" title={sample.path}>
          {sample.name}
        </span>
        {isAnnotated && (
          <span className="shrink-0 flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] font-medium bg-sky-900/60 text-sky-300 border border-sky-700/40">
            <PencilLine size={9} />
            annotated
          </span>
        )}
      </div>

      {/* Annotate button */}
      <button
        type="button"
        onClick={() => onOpenInAnnotate(absPath)}
        className="shrink-0 flex items-center gap-1 px-2 py-1 rounded text-xs font-medium bg-sky-700 hover:bg-sky-600 text-white opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <PencilSimple size={11} />
        Annotate
      </button>
    </div>
  );
}
