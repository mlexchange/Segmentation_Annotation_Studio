/**
 * AnnotationsColumn — drill-in column listing the annotations on one selected
 * slice, grouped by class. Rendered to the right of the Slices column, so the
 * Browse drill-down reads sample → slices → annotations.
 *
 * Read-only by design: rows describe what's on the slice, and the footer button
 * hands off to Annotate for actual editing.
 */
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { CaretLeft, PencilSimple, Polygon, Square, Circle, PaintBrush, Tag } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import { buildSourceKey } from '@/lib/sourceKey';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useSourceTaxonomyStore } from '@/stores/sourceTaxonomyStore';
import {
  summarizeSliceAnnotations,
  type AnnotationPayload,
  type PayloadSource,
  type ShapeSummary,
} from '@/lib/sliceAnnotationSummary';
import { sliceLabel } from './SlicesColumn';
import type { BrowseItem } from './hooks/useBrowseData';

interface AnnotationsColumnProps {
  /** The drilled-into dataset, whose source key holds per-slice annotations. */
  dataset: BrowseItem;
  slice: BrowseItem;
  /** Position of `slice` within the dataset — its slice key under the volume's source key. */
  sliceIndex: number;
  onOpenInAnnotate: (item: BrowseItem) => void;
  onClose: () => void;
  width: number;
  serverUri: string;
}

const SHAPE_ICONS = {
  polygon: Polygon,
  rectangle: Square,
  ellipse: Circle,
  brush: PaintBrush,
} as const;

const SHAPE_LABELS = {
  polygon: 'Polygon',
  rectangle: 'Rectangle',
  ellipse: 'Ellipse',
  brush: 'Brush',
} as const;

/**
 * Latest saved annotations for one source key: its draft, or — when no draft
 * exists — the newest saved version's payload. Null when the key has neither.
 */
async function fetchPayloadForKey(sourceKey: string): Promise<AnnotationPayload | null> {
  const qs = `source_key=${encodeURIComponent(sourceKey)}`;

  const draftRes = await fetch(`${API_BASE}/api/annotations/draft?${qs}`);
  if (draftRes.ok) {
    const doc = await draftRes.json();
    if (doc?.payload) return doc.payload as AnnotationPayload;
  }

  // No draft (404 for a sample only ever saved as a version, or drafts cleared).
  const versionsRes = await fetch(`${API_BASE}/api/annotations/versions?${qs}`);
  if (!versionsRes.ok) return null;
  const versions: { version: number }[] = await versionsRes.json();
  if (!Array.isArray(versions) || versions.length === 0) return null;

  const newest = versions.reduce((a, b) => (b.version > a.version ? b : a));
  const payloadRes = await fetch(`${API_BASE}/api/annotations/versions/${newest.version}?${qs}`);
  if (!payloadRes.ok) return null;
  const doc = await payloadRes.json();
  return (doc?.payload as AnnotationPayload) ?? null;
}

/**
 * Merge persisted annotations (draft / newest version) with this session's
 * unsaved store state for the slice, under both source keys it can live at.
 */
function useSliceAnnotations(dataset: BrowseItem, slice: BrowseItem, sliceIndex: number, serverUri: string) {
  const volumeKey = buildSourceKey('tiled', dataset.path, serverUri);
  const standaloneKey = buildSourceKey('tiled', slice.path, serverUri);
  const volumeSliceKey = String(sliceIndex);

  // Two independently-keyed queries, not one keyed on both: `standaloneKey` changes on
  // every slice click (it's the individual slice's own path) while `volumeKey` stays
  // fixed for the whole drilled-into dataset. Combining them into one key meant every
  // slice click missed cache and refetched the volume's own (unchanged) payload too —
  // this way that fetch stays cached across slice clicks within the same dataset.
  const volumeQuery = useQuery({
    queryKey: ['annotationPayload', volumeKey],
    queryFn: () => fetchPayloadForKey(volumeKey),
    staleTime: 10_000,
  });
  const standaloneQuery = useQuery({
    queryKey: ['annotationPayload', standaloneKey],
    queryFn: () => fetchPayloadForKey(standaloneKey),
    staleTime: 10_000,
    enabled: volumeKey !== standaloneKey,
  });

  const byImage = useAnnotationStore((s) => s.byImage);
  const negativeSlices = useAnnotationStore((s) => s.negativeSlices);
  const splitBySlice = useAnnotationStore((s) => s.splitBySlice);
  const classesBySource = useSourceTaxonomyStore((s) => s.classesBySource);

  /** This session's in-store state for one key, shaped like a fetched payload. */
  const liveFor = (key: string): AnnotationPayload | null => {
    if (!byImage[key] && !negativeSlices[key] && !splitBySlice[key]) return null;
    return {
      classes: classesBySource[key] ?? [],
      slices: byImage[key] ?? {},
      split_by_slice: splitBySlice[key] ?? {},
      negative_slices: negativeSlices[key] ?? [],
    };
  };

  // Live store state first so unsaved edits win on shape identity and class colors.
  const sources: PayloadSource[] = [
    { payload: liveFor(volumeKey), sliceKey: volumeSliceKey },
    { payload: volumeQuery.data ?? null, sliceKey: volumeSliceKey },
  ];
  if (volumeKey !== standaloneKey) {
    sources.push({ payload: liveFor(standaloneKey), sliceKey: '0' });
    sources.push({ payload: standaloneQuery.data ?? null, sliceKey: '0' });
  }

  const summary = useMemo(
    () => summarizeSliceAnnotations(sources),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      volumeQuery.data, standaloneQuery.data, byImage, negativeSlices, splitBySlice,
      classesBySource, volumeKey, standaloneKey, volumeSliceKey,
    ],
  );

  return {
    summary,
    isLoading: volumeQuery.isLoading || standaloneQuery.isLoading,
    isError: volumeQuery.isError || standaloneQuery.isError,
  };
}

/** Drill-in column of the selected slice's annotations, grouped by class. */
export default function AnnotationsColumn({
  dataset,
  slice,
  sliceIndex,
  onOpenInAnnotate,
  onClose,
  width,
  serverUri,
}: AnnotationsColumnProps) {
  const { summary, isLoading, isError } = useSliceAnnotations(dataset, slice, sliceIndex, serverUri);
  const label = sliceLabel(slice, dataset.sample);

  return (
    <div
      className="flex flex-col border-r border-slate-700 bg-slate-800 shrink-0"
      style={{ minWidth: 180, width, maxWidth: 600 }}
    >
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-900 flex items-center gap-2">
        <button
          type="button"
          onClick={onClose}
          title="Back to slices"
          className="shrink-0 p-0.5 rounded hover:bg-slate-700 transition-colors text-slate-400"
        >
          <CaretLeft size={13} />
        </button>
        <Tag size={13} className="text-sky-500" />
        {/* "Annotations" is a short, never-truncated label — the slice's own
            (possibly-long, truncated) name has its title on the line below instead. */}
        <span className="text-xs font-semibold text-slate-400 truncate">
          Annotations
        </span>
        {!isLoading && !isError && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full font-mono bg-slate-800 text-slate-400">
            {summary.totalShapes}
          </span>
        )}
      </div>

      <p className="px-3 pt-2 pb-1 text-[10px] text-slate-400 leading-snug truncate" title={label}>
        {label}
      </p>

      {(summary.isNegative || summary.split) && (
        <div className="flex flex-wrap gap-1 px-3 pb-2">
          {summary.isNegative && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-300 font-medium">
              negative example
            </span>
          )}
          {summary.split && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300 font-medium">
              {summary.split}
            </span>
          )}
        </div>
      )}

      <div className="flex-1 overflow-y-auto min-h-0 border-t border-slate-800">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-400">Loading annotations…</span>
          </div>
        )}
        {isError && !isLoading && (
          <p className="text-xs text-red-400 px-3 py-3">Could not load annotations.</p>
        )}
        {!isLoading && !isError && summary.classes.length === 0 && (
          <p className="text-xs text-slate-400 px-3 py-3">
            No annotations on this image yet. Open it in Annotate to add some.
          </p>
        )}
        {!isLoading &&
          !isError &&
          summary.classes.map((cls) => (
            <div key={cls.classId} className="border-b border-slate-800">
              <div className="flex items-center gap-1.5 px-2 py-1.5 bg-slate-800/60">
                <span
                  className="shrink-0 w-2.5 h-2.5 rounded-sm border border-slate-600"
                  style={{ backgroundColor: cls.color }}
                />
                <span className="text-xs font-medium text-slate-200 truncate" title={cls.label}>
                  {cls.label}
                </span>
                <span className="ml-auto text-[10px] font-mono text-slate-400">{cls.shapes.length}</span>
              </div>
              {cls.shapes.map((shape, i) => (
                <ShapeRow key={shape.id} shape={shape} index={i + 1} />
              ))}
            </div>
          ))}
      </div>

      <button
        type="button"
        onClick={() => onOpenInAnnotate(slice)}
        className="flex items-center justify-center gap-1.5 m-2 px-2 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-[11px] font-medium transition-colors"
      >
        <PencilSimple size={13} />
        Open in Annotate
      </button>
    </div>
  );
}

/** One annotation row: shape kind icon + numbered kind label. */
function ShapeRow({ shape, index }: { shape: ShapeSummary; index: number }) {
  // SHAPE_ICONS/SHAPE_LABELS cover every Shape['kind'] variant, so both lookups
  // are exhaustive — no fallback can ever actually be reached here.
  const Icon = SHAPE_ICONS[shape.kind];
  return (
    <div className="flex items-center gap-1.5 pl-5 pr-2 py-1 text-slate-300">
      <Icon size={11} className="shrink-0 text-slate-500" />
      <span className="text-[11px] truncate">
        {SHAPE_LABELS[shape.kind]} {index}
      </span>
    </div>
  );
}
