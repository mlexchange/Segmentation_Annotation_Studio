/**
 * SlicesColumn — drill-in column listing the individual array slices of a
 * multi-image dataset (volume/folder upload). Each slice is its own selectable,
 * individually-openable sample. Rendered to the right of the Samples column.
 */
import { Stack, PencilSimple, PencilLine, CaretLeft } from '@phosphor-icons/react';
import StarRating from './StarRating';
import { useRatingStore, type StarRating as StarRatingValue } from '@/stores/ratingStore';
import { useAnnotatedSourceKeys } from '@/hooks/useAnnotatedSourceKeys';
import { buildSourceKey } from '@/lib/sourceKey';
import type { BrowseItem } from './hooks/useBrowseData';

interface SlicesColumnProps {
  /** The dataset being drilled into (for the header label). */
  dataset: BrowseItem;
  slices: BrowseItem[];
  loading: boolean;
  selectedItem: BrowseItem | null;
  onSelect: (item: BrowseItem | null) => void;
  onOpenInAnnotate: (item: BrowseItem) => void;
  onClose: () => void;
  width: number;
  serverUri: string;
}

/** Shorten a slice key to its trailing number when it shares the dataset prefix. */
function sliceLabel(slice: BrowseItem, datasetSample: string): string {
  const imageNumber = slice.metadata?.image_number;
  if (imageNumber != null && String(imageNumber).trim() !== '') return `Slice ${imageNumber}`;
  if (slice.sample.startsWith(datasetSample)) {
    const tail = slice.sample.slice(datasetSample.length).replace(/^[_-]+/, '');
    if (tail) return `Slice ${tail}`;
  }
  return slice.sample;
}

/** Drill-in column of individual slices for the selected multi-image dataset. */
export default function SlicesColumn({
  dataset,
  slices,
  loading,
  selectedItem,
  onSelect,
  onOpenInAnnotate,
  onClose,
  width,
  serverUri,
}: SlicesColumnProps) {
  return (
    <div
      className="flex flex-col border-r border-slate-700 bg-slate-800 shrink-0"
      style={{ minWidth: 160, width, maxWidth: 600 }}
    >
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-900 flex items-center gap-2">
        <button
          type="button"
          onClick={onClose}
          title="Back to samples"
          className="shrink-0 p-0.5 rounded hover:bg-slate-700 transition-colors text-slate-400"
        >
          <CaretLeft size={13} />
        </button>
        <Stack size={13} className="text-sky-500" />
        <span className="text-xs font-semibold text-slate-400 truncate" title={dataset.sample}>
          Slices
        </span>
        {!loading && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full font-mono bg-slate-800 text-slate-500">
            {slices.length}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-500">Loading slices…</span>
          </div>
        )}
        {!loading && slices.length === 0 && (
          <p className="text-xs text-slate-500 px-3 py-3">No slices found.</p>
        )}
        {!loading &&
          slices.map((slice) => (
            <SliceRow
              key={slice.path}
              slice={slice}
              label={sliceLabel(slice, dataset.sample)}
              isSelected={selectedItem?.path === slice.path}
              onSelect={() => onSelect(selectedItem?.path === slice.path ? null : slice)}
              onOpenInAnnotate={() => onOpenInAnnotate(slice)}
              serverUri={serverUri}
            />
          ))}
      </div>
    </div>
  );
}

interface SliceRowProps {
  slice: BrowseItem;
  label: string;
  isSelected: boolean;
  onSelect: () => void;
  onOpenInAnnotate: () => void;
  serverUri: string;
}

/** Single slice row: label, star rating, annotated badge, and open-in-Annotate. */
function SliceRow({ slice, label, isSelected, onSelect, onOpenInAnnotate, serverUri }: SliceRowProps) {
  const sourceKey = buildSourceKey('tiled', slice.path, serverUri);
  const rating = useRatingStore((s) => s.ratings[sourceKey] ?? 0) as StarRatingValue;
  const setRating = useRatingStore((s) => s.setRating);
  const annotatedKeys = useAnnotatedSourceKeys();
  const isAnnotated =
    annotatedKeys.has(sourceKey) ||
    slice.metadata?.studio_annotated === 'yes' ||
    slice.metadata?.studio_annotated === true;

  const background = isSelected ? 'bg-blue-700' : 'bg-transparent hover:bg-slate-700';

  return (
    <div className={`flex items-center transition-colors border-b border-slate-800 ${background}`}>
      {/* role="button" div (not <button>) because it wraps StarRating's buttons. */}
      <div
        role="button"
        tabIndex={0}
        onClick={onSelect}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(); } }}
        className="flex-1 flex flex-col px-2 py-1.5 text-left min-w-0 cursor-pointer"
      >
        <span className={`text-xs font-medium truncate ${isSelected ? 'text-white' : 'text-slate-200'}`}>
          {label}
        </span>
        <div className="flex items-center gap-1.5 mt-0.5">
          <StarRating value={rating} onChange={(r) => setRating(sourceKey, r)} size={11} />
          {isAnnotated && (
            <span className="flex items-center gap-0.5 text-[9px] font-medium text-sky-300">
              <PencilLine size={9} />
              annotated
            </span>
          )}
        </div>
      </div>
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onOpenInAnnotate(); }}
        title="Open this slice in Annotate"
        className="shrink-0 px-2 py-2 self-center rounded hover:bg-slate-600 transition-colors"
        style={{ color: isSelected ? '#fff' : '#64748b' }}
      >
        <PencilSimple size={13} />
      </button>
    </div>
  );
}
