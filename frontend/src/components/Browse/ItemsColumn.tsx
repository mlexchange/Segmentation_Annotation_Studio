import { useState } from 'react';
import { File, PencilSimple, PencilLine } from '@phosphor-icons/react';
import StarRating from './StarRating';
import { useRatingStore, type StarRating as StarRatingValue } from '@/stores/ratingStore';
import { useAnnotatedSourceKeys } from '@/hooks/useAnnotatedSourceKeys';
import { buildSourceKey } from '@/lib/sourceKey';
import type { AnnotationFilter } from '@/types/annotationFilter';
import type { BrowseItem } from './hooks/useBrowseData';

interface ItemsColumnProps {
  items: BrowseItem[];
  total: number;
  loading: boolean;
  selectedItem: BrowseItem | null;
  onSelect: (item: BrowseItem | null) => void;
  onOpenInAnnotate: (item: BrowseItem) => void;
  width: number;
  serverUri: string;
  annotationFilter: AnnotationFilter;
}

/** Last column of the browser: leaf records matching the current filter chain. */
export default function ItemsColumn({
  items,
  total,
  loading,
  selectedItem,
  onSelect,
  onOpenInAnnotate,
  width,
  serverUri,
  annotationFilter,
}: ItemsColumnProps) {
  const [minStars, setMinStars] = useState<0 | 1 | 2 | 3>(0);
  const ratings = useRatingStore((s) => s.ratings);
  const annotatedKeys = useAnnotatedSourceKeys();

  const filtered = items.filter((item) => {
    const sk = buildSourceKey('tiled', item.path, serverUri);
    const isAnnotated = annotatedKeys.has(sk);

    if (annotationFilter === 'annotated' && !isAnnotated) return false;
    if (annotationFilter === 'unannotated' && isAnnotated) return false;
    if (minStars > 0 && (ratings[sk] ?? 0) < minStars) return false;
    return true;
  });

  return (
    <div
      className="flex flex-col border-r border-slate-700 bg-slate-800 shrink-0"
      style={{ minWidth: 140, width, maxWidth: 600 }}
    >
      {/* Column header */}
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-900 flex items-center gap-2">
        <File size={13} className="text-sky-500" />
        <span className="text-xs font-semibold text-slate-400">Samples</span>
        {!loading && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full font-mono bg-slate-800 text-slate-500">
            {filtered.length !== total ? `${filtered.length} / ${total}` : total}
          </span>
        )}
      </div>

      {/* Star filter */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-slate-700 bg-slate-900/60">
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
        {loading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-500">Loading…</span>
          </div>
        )}
        {!loading && filtered.length === 0 && (
          <p className="text-xs text-slate-500 px-3 py-3">
            {annotationFilter !== 'all' || minStars > 0
              ? 'No samples match the current filters.'
              : 'No matching samples'}
          </p>
        )}
        {!loading &&
          filtered.map((item) => (
            <ItemRow
              key={item.path}
              item={item}
              isSelected={selectedItem?.path === item.path}
              onSelect={() => onSelect(selectedItem?.path === item.path ? null : item)}
              onOpenInAnnotate={() => onOpenInAnnotate(item)}
              serverUri={serverUri}
            />
          ))}
      </div>
    </div>
  );
}

interface ItemRowProps {
  item: BrowseItem;
  isSelected: boolean;
  onSelect: () => void;
  onOpenInAnnotate: () => void;
  serverUri: string;
}

function ItemRow({ item, isSelected, onSelect, onOpenInAnnotate, serverUri }: ItemRowProps) {
  const sourceKey = buildSourceKey('tiled', item.path, serverUri);
  const rating = useRatingStore((s) => s.ratings[sourceKey] ?? 0) as StarRatingValue;
  const setRating = useRatingStore((s) => s.setRating);
  const annotatedKeys = useAnnotatedSourceKeys();

  const isAnnotated = annotatedKeys.has(sourceKey);

  const background = isSelected ? 'bg-blue-700' : 'bg-transparent hover:bg-slate-700';

  return (
    <div className={`flex items-center transition-colors border-b border-slate-800 ${background}`}>
      <button
        type="button"
        onClick={onSelect}
        className="flex-1 flex flex-col px-2 py-1.5 text-left min-w-0"
      >
        <span className={`text-xs font-medium truncate ${isSelected ? 'text-white' : 'text-slate-200'}`}>
          {item.sample}
        </span>
        <div className="flex items-center gap-1.5 mt-0.5">
          <StarRating
            value={rating}
            onChange={(r) => setRating(sourceKey, r)}
            size={11}
          />
          {isAnnotated && (
            <span className="flex items-center gap-0.5 text-[9px] font-medium text-sky-300">
              <PencilLine size={9} />
              annotated
            </span>
          )}
        </div>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onOpenInAnnotate();
        }}
        title="Open in Annotate"
        className="shrink-0 px-2 py-2 self-center rounded hover:bg-slate-600 transition-colors"
        style={{ color: isSelected ? '#fff' : '#64748b' }}
      >
        <PencilSimple size={13} />
      </button>
    </div>
  );
}
