import { CheckSquare, File, Square } from '@phosphor-icons/react';
import type { BrowseItem } from './hooks/useBrowseData';

interface ItemsColumnProps {
  items: BrowseItem[];
  total: number;
  loading: boolean;
  selectedItem: BrowseItem | null;
  onSelect: (item: BrowseItem | null) => void;
  width: number;
  checkedPaths: Set<string>;
  onToggleCheck: (path: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
}

/** Last column of the browser: leaf records matching the current filter chain. */
export default function ItemsColumn({
  items,
  total,
  loading,
  selectedItem,
  onSelect,
  width,
  checkedPaths,
  onToggleCheck,
  onSelectAll,
  onClearAll,
}: ItemsColumnProps) {
  const allChecked = items.length > 0 && items.every((i) => checkedPaths.has(i.path));
  const someChecked = checkedPaths.size > 0 && !allChecked;

  return (
    <div
      className="flex flex-col border-r border-slate-700 bg-slate-800 shrink-0"
      style={{ minWidth: 120, width, maxWidth: 600 }}
    >
      <div className="px-3 py-2 border-b border-slate-700 bg-slate-900 flex items-center gap-2">
        <button
          type="button"
          onClick={allChecked ? onClearAll : onSelectAll}
          className="shrink-0 p-0.5 rounded hover:bg-slate-700 transition-colors"
          title={allChecked ? 'Deselect all' : 'Select all'}
          style={{ color: someChecked || allChecked ? '#3b82f6' : '#475569' }}
        >
          {allChecked ? (
            <CheckSquare size={14} weight="fill" />
          ) : someChecked ? (
            <CheckSquare size={14} />
          ) : (
            <Square size={14} />
          )}
        </button>
        <File size={13} className="text-sky-500" />
        <span className="text-xs font-semibold text-slate-400">Samples</span>
        {!loading && (
          <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded-full font-mono bg-slate-800 text-slate-500">
            {checkedPaths.size > 0 ? `${checkedPaths.size}/${total}` : total}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-slate-500">Loading…</span>
          </div>
        )}
        {!loading && items.length === 0 && (
          <p className="text-xs text-slate-500 px-3 py-3">No matching samples</p>
        )}
        {!loading &&
          items.map((item) => (
            <ItemRow
              key={item.path}
              item={item}
              isSelected={selectedItem?.path === item.path}
              isChecked={checkedPaths.has(item.path)}
              onSelect={() => onSelect(selectedItem?.path === item.path ? null : item)}
              onToggleCheck={() => onToggleCheck(item.path)}
            />
          ))}
      </div>
    </div>
  );
}

interface ItemRowProps {
  item: BrowseItem;
  isSelected: boolean;
  isChecked: boolean;
  onSelect: () => void;
  onToggleCheck: () => void;
}

function ItemRow({ item, isSelected, isChecked, onSelect, onToggleCheck }: ItemRowProps) {
  const background = isSelected
    ? 'bg-blue-700'
    : isChecked
      ? 'bg-blue-950'
      : 'bg-transparent hover:bg-slate-700';

  return (
    <div className={`flex items-start transition-colors border-b border-slate-800 ${background}`}>
      <button
        type="button"
        onClick={onToggleCheck}
        className="shrink-0 pl-2 pr-1 py-2 self-center"
        title={isChecked ? 'Deselect' : 'Select for staging'}
        style={{ color: isChecked ? '#3b82f6' : isSelected ? 'rgba(255,255,255,0.5)' : '#475569' }}
      >
        {isChecked ? <CheckSquare size={13} weight="fill" /> : <Square size={13} />}
      </button>
      <button
        type="button"
        onClick={onSelect}
        className="flex-1 flex flex-col px-2 py-1.5 text-left min-w-0"
      >
        <span className={`text-xs font-medium truncate ${isSelected ? 'text-white' : 'text-slate-200'}`}>
          {item.sample}
        </span>
        {item.metadata.angle_id != null && (
          <span className={`text-xs truncate ${isSelected ? 'text-white/70' : 'text-slate-500'}`}>
            {String(item.metadata.angle_id)}
          </span>
        )}
      </button>
    </div>
  );
}
