import { useRef } from 'react';
import { X, CaretDown } from '@phosphor-icons/react';
import type { ColumnState, BrowseValue } from './hooks/useBrowseData';

interface BrowseColumnProps {
  colIndex: number;
  column: ColumnState;
  facets: string[];
  width: number;
  onFieldChange: (colIndex: number, field: string) => void;
  onSelect: (colIndex: number, value: string | null) => void;
  onRemove: (colIndex: number) => void;
  isLast: boolean;
}

export default function BrowseColumn({
  colIndex,
  column,
  facets,
  width,
  onFieldChange,
  onSelect,
  onRemove,
  isLast,
}: BrowseColumnProps) {
  const selectRef = useRef<HTMLSelectElement>(null);

  return (
    <div
      className="flex flex-col border-r"
      style={{
        minWidth: 120,
        width,
        maxWidth: 600,
        background: '#1e293b',
        borderColor: '#334155',
        flexShrink: 0,
      }}
    >
      {/* Column header: field picker + remove */}
      <div
        className="flex items-center gap-1 px-2 py-2 border-b"
        style={{ borderColor: '#334155', background: '#0f172a' }}
      >
        <div className="relative flex-1 min-w-0">
          <select
            ref={selectRef}
            value={column.field}
            onChange={e => onFieldChange(colIndex, e.target.value)}
            className="w-full appearance-none truncate pr-6 pl-2 py-1 text-xs rounded font-medium focus:outline-none focus:ring-1"
            style={{
              background: '#1e293b',
              color: '#e2e8f0',
              border: '1px solid #334155',
              lineHeight: '1.4',
            }}
          >
            {facets.map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
          <CaretDown
            size={10}
            className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2"
            style={{ color: '#64748b' }}
          />
        </div>
        <button
          onClick={() => onRemove(colIndex)}
          className="shrink-0 p-0.5 rounded hover:bg-slate-700 transition-colors"
          title="Remove column"
          style={{ color: '#64748b' }}
        >
          <X size={12} />
        </button>
      </div>

      {/* Value list */}
      <div className="flex-1 overflow-y-auto" style={{ minHeight: 0 }}>
        {column.loading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs" style={{ color: '#64748b' }}>Loading…</span>
          </div>
        )}
        {column.error && (
          <div className="px-3 py-3">
            <p className="text-xs" style={{ color: '#f87171' }}>{column.error}</p>
          </div>
        )}
        {!column.loading && !column.error && column.values.length === 0 && (
          <div className="px-3 py-3">
            <p className="text-xs" style={{ color: '#64748b' }}>No values</p>
          </div>
        )}
        {!column.loading &&
          column.values.map((item: BrowseValue) => {
            const isSelected = column.selected === item.value;
            return (
              <button
                key={item.value}
                onClick={() => onSelect(colIndex, isSelected ? null : item.value)}
                className="w-full flex items-center justify-between px-3 py-1.5 text-left transition-colors"
                style={{
                  background: isSelected ? '#1d4ed8' : 'transparent',
                  color: isSelected ? '#fff' : '#cbd5e1',
                  borderBottom: '1px solid #1e293b',
                }}
                onMouseEnter={e => {
                  if (!isSelected) (e.currentTarget as HTMLButtonElement).style.background = '#334155';
                }}
                onMouseLeave={e => {
                  if (!isSelected) (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                }}
              >
                <span className="text-xs truncate flex-1 mr-2" title={item.value}>
                  {item.value}
                </span>
                <span
                  className="shrink-0 text-xs px-1.5 py-0.5 rounded-full font-mono"
                  style={{
                    background: isSelected ? 'rgba(255,255,255,0.2)' : '#0f172a',
                    color: isSelected ? '#fff' : '#94a3b8',
                    fontSize: 10,
                  }}
                >
                  {item.count}
                </span>
              </button>
            );
          })}
      </div>
    </div>
  );
}
