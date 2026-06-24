import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowsClockwise, PencilSimple, Plus, Stack } from '@phosphor-icons/react';
import BrowseColumn from './BrowseColumn';
import BrowseDetailPanel from './BrowseDetailPanel';
import ItemsColumn from './ItemsColumn';
import ResizeDivider from './ResizeDivider';
import { useBrowseData, type BrowseItem } from './hooks/useBrowseData';
import { useOpenInAnnotate } from '@/hooks/useOpenInAnnotate';
import type { ServerInfo } from '@/types/server';
import { ANNOTATION_FILTER_OPTIONS, type AnnotationFilter } from '@/types/annotationFilter';

interface ColumnBrowserProps {
  serverUri: string;
  containerPath?: string | null;
  servers: ServerInfo[];
  selectedServerUri: string;
  onServerChange: (uri: string) => void;
  annotationFilter: AnnotationFilter;
  onAnnotationFilterChange: (filter: AnnotationFilter) => void;
}

const DEFAULT_COLUMN_WIDTH = 220;
const DEFAULT_ITEMS_WIDTH = 260;
const DEFAULT_DETAIL_WIDTH = 340;
const MIN_DETAIL_WIDTH = 280;
const MAX_DETAIL_WIDTH = 900;
const INITIAL_COLUMN_COUNT = 4;

/** Studio facets are useful as filters but should not fill the initial column set. */
const STUDIO_FACETS = new Set(['Annotated', 'Annotated at', 'Shape count', 'Class count']);

function columnWidthForField(field: string): number {
  if (field === 'sample_name') return 200;
  if (field === 'Annotated') return 160;
  if (field === 'Annotated at') return 280;
  if (field === 'Shape count' || field === 'Class count') return 160;
  // ~7px per character at text-xs + padding for the field picker
  return Math.min(360, Math.max(DEFAULT_COLUMN_WIDTH, field.length * 7 + 48));
}

function formatColumnValue(field: string, value: string): string {
  if (field === 'Annotated at') {
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      });
    }
  }
  return value;
}

export default function ColumnBrowser({
  serverUri,
  containerPath,
  servers,
  selectedServerUri,
  onServerChange,
  annotationFilter,
  onAnnotationFilterChange,
}: ColumnBrowserProps) {
  const { state, actions } = useBrowseData(serverUri, 'All', undefined, containerPath);
  const { openTiledArray } = useOpenInAnnotate();
  const scrollRef = useRef<HTMLDivElement>(null);

  const [columnWidths, setColumnWidths] = useState<number[]>([]);
  const [itemsColumnWidth, setItemsColumnWidth] = useState(DEFAULT_ITEMS_WIDTH);
  const [detailWidth, setDetailWidth] = useState(DEFAULT_DETAIL_WIDTH);
  const [openStatus, setOpenStatus] = useState<string | null>(null);

  // Keep columnWidths in sync with the number of columns.
  useEffect(() => {
    setColumnWidths((prev) => {
      const n = state.columns.length;
      if (prev.length === n) return prev;
      if (prev.length > n) return prev.slice(0, n);
      const additions = Array.from({ length: n - prev.length }, (_, i) => {
        const field = state.columns[prev.length + i]?.field ?? '';
        return columnWidthForField(field);
      });
      return [...prev, ...additions];
    });
  }, [state.columns]);

  const prevColumnCount = useRef(0);
  const skipNextAutoScroll = useRef(false);

  // Scroll to reveal newly added columns (not on the initial bulk load).
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    if (skipNextAutoScroll.current) {
      skipNextAutoScroll.current = false;
      el.scrollLeft = 0;
      prevColumnCount.current = state.columns.length;
      return;
    }

    if (state.columns.length > prevColumnCount.current) {
      el.scrollLeft = Math.max(0, el.scrollWidth - el.clientWidth);
    }
    prevColumnCount.current = state.columns.length;
  }, [state.columns.length]);

  // First-load: populate with the first few discovered facets.
  const initialised = useRef(false);
  useEffect(() => {
    if (initialised.current || state.facets.length === 0) return;
    initialised.current = true;
    skipNextAutoScroll.current = true;
    const preferred = state.facets.filter((f) => !STUDIO_FACETS.has(f));
    const pool = preferred.length >= INITIAL_COLUMN_COUNT ? preferred : state.facets;
    const n = Math.min(INITIAL_COLUMN_COUNT, pool.length);
    pool.slice(0, n).forEach((f) => actions.addColumn(f));
  }, [state.facets, actions]);

  const handleResizeColumn = useCallback((index: number, newWidth: number) => {
    setColumnWidths((prev) => {
      if (index < 0 || index >= prev.length) return prev;
      const next = [...prev];
      next[index] = newWidth;
      return next;
    });
  }, []);

  const handleAddColumn = useCallback(() => {
    const used = new Set(state.columns.map((c) => c.field));
    const next = state.facets.find((f) => !used.has(f));
    if (next) actions.addColumn(next);
  }, [state.columns, state.facets, actions]);

  const handleOpenInAnnotate = useCallback(
    async (item: BrowseItem) => {
      setOpenStatus(`Opening ${item.sample}…`);
      try {
        await openTiledArray(item.path, serverUri);
      } catch (err) {
        setOpenStatus(`Failed to open: ${err}`);
      }
    },
    [openTiledArray, serverUri],
  );

  const activeFilters = useMemo(() => {
    const out: Record<string, string> = {};
    state.columns.forEach((col) => {
      if (col.selected !== null) out[col.field] = col.selected;
    });
    return out;
  }, [state.columns]);

  const activeFilterCount = Object.keys(activeFilters).length;
  const lastColumn = state.columns[state.columns.length - 1];
  const showItems = state.showingAll || (state.columns.length > 0 && lastColumn?.selected !== null);

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-200">
      <Toolbar
        facetsLoading={state.facetsLoading}
        facetCount={state.facets.length}
        activeFilterCount={activeFilterCount}
        showingAll={state.showingAll}
        selectedItem={state.selectedItem}
        onOpenInAnnotate={handleOpenInAnnotate}
        onRefresh={actions.refresh}
        onAddColumn={handleAddColumn}
        onShowAll={actions.showAll}
        servers={servers}
        selectedServerUri={selectedServerUri}
        onServerChange={onServerChange}
        annotationFilter={annotationFilter}
        onAnnotationFilterChange={onAnnotationFilterChange}
      />

      {openStatus && (
        <div className="shrink-0 px-4 py-1.5 text-xs font-medium border-b border-slate-700 bg-slate-800 text-sky-200">
          {openStatus}
        </div>
      )}

      {state.connectionStatus === 'disconnected' && (
        <div className="shrink-0 px-4 py-2 text-xs border-b border-red-900 bg-red-950/80 text-red-300">
          Cannot reach the API server. Make sure the backend (port 8002) and Tiled server are running.
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div
          ref={scrollRef}
          className="relative z-0 flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
        >
          {state.columns.length === 0 && !state.showingAll && !state.facetsLoading && (
            <div className="flex items-center justify-center flex-1">
              <p className="text-sm text-slate-500 text-center px-6">
                {state.connectionStatus === 'disconnected'
                  ? 'Connect to a Tiled server to browse.'
                  : state.facets.length === 0
                    ? 'No metadata fields to filter by. Click "All samples" to view every dataset.'
                    : 'Click "Add column" to filter, or "All samples" to view everything.'}
              </p>
            </div>
          )}

          {state.columns.map((col, i) => (
            <React.Fragment key={i}>
              <BrowseColumn
                colIndex={i}
                column={col}
                facets={state.facets}
                width={columnWidths[i] ?? DEFAULT_COLUMN_WIDTH}
                formatValue={formatColumnValue}
                onFieldChange={actions.changeColumnField}
                onSelect={actions.selectValue}
                onRemove={actions.removeColumn}
                isLast={i === state.columns.length - 1}
              />
              <ResizeDivider
                currentWidth={columnWidths[i] ?? DEFAULT_COLUMN_WIDTH}
                onResize={(w) => handleResizeColumn(i, w)}
              />
            </React.Fragment>
          ))}

          {showItems && (
            <>
              <ResizeDivider
                key="resize-items"
                currentWidth={itemsColumnWidth}
                onResize={setItemsColumnWidth}
                resizeRight
              />
              <ItemsColumn
                items={state.items}
                total={state.itemsTotal}
                loading={state.itemsLoading}
                selectedItem={state.selectedItem}
                onSelect={actions.selectItem}
                onOpenInAnnotate={handleOpenInAnnotate}
                width={itemsColumnWidth}
                serverUri={serverUri}
                annotationFilter={annotationFilter}
              />
            </>
          )}
        </div>

        {state.selectedItem && (
          <>
            <ResizeDivider
              key="resize-detail"
              currentWidth={detailWidth}
              onResize={setDetailWidth}
              resizeRight
              minWidth={MIN_DETAIL_WIDTH}
              maxWidth={MAX_DETAIL_WIDTH}
            />
            <DetailPanelSlot
              item={state.selectedItem}
              width={detailWidth}
              onClose={() => actions.selectItem(null)}
              serverUri={serverUri}
              onOpenInAnnotate={handleOpenInAnnotate}
            />
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

interface ToolbarProps {
  facetsLoading: boolean;
  facetCount: number;
  activeFilterCount: number;
  showingAll: boolean;
  selectedItem: BrowseItem | null;
  onOpenInAnnotate: (item: BrowseItem) => void;
  onRefresh: () => void;
  onAddColumn: () => void;
  onShowAll: () => void;
  servers: ServerInfo[];
  selectedServerUri: string;
  onServerChange: (uri: string) => void;
  annotationFilter: AnnotationFilter;
  onAnnotationFilterChange: (filter: AnnotationFilter) => void;
}

function Toolbar({
  facetsLoading,
  facetCount,
  activeFilterCount,
  showingAll,
  selectedItem,
  onOpenInAnnotate,
  onRefresh,
  onAddColumn,
  onShowAll,
  servers,
  selectedServerUri,
  onServerChange,
  annotationFilter,
  onAnnotationFilterChange,
}: ToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
      <span className="text-sm font-semibold text-slate-200 whitespace-nowrap">Metadata Browser</span>

      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-xs text-slate-400">
          Server
          <select
            value={selectedServerUri}
            onChange={(e) => onServerChange(e.target.value)}
            className="text-xs rounded px-2 py-1 bg-slate-900 text-slate-200 border border-slate-600 focus:outline-none focus:ring-1 focus:ring-sky-500 min-w-[180px] max-w-[280px]"
          >
            {servers.map((s) => (
              <option key={s.uri} value={s.uri}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-slate-400">
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
      </div>

      {facetsLoading && <span className="text-xs text-slate-500">Loading fields…</span>}

      <div className="flex items-center gap-1 ml-auto">
        {activeFilterCount > 0 && (
          <span className="text-xs px-2 py-0.5 rounded bg-blue-950 text-blue-300">
            {activeFilterCount} filter{activeFilterCount !== 1 ? 's' : ''} active
          </span>
        )}

        {selectedItem && (
          <button
            type="button"
            onClick={() => onOpenInAnnotate(selectedItem)}
            title="Open this scan in the Annotate tab"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors bg-sky-700 text-white border border-sky-600 hover:bg-sky-600"
          >
            <PencilSimple size={13} />
            Open in Annotate
          </button>
        )}

        <button
          type="button"
          onClick={onRefresh}
          title="Refresh"
          className="flex items-center gap-1 px-2 py-1 rounded text-xs text-slate-500 hover:bg-slate-700 transition-colors"
        >
          <ArrowsClockwise size={13} />
        </button>
        <button
          type="button"
          onClick={onShowAll}
          title="Show every sample without filtering"
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium border transition-colors ${
            showingAll
              ? 'bg-slate-600 text-white border-slate-500'
              : 'bg-slate-700 text-slate-200 border-slate-600 hover:bg-slate-600'
          }`}
        >
          <Stack size={12} />
          All samples
        </button>
        <button
          type="button"
          onClick={onAddColumn}
          disabled={facetsLoading || facetCount === 0}
          className="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium bg-blue-700 text-white transition-colors disabled:opacity-40 hover:bg-blue-600"
        >
          <Plus size={12} />
          Add column
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

interface DetailPanelSlotProps {
  item: BrowseItem | null;
  width: number;
  onClose: () => void;
  serverUri: string;
  onOpenInAnnotate: (item: BrowseItem) => void;
}

function DetailPanelSlot({ item, width, onClose, serverUri, onOpenInAnnotate }: DetailPanelSlotProps) {
  return (
    <div
      className="relative z-10 flex h-full shrink-0 flex-col border-l border-slate-700 bg-slate-900 shadow-[-4px_0_12px_rgba(0,0,0,0.25)]"
      style={{ width }}
    >
      <BrowseDetailPanel
        item={item}
        width={width}
        onClose={onClose}
        serverUri={serverUri}
        onOpenInAnnotate={() => onOpenInAnnotate(item)}
      />
    </div>
  );
}
