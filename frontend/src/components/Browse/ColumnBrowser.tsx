import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowsClockwise, CheckSquare, CloudArrowUp, Plus } from '@phosphor-icons/react';
import BrowseColumn from './BrowseColumn';
import BrowseDetailPanel from './BrowseDetailPanel';
import ItemsColumn from './ItemsColumn';
import ResizeDivider from './ResizeDivider';
import { useBrowseData, type BrowseItem } from './hooks/useBrowseData';
import { useBrowseStore, type StagedItem } from '@/stores/browseStore';

interface ColumnBrowserProps {
  serverUri: string;
  technique: string;
  serverApiKey?: string;
}

const DEFAULT_COLUMN_WIDTH = 220;
const DEFAULT_ITEMS_WIDTH = 260;
const SAMPLE_NAME_COLUMN_WIDTH = 190;
const INITIAL_COLUMN_COUNT = 5;
const STAGED_MESSAGE_DURATION_MS = 3500;

export default function ColumnBrowser({ serverUri, technique, serverApiKey }: ColumnBrowserProps) {
  const { state, actions } = useBrowseData(serverUri, technique, serverApiKey);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [columnWidths, setColumnWidths] = useState<number[]>([]);
  const [itemsColumnWidth, setItemsColumnWidth] = useState(DEFAULT_ITEMS_WIDTH);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [stagedMsg, setStagedMsg] = useState<string | null>(null);

  const setStagedItems = useBrowseStore((s) => s.setStagedItems);
  const stagedItems = useBrowseStore((s) => s.stagedItems);
  const setMetadataDisplayKeys = useBrowseStore((s) => s.setMetadataDisplayKeys);

  // Keep columnWidths in sync with the number of columns.
  useEffect(() => {
    setColumnWidths((prev) => {
      const n = state.columns.length;
      if (prev.length === n) return prev;
      if (prev.length > n) return prev.slice(0, n);
      const additions = Array.from({ length: n - prev.length }, (_, i) => {
        const field = state.columns[prev.length + i]?.field;
        return field === 'sample_name' ? SAMPLE_NAME_COLUMN_WIDTH : DEFAULT_COLUMN_WIDTH;
      });
      return [...prev, ...additions];
    });
  }, [state.columns]);

  useEffect(() => {
    setSelectedPaths(new Set());
  }, [state.items]);

  // Auto-scroll to the newest column when columns are added.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [state.columns.length]);

  // First-load: populate with the first few discovered facets.
  const initialised = useRef(false);
  useEffect(() => {
    if (initialised.current || state.facets.length === 0) return;
    initialised.current = true;
    const n = Math.min(INITIAL_COLUMN_COUNT, state.facets.length);
    state.facets.slice(0, n).forEach((f) => actions.addColumn(f));
  }, [state.facets, actions]);

  // Publish current column fields for other tabs that mirror the selection.
  useEffect(() => {
    setMetadataDisplayKeys(state.columns.map((col) => col.field));
  }, [state.columns, setMetadataDisplayKeys]);

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

  const togglePath = useCallback((path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedPaths(new Set(state.items.map((i) => i.path)));
  }, [state.items]);

  const clearSelection = useCallback(() => setSelectedPaths(new Set()), []);

  const stageSelected = useCallback(() => {
    const stagedIds = new Set(stagedItems.map((s) => s.tiledId));
    const toAdd: StagedItem[] = state.items
      .filter((item) => selectedPaths.has(item.path) && !stagedIds.has(item.path))
      .map((item) => ({
        id: `browse-${item.path}-${Date.now()}`,
        name: String(item.metadata.sample_name ?? item.sample),
        source: 'tiled',
        tiledId: item.path,
        tiledUri: serverUri || undefined,
        tiledApiKey: serverApiKey || undefined,
        metadata: {
          sample_name: String(item.metadata.sample_name ?? item.sample),
          bar: typeof item.metadata.bar === 'number' ? item.metadata.bar : undefined,
          sample_folder: item.metadata.sample_folder as string | undefined,
          beamline: item.metadata.beamline as string | undefined,
        },
      }));

    if (toAdd.length === 0) return;
    setStagedItems([...stagedItems, ...toAdd]);

    const msg =
      toAdd.length === 1
        ? `"${toAdd[0].name}" added to staging`
        : `${toAdd.length} scans added to staging`;
    setStagedMsg(msg);
    setSelectedPaths(new Set());
    setTimeout(() => setStagedMsg(null), STAGED_MESSAGE_DURATION_MS);
  }, [state.items, selectedPaths, stagedItems, setStagedItems, serverUri, serverApiKey]);

  const activeFilters = useMemo(() => {
    const out: Record<string, string> = {};
    state.columns.forEach((col) => {
      if (col.selected !== null) out[col.field] = col.selected;
    });
    return out;
  }, [state.columns]);

  const activeFilterCount = Object.keys(activeFilters).length;
  const lastColumn = state.columns[state.columns.length - 1];
  const showItems = state.columns.length > 0 && lastColumn?.selected !== null;

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-200">
      <Toolbar
        facetsLoading={state.facetsLoading}
        facetCount={state.facets.length}
        activeFilterCount={activeFilterCount}
        selectedCount={selectedPaths.size}
        onStage={stageSelected}
        onRefresh={actions.refresh}
        onAddColumn={handleAddColumn}
      />

      {stagedMsg && (
        <div className="shrink-0 px-4 py-1.5 text-xs font-medium border-b border-green-800 bg-green-950 text-green-300 flex items-center gap-2">
          <CheckSquare size={13} weight="fill" />
          {stagedMsg}
        </div>
      )}

      {state.connectionStatus === 'disconnected' && (
        <div className="shrink-0 px-4 py-2 text-xs border-b border-red-900 bg-red-950/80 text-red-300">
          Cannot reach the API server. Make sure the backend (port 8002) and Tiled server are running.
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <div ref={scrollRef} className="flex flex-1 overflow-x-auto overflow-y-hidden min-w-0">
          {state.columns.length === 0 && !state.facetsLoading && (
            <div className="flex items-center justify-center flex-1">
              <p className="text-sm text-slate-500">
                {state.connectionStatus === 'disconnected'
                  ? 'Connect to a Tiled server to browse.'
                  : 'Click "Add column" to start browsing.'}
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
                width={itemsColumnWidth}
                checkedPaths={selectedPaths}
                onToggleCheck={togglePath}
                onSelectAll={selectAll}
                onClearAll={clearSelection}
              />
            </>
          )}
        </div>

        <DetailPanelSlot
          item={state.selectedItem}
          onClose={() => actions.selectItem(null)}
          serverUri={serverUri}
          serverApiKey={serverApiKey}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

interface ToolbarProps {
  facetsLoading: boolean;
  facetCount: number;
  activeFilterCount: number;
  selectedCount: number;
  onStage: () => void;
  onRefresh: () => void;
  onAddColumn: () => void;
}

function Toolbar({
  facetsLoading,
  facetCount,
  activeFilterCount,
  selectedCount,
  onStage,
  onRefresh,
  onAddColumn,
}: ToolbarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
      <span className="text-sm font-semibold text-slate-400">Metadata Browser</span>
      {facetsLoading && <span className="text-xs text-slate-500">Loading fields…</span>}

      <div className="flex items-center gap-1 ml-auto">
        {activeFilterCount > 0 && (
          <span className="text-xs px-2 py-0.5 rounded bg-blue-950 text-blue-300">
            {activeFilterCount} filter{activeFilterCount !== 1 ? 's' : ''} active
          </span>
        )}

        {selectedCount > 0 && (
          <button
            type="button"
            onClick={onStage}
            title="Add selected scans to the staging area"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors bg-green-900 text-green-300 border border-green-700 hover:bg-green-800"
          >
            <CloudArrowUp size={13} />
            Stage {selectedCount}
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
  onClose: () => void;
  serverUri: string;
  serverApiKey?: string;
}

function DetailPanelSlot({ item, onClose, serverUri, serverApiKey }: DetailPanelSlotProps) {
  return (
    <div
      className="flex flex-col h-full border-l border-slate-700 bg-slate-900 shrink-0"
      style={{ width: item ? 360 : 280, minWidth: 200 }}
    >
      {item ? (
        <BrowseDetailPanel
          item={item}
          onClose={onClose}
          serverUri={serverUri}
          serverApiKey={serverApiKey}
        />
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center px-4 py-8 text-center">
          <p className="text-sm font-medium text-slate-400">Metadata</p>
          <p className="text-xs mt-2 text-slate-500">
            Select a sample in the list to view its metadata here.
          </p>
        </div>
      )}
    </div>
  );
}
