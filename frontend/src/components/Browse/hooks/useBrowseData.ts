/**
 * Browse UI data hook.
 *
 * Fetches facets, column values, and leaf items from the backend Browse API
 * and exposes a small set of immutable actions to drive the column-browser UI.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { API_BASE } from '@/config';

const BROWSE_TIMEOUT_MS = 60_000;
const FACETS_POLL_INTERVAL_MS = 30_000;

async function fetchJson<T>(url: string, timeoutMs = BROWSE_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(id);
  }
}

export interface BrowseValue {
  value: string;
  count: number;
  sample_paths: string[];
}

export interface BrowseItem {
  path: string;
  sample: string;
  metadata: Record<string, unknown>;
}

export interface ColumnState {
  field: string;
  values: BrowseValue[];
  loading: boolean;
  error: string | null;
  selected: string | null;
}

export interface BrowseState {
  columns: ColumnState[];
  items: BrowseItem[];
  itemsTotal: number;
  itemsLoading: boolean;
  facets: string[];
  facetsLoading: boolean;
  selectedItem: BrowseItem | null;
  connectionStatus: 'loading' | 'connected' | 'disconnected';
}

const INITIAL_STATE: BrowseState = {
  columns: [],
  items: [],
  itemsTotal: 0,
  itemsLoading: false,
  facets: [],
  facetsLoading: true,
  selectedItem: null,
  connectionStatus: 'loading',
};

/** Build a filter dict from the first `upToIndex` selected column values. */
function buildFilters(columns: ColumnState[], upToIndex: number): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < upToIndex; i++) {
    const col = columns[i];
    if (col && col.selected !== null) out[col.field] = col.selected;
  }
  return out;
}

function buildUrl(
  path: string,
  params: Record<string, string | number>,
  serverUri?: string,
  serverApiKey?: string,
): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) qs.set(k, String(v));
  if (serverUri) qs.set('server_uri', serverUri);
  if (serverApiKey) qs.set('server_api_key', serverApiKey);
  return `${API_BASE}${path}?${qs.toString()}`;
}

export function useBrowseData(serverUri: string, technique: string, serverApiKey?: string) {
  const [state, setState] = useState<BrowseState>(INITIAL_STATE);

  // Stable reference so `refresh` can read the latest columns without re-creating itself.
  const stateRef = useRef(state);
  stateRef.current = state;

  // ---- server params wrapped in a stable ref so callbacks don't change identity ----
  const paramsRef = useRef({ serverUri, technique, serverApiKey });
  paramsRef.current = { serverUri, technique, serverApiKey };

  // ------------------------------------------------------------------
  // API calls
  // ------------------------------------------------------------------
  const loadFacets = useCallback(async (options?: { silent?: boolean }) => {
    const { serverUri: su, technique: tq, serverApiKey: sk } = paramsRef.current;
    const silent = Boolean(options?.silent);
    if (!silent) {
      setState((s) => ({ ...s, facetsLoading: true }));
    }
    try {
      // 'All' can hit stale empty facet caches; force a refresh.
      const params: Record<string, string> = { technique: tq };
      if (tq === 'All') params.refresh = 'true';
      const data = await fetchJson<{ facets?: string[] }>(buildUrl('/api/browse/facets', params, su, sk));
      setState((s) => ({
        ...s,
        facets: data.facets ?? [],
        facetsLoading: false,
        connectionStatus: 'connected',
      }));
    } catch (err) {
      console.warn('Browse facets unavailable:', err);
      setState((s) => ({ ...s, facetsLoading: false, connectionStatus: 'disconnected' }));
    }
  }, []);

  const loadColumn = useCallback(
    async (colIndex: number, field: string, filters: Record<string, string>) => {
      const { serverUri: su, technique: tq, serverApiKey: sk } = paramsRef.current;
      setState((s) => {
        if (!s.columns[colIndex]) return s;
        const cols = [...s.columns];
        cols[colIndex] = { ...cols[colIndex], loading: true, error: null };
        return { ...s, columns: cols };
      });

      try {
        const data = await fetchJson<{ values?: BrowseValue[] }>(
          buildUrl(
            '/api/browse/column',
            { technique: tq, field, filters: JSON.stringify(filters) },
            su,
            sk,
          ),
        );
        setState((s) => {
          if (!s.columns[colIndex] || s.columns[colIndex].field !== field) return s;
          const cols = [...s.columns];
          cols[colIndex] = { ...cols[colIndex], values: data.values ?? [], loading: false };
          return { ...s, columns: cols };
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setState((s) => {
          if (!s.columns[colIndex]) return s;
          const cols = [...s.columns];
          cols[colIndex] = { ...cols[colIndex], loading: false, error: msg };
          return { ...s, columns: cols };
        });
      }
    },
    [],
  );

  const loadItems = useCallback(async (filters: Record<string, string>) => {
    const { serverUri: su, technique: tq, serverApiKey: sk } = paramsRef.current;
    setState((s) => ({ ...s, itemsLoading: true }));
    try {
      // Array-level filters (e.g. `angle_id`) need a backend refresh so the
      // parent-container mapping doesn't return stale empty results.
      const shouldRefresh =
        tq === 'All' || 'angle_id' in filters || 'incident_angle_deg' in filters;

      const params: Record<string, string> = {
        technique: tq,
        filters: JSON.stringify(filters),
      };
      if (shouldRefresh) params.refresh = 'true';

      const data = await fetchJson<{ items?: BrowseItem[]; total?: number }>(
        buildUrl('/api/browse/items', params, su, sk),
      );
      setState((s) => ({
        ...s,
        items: data.items ?? [],
        itemsTotal: data.total ?? 0,
        itemsLoading: false,
      }));
    } catch (err) {
      console.warn('Browse items unavailable:', err);
      setState((s) => ({ ...s, itemsLoading: false, items: [], itemsTotal: 0 }));
    }
  }, []);

  // ------------------------------------------------------------------
  // Public actions
  // ------------------------------------------------------------------
  const addColumn = useCallback(
    (field: string) => {
      setState((s) => {
        const newCol: ColumnState = { field, values: [], loading: true, error: null, selected: null };
        const newCols = [...s.columns, newCol];
        const newIndex = newCols.length - 1;
        queueMicrotask(() => loadColumn(newIndex, field, buildFilters(newCols, newIndex)));
        return { ...s, columns: newCols };
      });
    },
    [loadColumn],
  );

  const removeColumn = useCallback((colIndex: number) => {
    setState((s) => ({
      ...s,
      columns: s.columns.slice(0, colIndex),
      items: [],
      itemsTotal: 0,
      selectedItem: null,
    }));
  }, []);

  const changeColumnField = useCallback(
    (colIndex: number, field: string) => {
      setState((s) => {
        const cols = s.columns.slice(0, colIndex);
        const newCol: ColumnState = { field, values: [], loading: true, error: null, selected: null };
        const newCols = [...cols, newCol];
        queueMicrotask(() => loadColumn(colIndex, field, buildFilters(newCols, colIndex)));
        return { ...s, columns: newCols, items: [], itemsTotal: 0, selectedItem: null };
      });
    },
    [loadColumn],
  );

  const selectValue = useCallback(
    (colIndex: number, value: string | null) => {
      setState((s) => {
        const cols = s.columns.slice(0, colIndex + 1).map((c, i) =>
          i === colIndex ? { ...c, selected: value } : c,
        );

        if (value === null) {
          return { ...s, columns: cols, items: [], itemsTotal: 0, selectedItem: null };
        }

        const hasNext = colIndex + 1 < s.columns.length;
        if (hasNext) {
          const nextCol = s.columns[colIndex + 1];
          const nextCols: ColumnState[] = [
            ...cols,
            { ...nextCol, values: [], loading: true, error: null, selected: null },
            ...s.columns
              .slice(colIndex + 2)
              .map((c) => ({ ...c, values: [], loading: false, error: null, selected: null })),
          ];
          queueMicrotask(() =>
            loadColumn(colIndex + 1, nextCol.field, buildFilters(nextCols, colIndex + 1)),
          );
          return { ...s, columns: nextCols, items: [], itemsTotal: 0, selectedItem: null };
        }

        // No next column — load leaf items.
        queueMicrotask(() => loadItems(buildFilters(cols, cols.length)));
        return { ...s, columns: cols, selectedItem: null };
      });
    },
    [loadColumn, loadItems],
  );

  const selectItem = useCallback((item: BrowseItem | null) => {
    setState((s) => ({ ...s, selectedItem: item }));
  }, []);

  const refresh = useCallback(() => {
    const s = stateRef.current;
    if (s.columns.length === 0) return;
    s.columns.forEach((col, i) => loadColumn(i, col.field, buildFilters(s.columns, i)));
    loadItems(buildFilters(s.columns, s.columns.length));
  }, [loadColumn, loadItems]);

  // ------------------------------------------------------------------
  // Init: load facets + reset columns whenever the server/technique changes
  // and re-poll facets periodically.
  // ------------------------------------------------------------------
  useEffect(() => {
    void loadFacets();
    setState((s) => ({ ...s, columns: [], items: [], itemsTotal: 0, selectedItem: null }));
    const interval = setInterval(() => {
      void loadFacets({ silent: true });
    }, FACETS_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [loadFacets, serverUri, technique, serverApiKey]);

  const actions = useMemo(
    () => ({ addColumn, removeColumn, changeColumnField, selectValue, selectItem, refresh, loadFacets }),
    [addColumn, removeColumn, changeColumnField, selectValue, selectItem, refresh, loadFacets],
  );

  return { state, actions };
}
