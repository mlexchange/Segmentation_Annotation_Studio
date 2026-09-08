/**
 * useBrowseData — hook tests for the Browse data-fetching/state layer.
 * Fetch is stubbed per-endpoint; renderHook drives the hook directly (no DOM).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useBrowseData, type BrowseItem } from './useBrowseData';

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  } as Response;
}

/** Routes fetch calls to per-endpoint handlers by pathname; unmatched calls 404. */
function makeFetchMock(
  overrides: Record<string, (url: URL) => Response | Promise<Response>> = {},
) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost');
    for (const [path, handler] of Object.entries(overrides)) {
      if (url.pathname === path) return handler(url);
    }
    return jsonResponse({}, false);
  });
}

const item = (path: string, extra: Partial<BrowseItem> = {}): BrowseItem => ({
  path,
  sample: path,
  metadata: {},
  ...extra,
});

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  cleanupFakeTimers();
  vi.restoreAllMocks();
});

function cleanupFakeTimers() {
  if (vi.isFakeTimers()) vi.useRealTimers();
}

describe('useBrowseData', () => {
  it('starts with facetsLoading true and loads facets on mount, marking connected', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['sample_name', 'technique'] }),
    });

    const { result } = renderHook(() => useBrowseData('http://server', 'All'));

    expect(result.current.state.facetsLoading).toBe(true);
    expect(result.current.state.connectionStatus).toBe('loading');

    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    expect(result.current.state.facets).toEqual(['sample_name', 'technique']);
    expect(result.current.state.connectionStatus).toBe('connected');
  });

  it('marks the connection disconnected when the facets request fails', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse('boom', false),
    });

    const { result } = renderHook(() => useBrowseData('http://server', 'All'));

    await waitFor(() => expect(result.current.state.connectionStatus).toBe('disconnected'));
    expect(result.current.state.facetsLoading).toBe(false);
  });

  it('requests facets with technique/container_path/server params', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
    });
    global.fetch = fetchMock;

    renderHook(() => useBrowseData('http://server', 'SAXS', 'secret-key', 'browse/foo'));

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const calledUrl = new URL(String(fetchMock.mock.calls[0][0]), 'http://localhost');
    expect(calledUrl.searchParams.get('technique')).toBe('SAXS');
    expect(calledUrl.searchParams.get('container_path')).toBe('browse/foo');
    expect(calledUrl.searchParams.get('server_uri')).toBe('http://server');
    expect(calledUrl.searchParams.get('server_api_key')).toBe('secret-key');
    // technique !== 'All' so no forced refresh.
    expect(calledUrl.searchParams.get('refresh')).toBeNull();
  });

  it('forces a refresh for technique "All"', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
    });
    global.fetch = fetchMock;

    renderHook(() => useBrowseData('http://server', 'All'));

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const calledUrl = new URL(String(fetchMock.mock.calls[0][0]), 'http://localhost');
    expect(calledUrl.searchParams.get('refresh')).toBe('true');
  });

  it('addColumn appends a loading column then fills in its values', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1'] }),
      '/api/browse/column': () =>
        jsonResponse({ values: [{ value: 'v1', count: 3, sample_paths: [] }] }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.addColumn('field1'));
    expect(result.current.state.columns).toHaveLength(1);
    expect(result.current.state.columns[0]).toMatchObject({ field: 'field1', loading: true });

    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    expect(result.current.state.columns[0].values).toEqual([
      { value: 'v1', count: 3, sample_paths: [] },
    ]);
  });

  it('records a column error when the column request fails', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1'] }),
      '/api/browse/column': () => jsonResponse('nope', false),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.addColumn('field1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    expect(result.current.state.columns[0].error).toBe('HTTP 500');
  });

  it('selectValue on the last column loads leaf items', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1'] }),
      '/api/browse/column': () =>
        jsonResponse({ values: [{ value: 'v1', count: 1, sample_paths: [] }] }),
      '/api/browse/items': () =>
        jsonResponse({ items: [item('p1', { sample: 's1' })], total: 1 }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.addColumn('field1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));

    act(() => result.current.actions.selectValue(0, 'v1'));
    expect(result.current.state.columns[0].selected).toBe('v1');

    await waitFor(() => expect(result.current.state.itemsLoading).toBe(false));
    expect(result.current.state.items).toHaveLength(1);
    expect(result.current.state.items[0].path).toBe('p1');
    expect(result.current.state.itemsTotal).toBe(1);
  });

  it('selectValue(null) clears the selection and items without fetching items', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1'] }),
      '/api/browse/column': () =>
        jsonResponse({ values: [{ value: 'v1', count: 1, sample_paths: [] }] }),
      '/api/browse/items': () => jsonResponse({ items: [item('p1')], total: 1 }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.addColumn('field1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    act(() => result.current.actions.selectValue(0, 'v1'));
    await waitFor(() => expect(result.current.state.items).toHaveLength(1));

    act(() => result.current.actions.selectValue(0, null));
    expect(result.current.state.columns[0].selected).toBeNull();
    expect(result.current.state.items).toHaveLength(0);
    expect(result.current.state.itemsTotal).toBe(0);
  });

  it('selecting a value with a next column loads that column instead of items', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1', 'f2'] }),
      '/api/browse/column': (url) => {
        const field = url.searchParams.get('field');
        if (field === 'f1') return jsonResponse({ values: [{ value: 'a', count: 2, sample_paths: [] }] });
        return jsonResponse({ values: [{ value: 'b', count: 1, sample_paths: [] }] });
      },
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.addColumn('f1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    act(() => result.current.actions.addColumn('f2'));
    await waitFor(() => expect(result.current.state.columns[1]?.loading).toBe(false));

    act(() => result.current.actions.selectValue(0, 'a'));
    // Second column reloads (filtered by field1=a) rather than leaf items loading.
    await waitFor(() => expect(result.current.state.columns[1].loading).toBe(false));
    expect(result.current.state.columns[1].selected).toBeNull();
    expect(result.current.state.columns[1].values).toEqual([{ value: 'b', count: 1, sample_paths: [] }]);
    expect(result.current.state.itemsLoading).toBe(false);
    expect(result.current.state.items).toHaveLength(0);
  });

  it('removeColumn drops that column and everything after it', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1', 'f2'] }),
      '/api/browse/column': () => jsonResponse({ values: [] }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.addColumn('f1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    act(() => result.current.actions.addColumn('f2'));
    await waitFor(() => expect(result.current.state.columns[1]?.loading).toBe(false));

    act(() => result.current.actions.removeColumn(1));
    expect(result.current.state.columns).toHaveLength(1);

    act(() => result.current.actions.removeColumn(0));
    expect(result.current.state.columns).toHaveLength(0);
  });

  it('changeColumnField replaces a column and drops later ones', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1', 'f2', 'f3'] }),
      '/api/browse/column': (url) =>
        jsonResponse({ values: [{ value: `v-${url.searchParams.get('field')}`, count: 1, sample_paths: [] }] }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.addColumn('f1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    act(() => result.current.actions.addColumn('f2'));
    await waitFor(() => expect(result.current.state.columns[1]?.loading).toBe(false));

    act(() => result.current.actions.changeColumnField(1, 'f3'));
    await waitFor(() => expect(result.current.state.columns[1]?.loading).toBe(false));
    expect(result.current.state.columns).toHaveLength(2);
    expect(result.current.state.columns[1].field).toBe('f3');
    expect(result.current.state.columns[1].values[0].value).toBe('v-f3');
  });

  it('showAll clears columns, sets showingAll, and loads every item', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1'] }),
      '/api/browse/items': () => jsonResponse({ items: [item('a'), item('b')], total: 2 }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.showAll());
    expect(result.current.state.showingAll).toBe(true);
    expect(result.current.state.columns).toHaveLength(0);

    await waitFor(() => expect(result.current.state.itemsLoading).toBe(false));
    expect(result.current.state.items).toHaveLength(2);
    expect(result.current.state.itemsTotal).toBe(2);
  });

  it('items fetch failure clears items and total', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () => jsonResponse('nope', false),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.showAll());

    await waitFor(() => expect(result.current.state.itemsLoading).toBe(false));
    expect(result.current.state.items).toEqual([]);
    expect(result.current.state.itemsTotal).toBe(0);
  });

  it('expandSample loads slices for a multi-image item, and null collapses it', async () => {
    const dataset = item('vol1', { sample: 'vol1', n_slices: 3 });
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/slices': () =>
        jsonResponse({ items: [item('vol1/0'), item('vol1/1')] }),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    act(() => result.current.actions.expandSample(dataset));
    expect(result.current.state.expandedSample).toEqual(dataset);
    expect(result.current.state.slicesLoading).toBe(true);

    await waitFor(() => expect(result.current.state.slicesLoading).toBe(false));
    expect(result.current.state.slices).toHaveLength(2);

    act(() => result.current.actions.expandSample(null));
    expect(result.current.state.expandedSample).toBeNull();
    expect(result.current.state.slices).toHaveLength(0);
  });

  it('slices fetch failure clears the slices list', async () => {
    const dataset = item('vol1', { n_slices: 2 });
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/slices': () => jsonResponse('boom', false),
    });

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.expandSample(dataset));

    await waitFor(() => expect(result.current.state.slicesLoading).toBe(false));
    expect(result.current.state.slices).toEqual([]);
  });

  it('selectItem sets and clears the selected leaf item', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    const it1 = item('p1');
    act(() => result.current.actions.selectItem(it1));
    expect(result.current.state.selectedItem).toEqual(it1);
    act(() => result.current.actions.selectItem(null));
    expect(result.current.state.selectedItem).toBeNull();
  });

  it('refresh reloads columns and items using the latest state', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1'] }),
      '/api/browse/column': () => jsonResponse({ values: [{ value: 'a', count: 1, sample_paths: [] }] }),
      '/api/browse/items': () => jsonResponse({ items: [item('a')], total: 1 }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.addColumn('f1'));
    await waitFor(() => expect(result.current.state.columns[0].loading).toBe(false));
    act(() => result.current.actions.selectValue(0, 'a'));
    await waitFor(() => expect(result.current.state.itemsLoading).toBe(false));

    const callsBefore = fetchMock.mock.calls.length;
    act(() => result.current.actions.refresh());
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore));
    // One call for the (only) column, one for items.
    expect(fetchMock.mock.calls.length).toBe(callsBefore + 2);
  });

  it('refresh with showingAll re-loads items with no filters', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () => jsonResponse({ items: [item('a')], total: 1 }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.showAll());
    await waitFor(() => expect(result.current.state.itemsLoading).toBe(false));

    const callsBefore = fetchMock.mock.calls.length;
    act(() => result.current.actions.refresh());
    await waitFor(() => expect(fetchMock.mock.calls.length).toBe(callsBefore + 1));
  });

  it('refresh is a no-op when there are no columns and showingAll is false', async () => {
    const fetchMock = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    const callsBefore = fetchMock.mock.calls.length;
    act(() => result.current.actions.refresh());
    expect(fetchMock.mock.calls.length).toBe(callsBefore);
  });

  it('sets up a facets poll interval and clears it on unmount', () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    const setSpy = vi.spyOn(global, 'setInterval');
    const clearSpy = vi.spyOn(global, 'clearInterval');

    const { unmount } = renderHook(() => useBrowseData('uri', 'tech'));
    expect(setSpy).toHaveBeenCalledWith(expect.any(Function), 30_000);

    unmount();
    expect(clearSpy).toHaveBeenCalled();
  });

  it('re-fetches and resets state when serverUri/technique change', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['f1'] }),
      '/api/browse/items': () => jsonResponse({ items: [item('a')], total: 1 }),
    });
    global.fetch = fetchMock;

    const { result, rerender } = renderHook(
      ({ serverUri, technique }) => useBrowseData(serverUri, technique),
      { initialProps: { serverUri: 'uri1', technique: 'All' } },
    );
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
    act(() => result.current.actions.showAll());
    await waitFor(() => expect(result.current.state.items).toHaveLength(1));

    rerender({ serverUri: 'uri2', technique: 'All' });
    // Reset wipes items/columns/showingAll immediately.
    expect(result.current.state.items).toHaveLength(0);
    expect(result.current.state.showingAll).toBe(false);
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));
  });

  it('loadFacets({silent: true}) does not toggle facetsLoading', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: ['x'] }) });
    const { result } = renderHook(() => useBrowseData('uri', 'tech'));
    await waitFor(() => expect(result.current.state.facetsLoading).toBe(false));

    let sawLoadingTrue = false;
    await act(async () => {
      const p = result.current.actions.loadFacets({ silent: true });
      sawLoadingTrue = result.current.state.facetsLoading;
      await p;
    });
    expect(sawLoadingTrue).toBe(false);
    expect(result.current.state.facetsLoading).toBe(false);
    expect(result.current.state.facets).toEqual(['x']);
  });
});
