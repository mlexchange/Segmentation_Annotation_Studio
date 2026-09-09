/**
 * ColumnBrowser — tests the Miller-column orchestration: toolbar wiring, the
 * auto "show all" on connect, adding columns, item/slice selection routing,
 * open-in-Annotate, and connection-error display.
 *
 * BrowseColumn/ItemsColumn/SlicesColumn/BrowseDetailPanel are mocked out — they
 * are separate, already-tested units (owned by a parallel test-writing pass);
 * ColumnBrowser only needs to exercise the props/callbacks it wires into them.
 * useBrowseData itself is real, so the actual fetch-driven state machine
 * (facets → show-all → items) still runs end to end against a stubbed fetch.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import ColumnBrowser from './ColumnBrowser';
import type { BrowseItem } from './hooks/useBrowseData';
import type { ServerInfo } from '@/types/server';

const openTiledArray = vi.fn(async () => {});
vi.mock('@/hooks/useOpenInAnnotate', () => ({
  useOpenInAnnotate: () => ({ openTiledArray, openLocalFile: vi.fn() }),
}));

vi.mock('./BrowseColumn', () => ({
  default: (props: any) => (
    <div data-testid={`browse-column-${props.colIndex}`}>
      <span>field:{props.column.field}</span>
      <button onClick={() => props.onRemove(props.colIndex)}>remove-{props.colIndex}</button>
      <button onClick={() => props.onSelect(props.colIndex, 'v1')}>select-v1-{props.colIndex}</button>
    </div>
  ),
}));

vi.mock('./ItemsColumn', () => ({
  default: (props: any) => (
    <div data-testid="items-column">
      {props.items.map((it: BrowseItem) => (
        <button key={it.path} onClick={() => props.onSelect(it)}>
          item:{it.sample}
        </button>
      ))}
    </div>
  ),
}));

vi.mock('./SlicesColumn', () => ({
  default: (props: any) => (
    <div data-testid="slices-column">
      <span>slices-of:{props.dataset.sample}</span>
      <button onClick={props.onClose}>close-slices</button>
    </div>
  ),
}));

vi.mock('./BrowseDetailPanel', () => ({
  default: (props: any) => (
    <div data-testid="detail-panel">
      <span>detail:{props.item.sample}</span>
      <button onClick={props.onClose}>close-detail</button>
    </div>
  ),
}));

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  } as Response;
}

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

const SERVERS: ServerInfo[] = [
  { name: 'Server A', uri: 'http://a', has_api_key: false },
  { name: 'Server B', uri: 'http://b', has_api_key: false },
];

function renderBrowser(overrides: Partial<React.ComponentProps<typeof ColumnBrowser>> = {}) {
  const props = {
    serverUri: 'http://a',
    containerPath: null,
    focusPath: null,
    servers: SERVERS,
    selectedServerUri: 'http://a',
    onServerChange: vi.fn(),
    annotationFilter: 'all' as const,
    onAnnotationFilterChange: vi.fn(),
    ...overrides,
  };
  return { ...render(<ColumnBrowser {...props} />, { wrapper: MemoryRouter }), props };
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ColumnBrowser', () => {
  it('renders the toolbar with server + annotation selects', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    renderBrowser();
    expect(screen.getByText('Metadata Browser')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Server' })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Annotation' })).toBeInTheDocument();
    // Let the facets fetch (and its show-all effect) settle before the test ends.
    await waitFor(() => expect(screen.getByTestId('items-column')).toBeInTheDocument());
  });

  it('auto shows every sample once connected, rendering the items column', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1'] }),
      '/api/browse/items': () =>
        jsonResponse({ items: [{ path: 'p1', sample: 'sample-1', metadata: {} }], total: 1 }),
    });
    renderBrowser();

    expect(await screen.findByTestId('items-column')).toBeInTheDocument();
    expect(screen.getByText('item:sample-1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /All samples/ })).toHaveClass('bg-slate-600');
  });

  it('shows the disconnected banner when facets cannot be reached, and no items column', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse('down', false) });
    renderBrowser();

    expect(await screen.findByText(/Cannot reach the API server/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /go to connect/i })).toBeInTheDocument();
    expect(screen.queryByTestId('items-column')).not.toBeInTheDocument();
  });

  it('Add column adds the first unused facet, and adding again picks the next one', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: ['field1', 'field2'] }),
      '/api/browse/items': () => jsonResponse({ items: [], total: 0 }),
      '/api/browse/column': () => jsonResponse({ values: [] }),
    });
    const user = userEvent.setup();
    renderBrowser();
    await screen.findByTestId('items-column');

    await user.click(screen.getByRole('button', { name: 'Add column' }));
    expect(await screen.findByTestId('browse-column-0')).toHaveTextContent('field:field1');

    await user.click(screen.getByRole('button', { name: 'Add column' }));
    expect(await screen.findByTestId('browse-column-1')).toHaveTextContent('field:field2');
  });

  it('selecting a single-image sample opens the detail panel', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () =>
        jsonResponse({ items: [{ path: 'p1', sample: 'single-1', metadata: {} }], total: 1 }),
    });
    const user = userEvent.setup();
    renderBrowser();

    await user.click(await screen.findByText('item:single-1'));
    expect(await screen.findByTestId('detail-panel')).toHaveTextContent('detail:single-1');

    await user.click(screen.getByText('close-detail'));
    await waitFor(() => expect(screen.queryByTestId('detail-panel')).not.toBeInTheDocument());
  });

  it('selecting a multi-slice sample expands the slices column instead of the detail panel', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () =>
        jsonResponse({
          items: [{ path: 'vol1', sample: 'volume-1', metadata: {}, n_slices: 5 }],
          total: 1,
        }),
      '/api/browse/slices': () =>
        jsonResponse({ items: [{ path: 'vol1/0', sample: '0', metadata: {} }] }),
    });
    const user = userEvent.setup();
    renderBrowser();

    await user.click(await screen.findByText('item:volume-1'));
    expect(await screen.findByTestId('slices-column')).toHaveTextContent('slices-of:volume-1');
    expect(screen.queryByTestId('detail-panel')).not.toBeInTheDocument();

    await user.click(screen.getByText('close-slices'));
    await waitFor(() => expect(screen.queryByTestId('slices-column')).not.toBeInTheDocument());
  });

  it('Open in Annotate opens the selected item via useOpenInAnnotate', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () =>
        jsonResponse({ items: [{ path: 'p1', sample: 'single-1', metadata: {} }], total: 1 }),
    });
    const user = userEvent.setup();
    renderBrowser();

    await user.click(await screen.findByText('item:single-1'));
    await user.click(screen.getByRole('button', { name: /Open in Annotate/ }));

    await waitFor(() => expect(openTiledArray).toHaveBeenCalledWith('p1', 'http://a'));
  });

  it('auto-selects the item matching focusPath once the full list has loaded', async () => {
    global.fetch = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () =>
        jsonResponse({
          items: [
            { path: 'p1', sample: 'first', metadata: {} },
            { path: 'p2', sample: 'second', metadata: {} },
          ],
          total: 2,
        }),
    });
    renderBrowser({ focusPath: 'p2' });

    expect(await screen.findByTestId('detail-panel')).toHaveTextContent('detail:second');
  });

  it('calls onServerChange when the server select changes', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    const onServerChange = vi.fn();
    const user = userEvent.setup();
    renderBrowser({ onServerChange });

    await user.selectOptions(screen.getByRole('combobox', { name: 'Server' }), 'http://b');
    expect(onServerChange).toHaveBeenCalledWith('http://b');
  });

  it('calls onAnnotationFilterChange when the annotation select changes', async () => {
    global.fetch = makeFetchMock({ '/api/browse/facets': () => jsonResponse({ facets: [] }) });
    const onAnnotationFilterChange = vi.fn();
    const user = userEvent.setup();
    renderBrowser({ onAnnotationFilterChange });

    await user.selectOptions(screen.getByRole('combobox', { name: 'Annotation' }), 'annotated');
    expect(onAnnotationFilterChange).toHaveBeenCalledWith('annotated');
  });

  it('Refresh triggers another items fetch while showing all', async () => {
    const fetchMock = makeFetchMock({
      '/api/browse/facets': () => jsonResponse({ facets: [] }),
      '/api/browse/items': () => jsonResponse({ items: [], total: 0 }),
    });
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderBrowser();

    await screen.findByTestId('items-column');
    const callsBefore = fetchMock.mock.calls.length;
    await user.click(screen.getByRole('button', { name: 'Refresh' }));
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore));
  });
});
