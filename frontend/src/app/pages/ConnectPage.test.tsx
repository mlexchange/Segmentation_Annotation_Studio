/**
 * ConnectPage — component tests for server selection, connect/verify flows
 * (Tiled + Local), navigation, and the optional dataset-container picker.
 *
 * IngestDropzone/ZarrLoader and useOpenInAnnotate are mocked: they are
 * separate, already-complex units with their own fetch/upload logic, and
 * ConnectPage only needs to exercise the callbacks it wires into them
 * (browseIngested / annotateIngested / annotateZarr).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ConnectPage from './ConnectPage';
import { useConnectionStore } from '@/stores/connectionStore';

const openTiledArray = vi.fn();

vi.mock('@/hooks/useOpenInAnnotate', () => ({
  useOpenInAnnotate: () => ({ openTiledArray }),
}));

vi.mock('@/components/Ingest/IngestDropzone', () => ({
  default: (props: { onBrowse?: (c: string, n: number) => void; onAnnotate?: (c: string, k: string) => void }) => (
    <div data-testid="ingest-dropzone">
      <button onClick={() => props.onBrowse?.('browse/foo', 3)}>mock-ingest-browse</button>
      <button onClick={() => props.onAnnotate?.('browse/foo', 'img1')}>mock-ingest-annotate</button>
    </div>
  ),
}));

vi.mock('@/components/Ingest/ZarrLoader', () => ({
  default: (props: { onAnnotate?: (p: string) => void }) => (
    <div data-testid="zarr-loader">
      <button onClick={() => props.onAnnotate?.('browse/vol/multiscale/level_0/array')}>mock-zarr-annotate</button>
    </div>
  ),
}));

const initialConnectionState = useConnectionStore.getState();

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  } as Response;
}

const SERVERS = [
  { name: 'Local Tiled', uri: 'http://localhost:8000/api', has_api_key: false },
  { name: 'Remote Tiled', uri: 'http://remote:8000/api', has_api_key: true },
];

/** Default fetch router; tests override specific endpoints as needed. */
function makeFetchMock(overrides: Partial<Record<string, (url: string) => Response | Promise<Response>>> = {}) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (overrides.servers && url.includes('/api/config/servers')) return overrides.servers(url);
    if (url.includes('/api/config/servers')) return jsonResponse(SERVERS);

    if (overrides.tiledList && url.includes('/api/tiled/list')) return overrides.tiledList(url);
    if (url.includes('/api/tiled/list')) return jsonResponse([]);

    if (overrides.localList && url.includes('/api/local/list')) return overrides.localList(url);
    if (url.includes('/api/local/list')) return jsonResponse([]);

    if (overrides.summary && url.includes('/api/connect/summary')) return overrides.summary(url);
    if (url.includes('/api/connect/summary')) return jsonResponse({ kind: 'tiled', server_uri: SERVERS[0].uri, label: 'Local Tiled', sample_count: 5 });

    throw new Error(`Unhandled fetch: ${url}`);
  });
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/connect']}>
        <Routes>
          <Route path="/connect" element={<ConnectPage />} />
          <Route path="/browse" element={<div>BROWSE PAGE</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('ConnectPage', () => {
  beforeEach(() => {
    useConnectionStore.setState(initialConnectionState, true);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('shows a loading state before the server list arrives, then lists servers', async () => {
    global.fetch = makeFetchMock();
    renderPage();

    expect(screen.getByText(/Loading servers…/)).toBeInTheDocument();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    const select = screen.getByRole('combobox', { name: 'Tiled server' }) as HTMLSelectElement;
    expect(within(select).getByText(/Local Tiled/)).toBeInTheDocument();
    expect(within(select).getByText(/Remote Tiled/)).toBeInTheDocument();
    // Auto-selects the first server once the list loads.
    await waitFor(() => expect(select.value).toBe(SERVERS[0].uri));
  });

  it('connects to the selected Tiled server (verify) and reveals "Go to Browse" without navigating', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByRole('button', { name: /Connect/ }));

    await screen.findByText(/Connected — 5 samples found/);
    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'tiled',
      serverUri: SERVERS[0].uri,
      label: 'Local Tiled',
      sampleCount: 5,
    });

    // Verify does not navigate away; it reveals a separate action instead.
    expect(screen.queryByText('BROWSE PAGE')).not.toBeInTheDocument();
    const goToBrowse = await screen.findByRole('button', { name: /Go to Browse/ });

    await user.click(goToBrowse);
    expect(await screen.findByText('BROWSE PAGE')).toBeInTheDocument();
  });

  it('shows a failure message when connecting to Tiled fails', async () => {
    global.fetch = makeFetchMock({
      summary: async () => jsonResponse('server unreachable', false),
    });
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByRole('button', { name: /Connect/ }));

    await screen.findByText(/Failed: Error: server unreachable/);
    expect(screen.queryByRole('button', { name: /Go to Browse/ })).not.toBeInTheDocument();
    expect(useConnectionStore.getState().kind).toBeNull();
  });

  it('switching Tiled server resets the verified/connected state', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByRole('button', { name: /Connect/ }));
    await screen.findByText(/Connected —/);

    const select = screen.getByRole('combobox', { name: 'Tiled server' });
    await user.selectOptions(select, SERVERS[1].uri);

    expect(screen.queryByRole('button', { name: /Go to Browse/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/Connected —/)).not.toBeInTheDocument();
  });

  it('lets you browse into a sub-container and connects with that container selected', async () => {
    global.fetch = makeFetchMock({
      tiledList: async (url) => {
        if (url.includes('path=sub')) return jsonResponse([]);
        return jsonResponse([{ name: 'sub', path: 'sub', is_dir: true, is_array: false }]);
      },
      summary: async (url) => {
        expect(url).toContain('container_path=sub');
        return jsonResponse({ kind: 'tiled', server_uri: SERVERS[0].uri, label: 'Local Tiled', sample_count: 2 });
      },
    });
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByRole('button', { name: /Dataset to view/ }));
    await user.click(await screen.findByText('sub'));
    await user.click(await screen.findByText('Browse "sub"'));
    expect(screen.getByText(/Browse will show/)).toHaveTextContent('sub');

    await user.click(screen.getByRole('button', { name: /Connect/ }));
    await screen.findByText(/Connected — 2 samples found/);
    expect(useConnectionStore.getState().browseContainerPath).toBe('sub');
  });

  it('local mode: grants a root, browses folders, selects one, and connects (navigates to Browse)', async () => {
    global.fetch = makeFetchMock({
      localList: async (url) => {
        if (url.includes('rel=data')) return jsonResponse([]);
        return jsonResponse([{ name: 'data', path: 'data', is_dir: true, size: null }]);
      },
      summary: async () => jsonResponse({ kind: 'local', server_uri: null, label: '/root/data', sample_count: 7 }),
    });
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByRole('button', { name: /Local Folder/ }));
    await user.type(screen.getByPlaceholderText('/absolute/path/to/data'), '/root');
    await user.click(screen.getByRole('button', { name: 'Grant' }));

    await screen.findByText('data');
    await user.click(screen.getByText('data'));
    await user.click(screen.getByText(/Use "data" as dataset folder/));

    const connectBtn = screen.getByRole('button', { name: 'Connect' });
    expect(connectBtn).not.toBeDisabled();
    await user.click(connectBtn);

    await screen.findByText(/Connected — 7 samples found/);
    expect(await screen.findByText('BROWSE PAGE')).toBeInTheDocument();
    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'local',
      localRoot: '/root',
      localRel: 'data',
      sampleCount: 7,
    });
  });

  it('local mode: Connect is disabled until a root is granted and a folder is selected', async () => {
    // Note: selecting "this root" itself sets selectedFolder to '' (the
    // relative path of the root), which is falsy in the component's
    // `canConnect` check — so Connect stays disabled unless an actual
    // sub-folder is chosen. This test documents that real behavior.
    global.fetch = makeFetchMock({
      localList: async (url) => {
        if (url.includes('rel=data')) return jsonResponse([]);
        return jsonResponse([{ name: 'data', path: 'data', is_dir: true, size: null }]);
      },
    });
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByRole('button', { name: /Local Folder/ }));
    expect(screen.getByRole('button', { name: 'Connect' })).toBeDisabled();

    await user.type(screen.getByPlaceholderText('/absolute/path/to/data'), '/root');
    await user.click(screen.getByRole('button', { name: 'Grant' }));
    await screen.findByText('data');
    // Root granted but no folder explicitly selected yet.
    expect(screen.getByRole('button', { name: 'Connect' })).toBeDisabled();
    // Selecting "this root" does not count as a folder selection either.
    await user.click(screen.getByText(/Use "this root" as dataset folder/));
    expect(screen.getByRole('button', { name: 'Connect' })).toBeDisabled();

    await user.click(screen.getByText('data'));
    await user.click(screen.getByText(/Use "data" as dataset folder/));
    expect(screen.getByRole('button', { name: 'Connect' })).not.toBeDisabled();
  });

  it('local mode: shows a failure message when connect fails', async () => {
    global.fetch = makeFetchMock({
      localList: async (url) => {
        if (url.includes('rel=data')) return jsonResponse([]);
        return jsonResponse([{ name: 'data', path: 'data', is_dir: true, size: null }]);
      },
      summary: async () => jsonResponse('bad path', false),
    });
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByRole('button', { name: /Local Folder/ }));
    await user.type(screen.getByPlaceholderText('/absolute/path/to/data'), '/root');
    await user.click(screen.getByRole('button', { name: 'Grant' }));
    await screen.findByText('data');
    await user.click(screen.getByText('data'));
    await user.click(screen.getByText(/Use "data" as dataset folder/));
    await user.click(screen.getByRole('button', { name: 'Connect' }));

    await screen.findByText(/Failed: Error: bad path/);
  });

  it('browsing an ingested dataset sets the connection (no container, focus on the dataset) and navigates', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByText('mock-ingest-browse'));

    expect(await screen.findByText('BROWSE PAGE')).toBeInTheDocument();
    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'tiled',
      browseContainerPath: null,
      browseFocusPath: 'browse/foo',
    });
  });

  it('annotating an ingested sample sets the connection (stays put) and opens it in Annotate', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByText('mock-ingest-annotate'));

    // annotateIngested does not navigate to Browse.
    expect(screen.queryByText('BROWSE PAGE')).not.toBeInTheDocument();
    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'tiled',
      browseContainerPath: 'browse/foo',
    });
    expect(openTiledArray).toHaveBeenCalledWith('browse/foo/img1', SERVERS[0].uri);
  });

  it('annotating a registered Zarr level derives the container path and opens it in Annotate', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    await screen.findByRole('combobox', { name: 'Tiled server' });
    await user.click(screen.getByText('mock-zarr-annotate'));

    expect(screen.queryByText('BROWSE PAGE')).not.toBeInTheDocument();
    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'tiled',
      browseContainerPath: 'browse/vol',
    });
    expect(openTiledArray).toHaveBeenCalledWith('browse/vol/multiscale/level_0/array', SERVERS[0].uri);
  });
});
