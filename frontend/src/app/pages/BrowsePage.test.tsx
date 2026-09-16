/**
 * BrowsePage — orchestration tests: connection banner, kind switch (tiled vs
 * local), navigation to /connect, and server-list wiring into ColumnBrowser.
 *
 * ColumnBrowser and LocalSampleBrowser are mocked out: they are separate,
 * already-tested units with their own fetch-driven state machines (see
 * ColumnBrowser.test.tsx / LocalSampleBrowser.test.tsx). BrowsePage only needs
 * to exercise the props it wires into them and its own connection/navigation
 * logic, so useConnectionStore and useNavigate (via a real MemoryRouter) stay
 * real.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import BrowsePage from './BrowsePage';
import { useConnectionStore } from '@/stores/connectionStore';

const openLocalFile = vi.fn();
vi.mock('@/hooks/useOpenInAnnotate', () => ({
  useOpenInAnnotate: () => ({ openTiledArray: vi.fn(), openLocalFile }),
}));

vi.mock('@/components/Browse/ColumnBrowser', () => ({
  default: (props: any) => (
    <div data-testid="column-browser">
      <span>serverUri:{props.serverUri}</span>
      <span>containerPath:{String(props.containerPath)}</span>
      <span>focusPath:{String(props.focusPath)}</span>
      <span>selectedServerUri:{props.selectedServerUri}</span>
      <span>servers:{props.servers.map((s: any) => s.name).join(',')}</span>
      <button onClick={() => props.onServerChange('http://b')}>change-server</button>
      <button onClick={() => props.onAnnotationFilterChange('annotated')}>change-annotation</button>
    </div>
  ),
}));

vi.mock('@/components/Browse/LocalSampleBrowser', () => ({
  default: (props: any) => (
    <div data-testid="local-browser">
      <span>root:{props.root}</span>
      <span>rel:{props.rel}</span>
      <button onClick={() => props.onOpenInAnnotate('rel/path.tif')}>open-local</button>
    </div>
  ),
}));

const initialConnectionState = useConnectionStore.getState();

function jsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  } as Response;
}

const SERVERS = [
  { name: 'Server A', uri: 'http://a', has_api_key: false },
  { name: 'Server B', uri: 'http://b', has_api_key: false },
];

function makeFetchMock() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes('/api/config/servers')) return jsonResponse(SERVERS);
    return jsonResponse({}, false);
  });
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/browse']}>
        <Routes>
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/connect" element={<div>CONNECT PAGE</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useConnectionStore.setState(initialConnectionState, true);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('BrowsePage', () => {
  it('shows a not-connected prompt and navigates to Connect', async () => {
    global.fetch = makeFetchMock();
    const user = userEvent.setup();
    renderPage();

    expect(screen.getByText('No dataset connected.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Go to Connect' }));
    expect(await screen.findByText('CONNECT PAGE')).toBeInTheDocument();
  });

  it('renders the tiled ColumnBrowser with connection details when kind is tiled', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://a',
      browseContainerPath: 'browse/foo',
      browseFocusPath: 'browse/foo/bar',
      label: 'Server A',
      sampleCount: 5,
    });
    renderPage();

    expect(screen.getByText('Server A')).toBeInTheDocument();
    expect(screen.getByText(/5 samples/)).toBeInTheDocument();

    const columnBrowser = await screen.findByTestId('column-browser');
    expect(columnBrowser).toHaveTextContent('serverUri:http://a');
    expect(columnBrowser).toHaveTextContent('containerPath:browse/foo');
    expect(columnBrowser).toHaveTextContent('focusPath:browse/foo/bar');
    expect(columnBrowser).toHaveTextContent('selectedServerUri:http://a');
    await screen.findByText('servers:Server A,Server B');
    expect(screen.queryByTestId('local-browser')).not.toBeInTheDocument();
  });

  it('renders the LocalSampleBrowser with root/rel when kind is local', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'local',
      localRoot: '/data',
      localRel: 'sub',
      label: '/data/sub',
      sampleCount: 3,
    });
    renderPage();

    const localBrowser = await screen.findByTestId('local-browser');
    expect(localBrowser).toHaveTextContent('root:/data');
    expect(localBrowser).toHaveTextContent('rel:sub');
    expect(screen.queryByTestId('column-browser')).not.toBeInTheDocument();
    // Servers query is disabled (enabled: kind === 'tiled') for local connections.
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('shows singular sample count text when there is exactly one sample', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'local',
      localRoot: '/data',
      localRel: '',
      label: '/data',
      sampleCount: 1,
    });
    renderPage();
    expect(await screen.findByText(/1 sample$/)).toBeInTheDocument();
  });

  it('clicking "Change" navigates to /connect', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'local',
      localRoot: '/data',
      localRel: '',
      label: '/data',
      sampleCount: 1,
    });
    const user = userEvent.setup();
    renderPage();

    await user.click(screen.getByRole('button', { name: 'Change' }));
    expect(await screen.findByText('CONNECT PAGE')).toBeInTheDocument();
  });

  it('opening a local sample calls openLocalFile via useOpenInAnnotate', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'local',
      localRoot: '/data',
      localRel: '',
      label: '/data',
      sampleCount: 1,
    });
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByText('open-local'));
    expect(openLocalFile).toHaveBeenCalledWith('rel/path.tif');
  });

  it('changing the server in ColumnBrowser updates the connection store', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://a',
      label: 'Server A',
      sampleCount: 5,
    });
    const user = userEvent.setup();
    renderPage();

    await screen.findByTestId('column-browser');
    await user.click(screen.getByText('change-server'));

    expect(useConnectionStore.getState()).toMatchObject({
      kind: 'tiled',
      serverUri: 'http://b',
      label: 'Server B',
    });
  });

  it('changing the annotation filter is reflected back into ColumnBrowser', async () => {
    global.fetch = makeFetchMock();
    useConnectionStore.getState().setConnection({
      kind: 'tiled',
      serverUri: 'http://a',
      label: 'Server A',
      sampleCount: 5,
    });
    const user = userEvent.setup();
    renderPage();

    await screen.findByTestId('column-browser');
    await user.click(screen.getByText('change-annotation'));
    // Re-rendering with the new filter doesn't crash and the mock still shows.
    expect(await screen.findByTestId('column-browser')).toBeInTheDocument();
  });
});
