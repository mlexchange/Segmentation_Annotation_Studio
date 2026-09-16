/**
 * LocalSampleBrowser — flat local-folder sample list: loading/error states,
 * annotation + star filters, rating persistence, joinPath, and Annotate wiring.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import LocalSampleBrowser, { joinPath } from './LocalSampleBrowser';
import { useRatingStore } from '@/stores/ratingStore';
import { buildSourceKey } from '@/lib/sourceKey';

const initialRatingState = useRatingStore.getState();

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
    if (url.pathname === '/api/annotations/drafts') return jsonResponse([]);
    return jsonResponse({}, false);
  });
}

const SAMPLES = [
  { name: 'a.tif', path: 'a.tif' },
  { name: 'b.tif', path: 'sub/b.tif' },
];

function renderBrowser(
  overrides: Partial<React.ComponentProps<typeof LocalSampleBrowser>> = {},
) {
  const props = {
    root: '/data',
    rel: '',
    onOpenInAnnotate: vi.fn(),
    annotationFilter: 'all' as const,
    onAnnotationFilterChange: vi.fn(),
    ...overrides,
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <LocalSampleBrowser {...props} />
      </QueryClientProvider>,
    ),
    props,
  };
}

beforeEach(() => {
  useRatingStore.setState(initialRatingState, true);
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('joinPath', () => {
  it('returns root unchanged when rel is empty', () => {
    expect(joinPath('/data', '')).toBe('/data');
  });

  it('joins root and rel, normalising duplicate slashes', () => {
    expect(joinPath('/data/', '/sub/file.tif')).toBe('/data/sub/file.tif');
    expect(joinPath('/data', 'sub/file.tif')).toBe('/data/sub/file.tif');
  });
});

describe('LocalSampleBrowser', () => {
  it('shows a loading state before samples arrive', async () => {
    let resolveFetch!: (v: Response) => void;
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/local/samples')) {
        return new Promise((resolve) => {
          resolveFetch = resolve;
        });
      }
      return Promise.resolve(jsonResponse([]));
    }) as unknown as typeof fetch;

    renderBrowser();
    expect(screen.getByText('Loading samples…')).toBeInTheDocument();

    resolveFetch(jsonResponse({ items: [], total: 0 }));
    await waitFor(() => expect(screen.queryByText('Loading samples…')).not.toBeInTheDocument());
  });

  it('shows an error message when the samples request fails', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse('disk unavailable', false),
    });
    renderBrowser();
    expect(await screen.findByText('disk unavailable')).toBeInTheDocument();
  });

  it('lists samples with a total count and the current folder path', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
    });
    renderBrowser({ rel: 'sub' });

    expect(await screen.findByText('a.tif')).toBeInTheDocument();
    expect(screen.getByText('b.tif')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('/data/sub')).toBeInTheDocument();
  });

  it('shows an empty-folder message when there are no samples', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: [], total: 0 }),
    });
    renderBrowser();
    expect(await screen.findByText('No image files found in this folder.')).toBeInTheDocument();
  });

  it('shows an annotated badge for samples with drafts, based on their source key', async () => {
    const sk = buildSourceKey('local', joinPath('/data', 'a.tif'));
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
      '/api/annotations/drafts': () => jsonResponse([{ source_key: sk, has_annotations: true }]),
    });
    renderBrowser();

    await screen.findByText('a.tif');
    const row = screen.getByText('a.tif').closest('.group') as HTMLElement;
    expect(row).toHaveTextContent('annotated');
    const otherRow = screen.getByText('b.tif').closest('.group') as HTMLElement;
    expect(otherRow).not.toHaveTextContent('annotated');
  });

  it('filters by the annotation-status select via the onAnnotationFilterChange callback', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
    });
    const onAnnotationFilterChange = vi.fn();
    const user = userEvent.setup();
    renderBrowser({ onAnnotationFilterChange });

    await screen.findByText('a.tif');
    await user.selectOptions(screen.getByRole('combobox'), 'annotated');
    expect(onAnnotationFilterChange).toHaveBeenCalledWith('annotated');
  });

  it('unannotated filter hides samples with drafts', async () => {
    const sk = buildSourceKey('local', joinPath('/data', 'a.tif'));
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
      '/api/annotations/drafts': () => jsonResponse([{ source_key: sk, has_annotations: true }]),
    });
    renderBrowser({ annotationFilter: 'unannotated' });

    await screen.findByText('b.tif');
    expect(screen.queryByText('a.tif')).not.toBeInTheDocument();
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
  });

  it('star filter hides samples below the chosen minimum rating', async () => {
    const skA = buildSourceKey('local', joinPath('/data', 'a.tif'));
    useRatingStore.setState({ ratings: { [skA]: 2 } });
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
    });
    const user = userEvent.setup();
    renderBrowser();

    await screen.findByText('a.tif');
    await user.click(screen.getByText('★★'));
    expect(screen.getByText('a.tif')).toBeInTheDocument();
    expect(screen.queryByText('b.tif')).not.toBeInTheDocument();

    await user.click(screen.getByText('All'));
    expect(screen.getByText('b.tif')).toBeInTheDocument();
  });

  it('clicking a star rates the sample and persists it in the rating store', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
    });
    const user = userEvent.setup();
    renderBrowser();

    await screen.findByText('a.tif');
    const row = screen.getByText('a.tif').closest('.group') as HTMLElement;
    await user.click(within(row).getByLabelText('2 stars'));

    const sk = buildSourceKey('local', joinPath('/data', 'a.tif'));
    expect(useRatingStore.getState().ratings[sk]).toBe(2);
  });

  it('clicking Annotate calls onOpenInAnnotate with the absolute path', async () => {
    global.fetch = makeFetchMock({
      '/api/local/samples': () => jsonResponse({ items: SAMPLES, total: 2 }),
    });
    const onOpenInAnnotate = vi.fn();
    const user = userEvent.setup();
    renderBrowser({ onOpenInAnnotate });

    await screen.findByText('b.tif');
    const row = screen.getByText('b.tif').closest('.group') as HTMLElement;
    await user.click(within(row).getByRole('button', { name: /Annotate/ }));

    expect(onOpenInAnnotate).toHaveBeenCalledWith('/data/sub/b.tif');
  });
});
