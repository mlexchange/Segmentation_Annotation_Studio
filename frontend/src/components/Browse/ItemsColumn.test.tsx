import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ItemsColumn from './ItemsColumn';
import { useRatingStore } from '@/stores/ratingStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { buildSourceKey } from '@/lib/sourceKey';
import type { BrowseItem } from './hooks/useBrowseData';

const initialRatingState = useRatingStore.getState();

function makeItem(overrides: Partial<BrowseItem> = {}): BrowseItem {
  return {
    path: 'ds/a',
    sample: 'sample-a',
    metadata: {},
    ...overrides,
  };
}

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  useRatingStore.setState(initialRatingState, true);
  useAnnotationStore.getState().reset();
  window.localStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({ ok: true, json: async () => [] }),
  );
  // jsdom doesn't implement scrollIntoView.
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const baseProps = {
  items: [] as BrowseItem[],
  total: 0,
  loading: false,
  selectedItem: null,
  onSelect: vi.fn(),
  onOpenInAnnotate: vi.fn(),
  width: 220,
  serverUri: 'http://server',
  annotationFilter: 'all' as const,
};

describe('ItemsColumn', () => {
  it('shows a loading state', () => {
    renderWithClient(<ItemsColumn {...baseProps} loading />);
    expect(screen.getByText('Loading…')).toBeInTheDocument();
  });

  it('shows an empty state when there are no matching samples', () => {
    renderWithClient(<ItemsColumn {...baseProps} items={[]} total={0} />);
    expect(screen.getByText('No matching samples')).toBeInTheDocument();
  });

  it('renders items and the total count badge', () => {
    const items = [makeItem({ path: 'a', sample: 'a' }), makeItem({ path: 'b', sample: 'b' })];
    renderWithClient(<ItemsColumn {...baseProps} items={items} total={2} />);
    expect(screen.getByText('a')).toBeInTheDocument();
    expect(screen.getByText('b')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('clicking a row selects it, clicking again clears the selection', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const item = makeItem({ path: 'a', sample: 'a' });
    const { rerender } = renderWithClient(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ItemsColumn {...baseProps} items={[item]} onSelect={onSelect} selectedItem={null} />
      </QueryClientProvider>,
    );
    await user.click(screen.getByText('a'));
    expect(onSelect).toHaveBeenCalledWith(item);

    onSelect.mockClear();
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ItemsColumn {...baseProps} items={[item]} onSelect={onSelect} selectedItem={item} />
      </QueryClientProvider>,
    );
    await user.click(screen.getByText('a'));
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it('clicking the open-in-annotate button calls onOpenInAnnotate with the item and does not select it', async () => {
    const onOpenInAnnotate = vi.fn();
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const item = makeItem({ path: 'a', sample: 'a' });
    renderWithClient(
      <ItemsColumn {...baseProps} items={[item]} onOpenInAnnotate={onOpenInAnnotate} onSelect={onSelect} />,
    );
    await user.click(screen.getByTitle('Open in Annotate'));
    expect(onOpenInAnnotate).toHaveBeenCalledWith(item);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('shows a volume badge and caret for multi-slice items, not for single-slice items', () => {
    const volume = makeItem({ path: 'v', sample: 'vol', n_slices: 5 });
    const single = makeItem({ path: 's', sample: 'single', n_slices: 1 });
    renderWithClient(<ItemsColumn {...baseProps} items={[volume, single]} />);
    expect(screen.getByText('5')).toBeInTheDocument();
    expect(screen.getByTitle('5 slices — click to browse them')).toBeInTheDocument();
  });

  it('shows the annotated badge when metadata marks the item annotated', () => {
    const item = makeItem({ path: 'a', sample: 'a', metadata: { studio_annotated: 'yes' } });
    renderWithClient(<ItemsColumn {...baseProps} items={[item]} />);
    expect(screen.getByText('annotated')).toBeInTheDocument();
  });

  it('does not show the annotated badge for unannotated items', () => {
    const item = makeItem({ path: 'a', sample: 'a' });
    renderWithClient(<ItemsColumn {...baseProps} items={[item]} />);
    expect(screen.queryByText('annotated')).not.toBeInTheDocument();
  });

  it('filters out unannotated items when annotationFilter is "annotated"', () => {
    const annotated = makeItem({ path: 'a', sample: 'a', metadata: { studio_annotated: 'yes' } });
    const plain = makeItem({ path: 'b', sample: 'b' });
    renderWithClient(
      <ItemsColumn {...baseProps} items={[annotated, plain]} total={2} annotationFilter="annotated" />,
    );
    expect(screen.getByText('a')).toBeInTheDocument();
    expect(screen.queryByText('b')).not.toBeInTheDocument();
    // filtered count differs from total, so shows "1 / 2"
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
  });

  it('filters out annotated items when annotationFilter is "unannotated"', () => {
    const annotated = makeItem({ path: 'a', sample: 'a', metadata: { studio_annotated: 'yes' } });
    const plain = makeItem({ path: 'b', sample: 'b' });
    renderWithClient(
      <ItemsColumn {...baseProps} items={[annotated, plain]} total={2} annotationFilter="unannotated" />,
    );
    expect(screen.queryByText('a')).not.toBeInTheDocument();
    expect(screen.getByText('b')).toBeInTheDocument();
  });

  it('filters by minimum star rating', async () => {
    const item = makeItem({ path: 'a', sample: 'a' });
    const sourceKey = buildSourceKey('tiled', item.path, baseProps.serverUri);
    useRatingStore.setState({ ratings: { [sourceKey]: 1 } });
    const user = userEvent.setup();
    renderWithClient(<ItemsColumn {...baseProps} items={[item]} total={1} />);

    expect(screen.getByText('a')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '★★' }));
    expect(screen.queryByText('a')).not.toBeInTheDocument();
    expect(screen.getByText('No samples match the current filters.')).toBeInTheDocument();
  });

  it('shows the "no matches" filtered message distinctly from the plain empty state', () => {
    renderWithClient(<ItemsColumn {...baseProps} items={[]} total={0} annotationFilter="annotated" />);
    expect(screen.getByText('No samples match the current filters.')).toBeInTheDocument();
  });
});
