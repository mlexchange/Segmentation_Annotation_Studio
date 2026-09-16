import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SlicesColumn from './SlicesColumn';
import { useRatingStore } from '@/stores/ratingStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { buildSourceKey } from '@/lib/sourceKey';
import type { BrowseItem } from './hooks/useBrowseData';

const initialRatingState = useRatingStore.getState();

function makeItem(overrides: Partial<BrowseItem> = {}): BrowseItem {
  return {
    path: 'ds/a',
    sample: 'ds_0001',
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
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const dataset = makeItem({ path: 'ds', sample: 'ds' });

const baseProps = {
  dataset,
  slices: [] as BrowseItem[],
  loading: false,
  selectedItem: null,
  onSelect: vi.fn(),
  onOpenInAnnotate: vi.fn(),
  onClose: vi.fn(),
  width: 200,
  serverUri: 'http://server',
};

describe('SlicesColumn', () => {
  it('shows a loading state', () => {
    renderWithClient(<SlicesColumn {...baseProps} loading />);
    expect(screen.getByText('Loading slices…')).toBeInTheDocument();
  });

  it('shows an empty state when there are no slices', () => {
    renderWithClient(<SlicesColumn {...baseProps} slices={[]} />);
    expect(screen.getByText('No slices found.')).toBeInTheDocument();
  });

  it('renders the slice count badge and the "open all" button count', () => {
    const slices = [makeItem({ path: 'ds/1', sample: 'ds_0001' }), makeItem({ path: 'ds/2', sample: 'ds_0002' })];
    renderWithClient(<SlicesColumn {...baseProps} slices={slices} />);
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText(/Open all 2 slices as a volume/)).toBeInTheDocument();
  });

  it('labels a slice by its image_number metadata when present', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'ds_weird', metadata: { image_number: 7 } });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    expect(screen.getByText('Slice 7')).toBeInTheDocument();
  });

  it('labels a slice by its trailing suffix when it shares the dataset prefix', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0003' });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    expect(screen.getByText('Slice 0003')).toBeInTheDocument();
  });

  it('falls back to the raw sample name when no image_number and no shared prefix', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'totally_different' });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    expect(screen.getByText('totally_different')).toBeInTheDocument();
  });

  it('clicking the back button calls onClose', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<SlicesColumn {...baseProps} onClose={onClose} />);
    await user.click(screen.getByTitle('Back to samples'));
    expect(onClose).toHaveBeenCalled();
  });

  it('clicking "open all as volume" calls onOpenInAnnotate with the dataset', async () => {
    const onOpenInAnnotate = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<SlicesColumn {...baseProps} onOpenInAnnotate={onOpenInAnnotate} />);
    await user.click(screen.getByText(/Open all/));
    expect(onOpenInAnnotate).toHaveBeenCalledWith(dataset);
  });

  it('clicking a slice row selects it, clicking again clears the selection', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0001' });
    const { rerender } = renderWithClient(
      <SlicesColumn {...baseProps} slices={[slice]} onSelect={onSelect} selectedItem={null} />,
    );
    await user.click(screen.getByText('Slice 0001'));
    expect(onSelect).toHaveBeenCalledWith(slice);

    onSelect.mockClear();
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <SlicesColumn {...baseProps} slices={[slice]} onSelect={onSelect} selectedItem={slice} />
      </QueryClientProvider>,
    );
    await user.click(screen.getByText('Slice 0001'));
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it('clicking the per-slice open-in-annotate button calls onOpenInAnnotate with that slice, not the dataset', async () => {
    const onOpenInAnnotate = vi.fn();
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0001' });
    renderWithClient(
      <SlicesColumn {...baseProps} slices={[slice]} onOpenInAnnotate={onOpenInAnnotate} onSelect={onSelect} />,
    );
    await user.click(screen.getByTitle('Open this slice in Annotate'));
    expect(onOpenInAnnotate).toHaveBeenCalledWith(slice);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('shows the annotated badge when the slice is flagged annotated', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0001', metadata: { studio_annotated: 'yes' } });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    expect(screen.getByText('annotated')).toBeInTheDocument();
  });

  it('does not show the annotated badge for an unannotated slice', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0001' });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    expect(screen.queryByText('annotated')).not.toBeInTheDocument();
  });

  it('renders a star rating control reflecting the stored rating for the slice', () => {
    const slice = makeItem({ path: 'ds/1', sample: 'ds_0001' });
    const sourceKey = buildSourceKey('tiled', slice.path, baseProps.serverUri);
    useRatingStore.setState({ ratings: { [sourceKey]: 2 } });
    renderWithClient(<SlicesColumn {...baseProps} slices={[slice]} />);
    // StarRating renders 3 star buttons per row.
    expect(screen.getAllByRole('button', { name: /star/ })).toHaveLength(3);
  });
});
