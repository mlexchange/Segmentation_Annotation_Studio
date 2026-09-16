import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import InsightsModal from './InsightsModal';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useClassStore } from '@/stores/classStore';

const SOURCE = 'local:sample.tif';

const baseMeta = {
  nSlices: 4,
  height: 100,
  width: 100,
  dtype: 'uint8',
  isRgb: false,
  valueRange: [0, 255] as [number, number],
};

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useClassStore.setState({ classes: [] });
  useDatasetStore.setState({ meta: null, kind: null, source: null, serverUri: null } as any);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('InsightsModal', () => {
  it('shows a placeholder when there is no open sample', () => {
    render(<InsightsModal sourceKey={null} onClose={vi.fn()} onFocus={vi.fn()} />);
    expect(screen.getByText(/Open a sample to see its dataset health/)).toBeInTheDocument();
  });

  it('shows a loading indicator, then resolves to stats', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    expect(screen.getByText(/Analyzing sample/)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Annotated slices')).toBeInTheDocument();
    });
  });

  it('shows coverage tiles and "No classes defined" when there are no classes', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText('Annotated slices'));
    expect(screen.getByText('No classes defined.')).toBeInTheDocument();
  });

  it('shows "No issues found" when all slices are covered and clean', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    useAnnotationStore.setState({
      negativeSlices: { [SOURCE]: ['0', '1', '2', '3'] },
    });
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText('Annotated slices'));
    expect(screen.getByText('No issues found on this sample.')).toBeInTheDocument();
  });

  it('renders class balance rows with shape count and area', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    useClassStore.setState({
      classes: [{ classId: 1, label: 'Pore', color: '#ff0000', isVisible: true }],
    });
    useAnnotationStore.setState({
      byImage: {
        [SOURCE]: {
          '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 20, h: 20 } as any],
        },
      },
    });
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText('Pore'));
    expect(screen.getByText(/1 · /)).toBeInTheDocument();
  });

  it('flags a sliver shape and clicking it calls onFocus with the slice/bbox and closes the modal', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    useClassStore.setState({
      classes: [{ classId: 1, label: 'Pore', color: '#ff0000', isVisible: true }],
    });
    useAnnotationStore.setState({
      byImage: {
        [SOURCE]: {
          '2': [{ id: 's1', classId: 1, kind: 'rectangle', x: 5, y: 5, w: 1, h: 1 } as any],
        },
      },
    });
    const onFocus = vi.fn();
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<InsightsModal sourceKey={SOURCE} onClose={onClose} onFocus={onFocus} />);
    await waitFor(() => screen.getByText(/Tiny regions/));
    const flagButton = screen.getByText(/Tiny Pore region/).closest('button')!;
    await user.click(flagButton);
    expect(onFocus).toHaveBeenCalledWith(2, { x: 5, y: 5, w: 1, h: 1 });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('flags a cross-class overlap between two classes on the same slice', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    useClassStore.setState({
      classes: [
        { classId: 1, label: 'Pore', color: '#ff0000', isVisible: true },
        { classId: 2, label: 'Grain', color: '#00ff00', isVisible: true },
      ],
    });
    useAnnotationStore.setState({
      byImage: {
        [SOURCE]: {
          '0': [
            { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 40, h: 40 } as any,
            { id: 's2', classId: 2, kind: 'rectangle', x: 10, y: 10, w: 40, h: 40 } as any,
          ],
        },
      },
    });
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText(/Cross-class overlaps/));
    expect(screen.getByText(/Pore and Grain overlap/)).toBeInTheDocument();
  });

  it('shows session totals from all annotated samples', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    useAnnotationStore.setState({
      byImage: {
        [SOURCE]: { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 } as any] },
        'local:other.tif': { '0': [{ id: 's2', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 } as any] },
      },
    });
    render(<InsightsModal sourceKey={SOURCE} onClose={vi.fn()} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText('Samples this session'));
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('2 shapes')).toBeInTheDocument();
  });

  it('calls onClose when the header close button is clicked', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<InsightsModal sourceKey={SOURCE} onClose={onClose} onFocus={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: '' }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose when clicking the backdrop but not when clicking inside the dialog', async () => {
    useDatasetStore.setState({ meta: baseMeta } as any);
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<InsightsModal sourceKey={SOURCE} onClose={onClose} onFocus={vi.fn()} />);
    await waitFor(() => screen.getByText('Dataset Insights'));
    await user.click(screen.getByText('Dataset Insights'));
    expect(onClose).not.toHaveBeenCalled();
    const backdrop = screen.getByText('Dataset Insights').closest('.fixed')!;
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
