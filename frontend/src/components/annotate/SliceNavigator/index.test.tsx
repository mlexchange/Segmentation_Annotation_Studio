import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SliceNavigator from './index';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';

const SOURCE_KEY = 'local:sample.tif';

function setLoadedDataset(overrides: Partial<ReturnType<typeof useDatasetStore.getState>> = {}) {
  useDatasetStore.setState({
    meta: {
      nSlices: 10, height: 32, width: 32, dtype: 'uint8', isRgb: false, valueRange: [0, 255],
    },
    currentSlice: 3,
    source: 'sample.tif',
    kind: 'local',
    serverUri: null,
    ...overrides,
  } as any);
}

beforeEach(() => {
  useAnnotationStore.getState().reset();
});

afterEach(() => {
  cleanup();
});

describe('SliceNavigator', () => {
  it('shows a placeholder message when no dataset is loaded', () => {
    useDatasetStore.setState({ meta: null, source: null, kind: null } as any);
    render(<SliceNavigator />);
    expect(screen.getByText('No dataset loaded.')).toBeInTheDocument();
  });

  it('shows the current slice / total in the header', () => {
    setLoadedDataset();
    render(<SliceNavigator />);
    expect(screen.getByText('Slice 4 / 10')).toBeInTheDocument();
  });

  it('prev/next buttons step the slice and clamp at the ends', async () => {
    setLoadedDataset({ currentSlice: 0 } as any);
    const user = userEvent.setup();
    render(<SliceNavigator />);
    expect(screen.getByLabelText('Previous slice')).toBeDisabled();

    await user.click(screen.getByLabelText('Next slice'));
    expect(useDatasetStore.getState().currentSlice).toBe(1);
  });

  it('next is disabled on the last slice', () => {
    setLoadedDataset({ currentSlice: 9 } as any);
    render(<SliceNavigator />);
    expect(screen.getByLabelText('Next slice')).toBeDisabled();
  });

  it('shows a jump-to-annotated dropdown only when slices are annotated', () => {
    setLoadedDataset();
    render(<SliceNavigator />);
    expect(screen.queryByLabelText('Jump to annotated slice')).not.toBeInTheDocument();

    cleanup();
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE_KEY, 5, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    render(<SliceNavigator />);
    expect(screen.getByLabelText('Jump to annotated slice')).toBeInTheDocument();
  });

  it('marks the current slice as a negative example', async () => {
    setLoadedDataset();
    const user = userEvent.setup();
    render(<SliceNavigator />);
    const toggle = screen.getByRole('button', { name: /Mark as negative/ });
    await user.click(toggle);
    expect(useAnnotationStore.getState().negativeSlices[SOURCE_KEY]).toContain('3');
    expect(await screen.findByRole('button', { name: /Negative example/ })).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows a coarse-level warning when a non-finest pyramid level is open', () => {
    setLoadedDataset({
      meta: {
        nSlices: 10, height: 32, width: 32, dtype: 'uint8', isRgb: false, valueRange: [0, 255],
        levelKey: 'scale1', levelIndex: 1, levelWidth: 16, levelNSlices: 5,
      },
    } as any);
    render(<SliceNavigator />);
    expect(screen.getByText(/Viewing scale1/)).toBeInTheDocument();
  });
});
