import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MaskToolsPanel from './index';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';

const SOURCE = 'local:sample.tif';

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#f00', isVisible: true }] });
  useDatasetStore.setState({
    meta: {
      nSlices: 5, height: 32, width: 32, dtype: 'uint8', isRgb: false, valueRange: [0, 255],
    },
    currentSlice: 0,
  } as any);
});

afterEach(() => {
  cleanup();
});

describe('MaskToolsPanel', () => {
  it('renders nothing for a single-slice (non-volume) dataset', () => {
    useDatasetStore.setState({
      meta: { nSlices: 1, height: 32, width: 32, dtype: 'uint8', isRgb: false, valueRange: [0, 255] },
    } as any);
    const { container } = render(<MaskToolsPanel sourceKey={SOURCE} activeClassId={1} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows the active class name and a disabled copy button with no shapes here', () => {
    render(<MaskToolsPanel sourceKey={SOURCE} activeClassId={1} />);
    expect(screen.getByText(/Acts on ‘Cell’/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Copy/ })).toBeDisabled();
  });

  it('enables the copy button once the active class has a shape on the current slice', () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 4, h: 4 },
    ]);
    render(<MaskToolsPanel sourceKey={SOURCE} activeClassId={1} />);
    expect(screen.getByRole('button', { name: /Copy/ })).toBeEnabled();
  });

  it('copying reports success and advances the shape to the next slice', async () => {
    useAnnotationStore.getState().replaceClassShapesOnSlice(SOURCE, 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 4, h: 4 },
    ]);
    const user = userEvent.setup();
    render(<MaskToolsPanel sourceKey={SOURCE} activeClassId={1} />);
    await user.click(screen.getByRole('button', { name: /Copy/ }));
    expect(await screen.findByText(/Copied ‘Cell’ to slice 2/)).toBeInTheDocument();
    expect(useAnnotationStore.getState().byImage[SOURCE]?.['1']).toHaveLength(1);
  });

  it('disabled with no sourceKey', () => {
    render(<MaskToolsPanel sourceKey={null} activeClassId={1} />);
    expect(screen.getByRole('button', { name: /Copy/ })).toBeDisabled();
  });
});
