import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MeasurementPanel from './index';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';

const SK = 'local:sample.tif';

beforeEach(() => {
  useAnnotationStore.getState().reset();
  useToolStore.setState({ selectedShapeIds: [] });
  useDatasetStore.setState({
    meta: {
      width: 100, height: 100, nSlices: 1, dtype: 'uint8', isRgb: false, valueRange: [0, 255],
    },
    currentSlice: 0,
  } as any);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ pixel_count: 25, min: 10, max: 200, mean: 105.5, std: 12.25 }),
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('MeasurementPanel', () => {
  it('shows a hint when nothing is selected', () => {
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('Select one or more regions to measure.')).toBeInTheDocument();
  });

  it('shows a hint when there is no dataset meta', () => {
    useDatasetStore.setState({ meta: null } as any);
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('Select one or more regions to measure.')).toBeInTheDocument();
  });

  it('shows geometry stats (region count, area, centroid, bbox) in px units without calibration', async () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('Regions')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText(/px²$/)).toBeInTheDocument();
    expect(screen.getByText('Centroid')).toBeInTheDocument();
    expect(screen.getByText('Bounds')).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });

  it('only measures shapes selected on the current slice', () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
    });
    useAnnotationStore.getState().addShape(SK, 1, {
      id: 's2', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
    });
    useToolStore.setState({ selectedShapeIds: ['s1', 's2'] });
    // currentSlice is 0, so only s1 should count.
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  it('shows perimeter as — for a brush shape (undefined for that kind)', () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 'b1', classId: 1, kind: 'brush', strokes: [{ points: [0, 0, 5, 5], radius: 3, mode: 'paint' }],
    });
    useToolStore.setState({ selectedShapeIds: ['b1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('Perimeter')).toBeInTheDocument();
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('shows a computed perimeter for a rectangle', () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 5,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    // perimeter = 2*(10+5) = 30
    expect(screen.getByText('30.0 px')).toBeInTheDocument();
  });

  it('applies pixel-size calibration to area/perimeter/length units', async () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 5,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    const user = userEvent.setup();
    render(<MeasurementPanel sourceKey={SK} />);
    const pixelSizeInput = screen.getByPlaceholderText('—');
    await user.type(pixelSizeInput, '2');
    // perimeter 30px * 2 = 60, unit default µm
    expect(await screen.findByText(/60\.00 µm$/)).toBeInTheDocument();
  });

  it('ignores an invalid (zero/negative/non-numeric) pixel size and stays in px units', async () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 5,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    const user = userEvent.setup();
    render(<MeasurementPanel sourceKey={SK} />);
    const pixelSizeInput = screen.getByPlaceholderText('—');
    await user.type(pixelSizeInput, '0');
    expect(screen.getByText('30.0 px')).toBeInTheDocument();
  });

  it('fetches intensity stats for the selection and renders them', async () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    expect(screen.getByText('Measuring…')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/105\.5/)).toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText('10 / 200')).toBeInTheDocument();
    expect(screen.getByText('25')).toBeInTheDocument();
    const [url, init] = (fetch as any).mock.calls[0];
    expect(url).toContain('/api/measure?source_key=');
    expect(init.method).toBe('POST');
    const body = JSON.parse(init.body);
    expect(body.slice_index).toBe(0);
  });

  it('formats integer intensity values plainly and non-integers to 4 sig figs', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ pixel_count: 4, min: 0, max: 255, mean: 100, std: 12.3456 }),
    });
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    await waitFor(() => expect(screen.getByText(/100 ±/)).toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText('100 ± 12.35')).toBeInTheDocument();
    expect(screen.getByText('0 / 255')).toBeInTheDocument();
  });

  it('shows — for intensity when the request fails', async () => {
    (fetch as any).mockResolvedValue({ ok: false, json: async () => ({}) });
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    await waitFor(() => expect(screen.queryByText('Measuring…')).not.toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('shows — for intensity when the request throws', async () => {
    (fetch as any).mockRejectedValue(new Error('network down'));
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2,
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    render(<MeasurementPanel sourceKey={SK} />);
    await waitFor(() => expect(screen.queryByText('Measuring…')).not.toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('does not fetch intensity when sourceKey is null', () => {
    useAnnotationStore.getState().addShape(SK, 0, {
      id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2,
    });
    render(<MeasurementPanel sourceKey={null} />);
    expect(fetch).not.toHaveBeenCalled();
    expect(screen.getByText('Select one or more regions to measure.')).toBeInTheDocument();
  });
});
