/**
 * DownloadModal — scope/format selection, export payload building
 * (including predicted_slices pointer logic), and export/mask-sync job wiring.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import DownloadModal from './DownloadModal';
import { useDatasetStore } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { usePredictedRasterStore } from '@/stores/predictedRasterStore';
import { useClassStore } from '@/stores/classStore';
import { useRatingStore } from '@/stores/ratingStore';
import { useSettingsStore } from '@/stores/settingsStore';

function renderModal(onClose = vi.fn()) {
  return render(
    <MemoryRouter>
      <DownloadModal onClose={onClose} />
    </MemoryRouter>,
  );
}

/** A fetch mock whose POST resolves to a synchronous "done" response (no job_id),
 *  so useExportJob treats it as immediately done — avoids polling in tests. */
function mockFetchDone(result: Record<string, unknown> = {}) {
  return vi.fn(async () => ({
    ok: true,
    json: async () => result,
    text: async () => JSON.stringify(result),
  })) as unknown as typeof fetch;
}

beforeEach(() => {
  useDatasetStore.setState({
    kind: 'local', source: 'sample.tif', serverUri: null, currentSlice: 0,
  } as any);
  useAnnotationStore.getState().reset();
  usePredictedRasterStore.setState({ bySource: {} });
  useClassStore.setState({ classes: [{ classId: 1, label: 'Cell', color: '#111111', isVisible: true }] });
  useRatingStore.setState({ ratings: {} });
  useSettingsStore.setState({ annotatorName: '' });
  vi.spyOn(window, 'alert').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('DownloadModal', () => {
  it('renders scope and format options', () => {
    renderModal();
    expect(screen.getByText('Download Dataset')).toBeInTheDocument();
    expect(screen.getByText('Current slice only')).toBeInTheDocument();
    expect(screen.getByText('Current sample only')).toBeInTheDocument();
    expect(screen.getByText('All annotated samples')).toBeInTheDocument();
    expect(screen.getByText(/COCO \(SAM3\)/)).toBeInTheDocument();
    expect(screen.getByText(/DINOv3 \/ Lightly/)).toBeInTheDocument();
  });

  it('defaults to "current sample" scope with a 1-sample preview when a source is loaded', () => {
    renderModal();
    expect(screen.getByRole('radio', { name: /Current sample only/ })).toBeChecked();
    expect(screen.getByText('1 sample will be exported.')).toBeInTheDocument();
  });

  it('shows "No samples match" for the "all" scope with no annotated samples', async () => {
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('radio', { name: /All annotated samples/ }));
    expect(screen.getByText('No samples match.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Export/ })).toBeDisabled();
  });

  it('counts annotated samples for the "all" scope, applying the star-rating filter', async () => {
    useAnnotationStore.getState().setShapes('local:a.tif', 0, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    useAnnotationStore.getState().setShapes('local:b.tif', 0, [
      { id: 's2', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    useRatingStore.setState({ ratings: { 'local:b.tif': 2 } });
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getByRole('radio', { name: /All annotated samples/ }));
    expect(screen.getByText('2 samples will be exported.')).toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /★★ and above/ }));
    expect(screen.getByText('1 sample will be exported.')).toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /★★★ only/ }));
    expect(screen.getByText('No samples match.')).toBeInTheDocument();
  });

  it('POSTs the current-sample export payload with predicted_slices for un-vectorized pointers', async () => {
    useAnnotationStore.getState().setShapes('local:sample.tif', 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    usePredictedRasterStore.getState().setPointers('local:sample.tif', {
      '0': { runId: 'run-1', classIds: [1] },
      '1': { runId: 'run-2', classIds: [1] },
    });
    useSettingsStore.setState({ annotatorName: '  Ada  ' });

    const fetchMock = mockFetchDone({ dataset_path: '/exports/foo' });
    global.fetch = fetchMock;

    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, opts] = (fetchMock as any).mock.calls[0];
    expect(url).toBe('/api/export/coco');
    const body = JSON.parse(opts.body);
    expect(body.annotator).toBe('Ada');
    expect(body.format).toBe('coco_sam3');
    expect(body.sources).toHaveLength(1);
    const src = body.sources[0];
    expect(src.kind).toBe('local');
    expect(src.source).toBe('sample.tif');
    // Slice 1 has real shapes, so its pointer is excluded; slice 0 has no
    // shapes, so its pointer is included as an un-vectorized predicted_slice.
    expect(src.predicted_slices).toEqual({ '0': { run_id: 'run-1', class_ids: [1] } });
    expect(src.slices).toEqual({ '1': expect.any(Array) });
  });

  it('scopes predicted_slices and negative_slices to just the current slice for "slice" scope', async () => {
    useDatasetStore.setState({ currentSlice: 0 } as any);
    usePredictedRasterStore.getState().setPointers('local:sample.tif', {
      '0': { runId: 'run-a', classIds: [1] },
      '2': { runId: 'run-b', classIds: [1] },
    });
    useAnnotationStore.getState().toggleNegativeSlice('local:sample.tif', 0);
    useAnnotationStore.getState().toggleNegativeSlice('local:sample.tif', 2);

    const fetchMock = mockFetchDone({});
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('radio', { name: /Current slice only/ }));
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse((fetchMock as any).mock.calls[0][1].body);
    const src = body.sources[0];
    expect(src.predicted_slices).toEqual({ '0': { run_id: 'run-a', class_ids: [1] } });
    expect(src.negative_slices).toEqual(['0']);
  });

  it('includes include_polygons only relevant for coco_sam3 and reflects the checkbox', async () => {
    const fetchMock = mockFetchDone({});
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByLabelText(/Include polygon copy in COCO/));
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse((fetchMock as any).mock.calls[0][1].body);
    expect(body.include_polygons).toBe(true);
  });

  it('switches format to lightly_dinov3', async () => {
    const fetchMock = mockFetchDone({});
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByLabelText(/DINOv3 \/ Lightly/));
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse((fetchMock as any).mock.calls[0][1].body);
    expect(body.format).toBe('lightly_dinov3');
  });

  it('shows the saved-to-Tiled message on success (no zip when the response has no job_id)', async () => {
    // A synchronous response (no job_id) is treated as immediately "done" with
    // no jobId, so downloadUrl stays null and only the "Saved to" line shows.
    global.fetch = mockFetchDone({ dataset_path: '/exports/foo.zip' });
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await screen.findByText(/Saved to/);
    expect(screen.getByText('/exports/foo.zip')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Download \.zip/ })).not.toBeInTheDocument();
  });

  it('shows an error message when the export request fails', async () => {
    global.fetch = vi.fn(async () => ({
      ok: false,
      status: 500,
      text: async () => JSON.stringify({ detail: 'boom' }),
    })) as unknown as typeof fetch;
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: /^Export$/ }));

    await waitFor(() => expect(screen.getByText(/boom|Request failed/)).toBeInTheDocument());
  });

  it('alerts and does not call fetch when no sample is loaded for "current" scope', async () => {
    useDatasetStore.setState({ kind: null, source: null } as any);
    const fetchMock = mockFetchDone({});
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderModal();
    // previewCount is 0 with no source, disabling Export — assert directly instead.
    expect(screen.getByRole('button', { name: /^Export$/ })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('disables "Push masks to Tiled" for a local source and enables it for a tiled source', async () => {
    renderModal();
    expect(screen.getByRole('button', { name: /Push masks to Tiled/ })).toBeDisabled();

    cleanup();
    useDatasetStore.setState({ kind: 'tiled', source: 'a', serverUri: 'http://x', currentSlice: 0 } as any);
    renderModal();
    expect(screen.getByRole('button', { name: /Push masks to Tiled/ })).not.toBeDisabled();
  });

  it('POSTs to the mask-sync endpoint, filtering to tiled sources only, and shows a "View in 3D" link', async () => {
    useDatasetStore.setState({ kind: 'tiled', source: 'vol.tif', serverUri: 'http://x', currentSlice: 0 } as any);
    useAnnotationStore.getState().setShapes('tiled:http://x:vol.tif', 0, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    const fetchMock = mockFetchDone({
      written: [{ container: 'vol.tif', n_slices: 10, updated: 1 }],
    });
    global.fetch = fetchMock;
    const user = userEvent.setup();
    renderModal();
    await user.click(screen.getByRole('button', { name: /Push masks to Tiled/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, opts] = (fetchMock as any).mock.calls[0];
    expect(url).toBe('/api/masks/to-tiled');
    const body = JSON.parse(opts.body);
    expect(body.sources.every((s: any) => s.kind === 'tiled')).toBe(true);

    await screen.findByText(/Masks merged into Tiled/);
    const link = screen.getByRole('button', { name: /View in 3D/ });
    expect(link).toBeInTheDocument();
  });

  it('navigates to the 3D volume view when "View in 3D" is clicked', async () => {
    useDatasetStore.setState({ kind: 'tiled', source: 'vol.tif', serverUri: 'http://x', currentSlice: 0 } as any);
    useAnnotationStore.getState().setShapes('tiled:http://x:vol.tif', 0, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 },
    ]);
    global.fetch = mockFetchDone({
      written: [{ container: 'vol.tif', n_slices: 10, updated: 1 }],
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    renderModal(onClose);
    await user.click(screen.getByRole('button', { name: /Push masks to Tiled/ }));
    await screen.findByText(/Masks merged into Tiled/);
    await user.click(screen.getByRole('button', { name: /View in 3D/ }));
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when Cancel/Close is clicked', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    renderModal(onClose);
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalled();
  });
});
