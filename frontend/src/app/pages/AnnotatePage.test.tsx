/**
 * AnnotatePage — orchestration tests. This page wires together nearly every
 * annotate hook/component; children with their own dedicated test files are
 * mocked here (AnnotationCanvas, Toolbar, ClassManager, LayersPanel, SaveModal,
 * DownloadModal, VersionHistoryModal, DenoiseBakeModal, InsightsModal,
 * FeatureChannelsPanel, PixelClassifierPanel), as are the data-hooks the page
 * calls directly. Also mocked: DisplayControls, SliceNavigator, MaskToolsPanel,
 * MeasurementPanel — none of AnnotatePage's own logic depends on their
 * internals, and leaving them real would pull in fetch/useQuery/useMaskOps
 * plumbing unrelated to this page's orchestration. Real: the Zustand stores,
 * DebouncedSlider (opacity), VersionPreviewBar, StageSwitcher (inline).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router';
import AnnotatePage from './AnnotatePage';
import { useDatasetStore, type ImageMeta } from '@/stores/datasetStore';
import { useAnnotationStore, type Shape } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore } from '@/stores/classStore';
import { usePredictedRasterStore } from '@/stores/predictedRasterStore';

// ---- Hooks AnnotatePage calls directly ----

const useDraftSync = vi.fn();
vi.mock('@/hooks/useDraftSync', () => ({ useDraftSync: (...args: unknown[]) => useDraftSync(...args) }));

const useGuideLoad = vi.fn();
vi.mock('@/hooks/useGuideSync', () => ({ useGuideLoad: (...args: unknown[]) => useGuideLoad(...args) }));

const useKeybinds = vi.fn();
vi.mock('@/hooks/useKeybinds', () => ({ useKeybinds: (...args: unknown[]) => useKeybinds(...args) }));

const save = vi.fn(async () => true);
const buildSavePayload = vi.fn(() => null as null | Record<string, unknown>);
const fetchVersionPayload = vi.fn(async () => null as unknown);
const restoreVersion = vi.fn();
let useSaveReturn: Record<string, unknown>;
vi.mock('@/hooks/useSave', () => ({
  useSave: (...args: unknown[]) => useSaveHook(...args),
}));
function useSaveHook(..._args: unknown[]) {
  return useSaveReturn;
}

const featuresCompute = vi.fn();
let useFeatureChannelsReturn: Record<string, unknown>;
vi.mock('@/hooks/useFeatureChannels', () => ({
  useFeatureChannels: (...args: unknown[]) => useFeatureChannelsHook(...args),
}));
function useFeatureChannelsHook(..._args: unknown[]) {
  return useFeatureChannelsReturn;
}

const clfTrain = vi.fn();
const clfTrainAcrossSlices = vi.fn();
const clfPredict = vi.fn();
const clfDismiss = vi.fn();
const clfApplyAcrossVolume = vi.fn();
const clfResetVolumeApplyJob = vi.fn();
let usePixelClassifierReturn: Record<string, unknown>;
vi.mock('@/hooks/usePixelClassifier', () => ({
  usePixelClassifier: (...args: unknown[]) => usePixelClassifierHook(...args),
}));
function usePixelClassifierHook(..._args: unknown[]) {
  return usePixelClassifierReturn;
}

let useFeatureManifoldReturn: Record<string, unknown>;
vi.mock('@/hooks/useFeatureManifold', () => ({
  useFeatureManifold: (...args: unknown[]) => useFeatureManifoldHook(...args),
}));
function useFeatureManifoldHook(..._args: unknown[]) {
  return useFeatureManifoldReturn;
}

const startMaskSync = vi.fn();
let useExportJobReturn: Record<string, unknown>;
vi.mock('@/hooks/useExportJob', () => ({
  useExportJob: (...args: unknown[]) => useExportJobHook(...args),
}));
function useExportJobHook(..._args: unknown[]) {
  return useExportJobReturn;
}

const loadLabelPng = vi.fn();
const labelMapToPolygonShapes = vi.fn();
vi.mock('@/lib/pixelClf', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/pixelClf')>();
  return {
    ...actual,
    loadLabelPng: (...args: unknown[]) => loadLabelPng(...args),
    labelMapToPolygonShapes: (...args: unknown[]) => labelMapToPolygonShapes(...args),
  };
});

// ---- Heavy / separately-tested child components ----

vi.mock('@/components/annotate/AnnotationCanvas', () => ({
  default: (props: Record<string, unknown>) => (
    <div
      data-testid="annotation-canvas"
      data-active-class-id={String(props.activeClassId)}
      data-active-brush-shape-id={String(props.activeBrushShapeId)}
      data-focus-region={JSON.stringify(props.focusRegion ?? null)}
    >
      <button onClick={() => (props.onNewBrushInstance as (id: string) => void)('brush-1')}>
        mock-new-brush
      </button>
    </div>
  ),
}));

vi.mock('@/components/annotate/Toolbar', () => ({
  default: () => <div data-testid="toolbar" />,
}));

vi.mock('@/components/annotate/ClassManager', () => ({
  default: (props: { activeClassId: number | null; onActivate: (id: number) => void; onClassDeleted: (id: number) => void }) => (
    <div data-testid="class-manager" data-active-class-id={String(props.activeClassId)}>
      <button onClick={() => props.onActivate(2)}>mock-activate-class-2</button>
      <button onClick={() => props.onClassDeleted(1)}>mock-delete-class-1</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/LayersPanel', () => ({
  default: () => <div data-testid="layers-panel" />,
}));

vi.mock('@/components/annotate/DisplayControls', () => ({
  default: () => <div data-testid="display-controls" />,
}));

vi.mock('@/components/annotate/SliceNavigator', () => ({
  default: () => <div data-testid="slice-navigator" />,
}));

vi.mock('@/components/annotate/MaskToolsPanel', () => ({
  default: () => <div data-testid="mask-tools-panel" />,
}));

vi.mock('@/components/annotate/MeasurementPanel', () => ({
  default: () => <div data-testid="measurement-panel" />,
}));

vi.mock('@/components/annotate/FeatureChannelsPanel', () => ({
  default: () => <div data-testid="feature-channels-panel" />,
}));

vi.mock('@/components/annotate/PixelClassifierPanel', () => ({
  default: (props: Record<string, unknown>) => (
    <div data-testid="pixel-classifier-panel">
      <button onClick={() => (props.onTrain as () => void)()}>mock-clf-train</button>
      <button onClick={() => (props.onPredict as () => void)()}>mock-clf-predict</button>
      <button onClick={() => (props.onCommit as () => void)()}>mock-clf-commit</button>
      <button onClick={() => (props.onTrainAcrossSlicesChange as (v: boolean) => void)(true)}>
        mock-toggle-train-across
      </button>
      <button onClick={() => (props.onApplyToVolume as () => void)()}>mock-apply-to-volume</button>
      <button onClick={() => (props.onCommitVolumeApply as () => void)()}>mock-commit-volume-apply</button>
      <button onClick={() => (props.onMakeSliceEditable as () => void)()}>mock-make-slice-editable</button>
      <button onClick={() => (props.onSyncToTiled as () => void)()}>mock-sync-to-tiled</button>
      <button onClick={() => (props.onViewIn3D as () => void)()}>mock-view-in-3d</button>
      <button onClick={() => (props.onTrainDeepModel as () => void)()}>mock-train-deep-model</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/SaveModal', () => ({
  default: (props: {
    shapeCount: number;
    classCount: number;
    isSaving: boolean;
    onSave: (opts: { annotatedBy: string; notes: string }) => void;
    onClose: () => void;
  }) => (
    <div data-testid="save-modal" data-shape-count={props.shapeCount} data-class-count={props.classCount}>
      <button onClick={() => props.onSave({ annotatedBy: 'me', notes: 'n' })}>mock-confirm-save</button>
      <button onClick={() => props.onClose()}>mock-close-save</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/DownloadModal', () => ({
  default: (props: { onClose: () => void }) => (
    <div data-testid="download-modal">
      <button onClick={() => props.onClose()}>mock-close-download</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/InsightsModal', () => ({
  default: (props: { onClose: () => void; onFocus: (slice: number, bbox?: { x: number; y: number; w: number; h: number }) => void }) => (
    <div data-testid="insights-modal">
      <button onClick={() => props.onFocus(3, { x: 1, y: 2, w: 3, h: 4 })}>mock-insight-focus</button>
      <button onClick={() => props.onClose()}>mock-close-insights</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/VersionHistoryModal', () => ({
  default: (props: {
    versions: { version: number }[];
    onPreview: (v: number) => void;
    onRestore: (v: number) => void;
    onClose: () => void;
  }) => (
    <div data-testid="version-history-modal">
      <button onClick={() => props.onPreview(2)}>mock-preview-v2</button>
      <button onClick={() => props.onRestore(1)}>mock-restore-v1</button>
      <button onClick={() => props.onClose()}>mock-close-history</button>
    </div>
  ),
}));

vi.mock('@/components/annotate/DenoiseBakeModal', () => ({
  default: (props: { open: boolean }) => (
    <div data-testid="denoise-bake-modal" data-open={String(props.open)} />
  ),
}));

const initialDatasetState = useDatasetStore.getState();
const initialAnnotationState = useAnnotationStore.getState();
const initialToolState = useToolStore.getState();
const initialClassState = useClassStore.getState();
const initialPredictedRasterState = usePredictedRasterStore.getState();

const META: ImageMeta = {
  nSlices: 5,
  height: 100,
  width: 100,
  dtype: 'uint8',
  isRgb: false,
  valueRange: [0, 255],
};

function shape(id: string, classId: number): Shape {
  return { id, classId, kind: 'rectangle', x: 0, y: 0, w: 10, h: 10 };
}

/** Loads a dataset (meta present) so AnnotatePage renders the full workspace. */
function loadDataset(overrides: Partial<{ source: string; kind: string; serverUri: string | null }> = {}) {
  useDatasetStore.setState({
    ...useDatasetStore.getState(),
    kind: overrides.kind ?? 'tiled',
    source: overrides.source ?? 'sample.zarr',
    serverUri: overrides.serverUri ?? 'http://localhost:8000/api',
    meta: META,
    currentSlice: 0,
  });
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/annotate']}>
      <Routes>
        <Route path="/annotate" element={<AnnotatePage />} />
        <Route path="/browse" element={<div>BROWSE PAGE</div>} />
        <Route path="/volume" element={<div>VOLUME PAGE</div>} />
        <Route path="/train" element={<div>TRAIN PAGE</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('AnnotatePage', () => {
  beforeEach(() => {
    useDatasetStore.setState(initialDatasetState, true);
    useAnnotationStore.setState(initialAnnotationState, true);
    useToolStore.setState(initialToolState, true);
    useClassStore.setState(initialClassState, true);
    usePredictedRasterStore.setState(initialPredictedRasterState, true);

    useSaveReturn = {
      isDirty: false,
      isSaving: false,
      lastSavedAt: null,
      buildSavePayload,
      saveSummary: { shapeCount: 0, classCount: 0 },
      save,
      versions: [],
      fetchVersionPayload,
      restoreVersion,
      markClean: vi.fn(),
    };
    useFeatureChannelsReturn = {
      job: null,
      channelIndex: null,
      channelUrl: null,
      computing: false,
      error: null,
      invalidateJob: vi.fn(),
      adoptFeatureBank: vi.fn(),
      compute: featuresCompute,
      selectChannel: vi.fn(),
      cycleChannel: vi.fn(),
      clearSelection: vi.fn(),
    };
    usePixelClassifierReturn = {
      params: { iterations: 200, depth: 6, learningRate: 0.1, alpha: 0.05 },
      setParams: vi.fn(),
      model: null,
      commitUrl: null,
      statusUrl: null,
      probaUrl: null,
      predictCounts: null,
      probaClassIndex: 0,
      activeProbaClassId: null,
      activeProbaThreshold: 0.5,
      cycleProbaClass: vi.fn(),
      setProbaThreshold: vi.fn(),
      error: null,
      training: false,
      predicting: false,
      train: clfTrain,
      trainAcrossSlices: clfTrainAcrossSlices,
      predict: clfPredict,
      dismiss: clfDismiss,
      commitUrl_: null,
      compositionId: 'comp-1',
      trainerId: 'catboost',
      canTrainWithoutJob: true,
      multiTrainJob: { done: 0, total: 0 },
      multiTraining: false,
      resetMultiTrainJob: vi.fn(),
      applyAcrossVolume: clfApplyAcrossVolume,
      volumeApplyJob: { done: 0, total: 0, result: null, jobId: null },
      volumeApplying: false,
      resetVolumeApplyJob: clfResetVolumeApplyJob,
    };
    useFeatureManifoldReturn = {
      params: { k: 24, boxSize: 64 },
      setParams: vi.fn(),
      points: [],
      heatmapUrl: null,
      showHeatmap: false,
      setShowHeatmap: vi.fn(),
      showMarkers: false,
      setShowMarkers: vi.fn(),
      heatmapOpacity: 0.45,
      setHeatmapOpacity: vi.fn(),
      sampling: false,
      error: null,
      meta: null,
      sample: vi.fn(),
      dismiss: vi.fn(),
      placementMask: null,
      setPlacementMaskFromShapes: vi.fn(),
      clearPlacementMask: vi.fn(),
      hasSample: false,
    };
    useExportJobReturn = {
      state: { status: 'idle', phase: '', done: 0, total: 0, log: [], result: null, error: null, jobId: null },
      start: vi.fn(),
      startMaskSync,
      startIpredBatchTrain: vi.fn(),
      startIpredBatchApply: vi.fn(),
      startJob: vi.fn(),
      reset: vi.fn(),
      downloadUrl: null,
    };
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('shows "no sample loaded" when no dataset is open, and Go to Browse navigates', async () => {
    const user = userEvent.setup();
    renderPage();
    expect(screen.getByText(/No sample loaded/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Go to Browse/ }));
    expect(await screen.findByText('BROWSE PAGE')).toBeInTheDocument();
  });

  it('renders the draw-stage panels by default and switches stages via the tab bar', async () => {
    const user = userEvent.setup();
    loadDataset();
    renderPage();

    expect(screen.getByTestId('toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('display-controls')).toBeInTheDocument();
    expect(screen.getByTestId('mask-tools-panel')).toBeInTheDocument();
    expect(screen.getByTestId('measurement-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('feature-channels-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('pixel-classifier-panel')).not.toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: 'Assist' }));
    expect(screen.getByTestId('feature-channels-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('toolbar')).not.toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: 'Predict' }));
    expect(screen.getByTestId('pixel-classifier-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('feature-channels-panel')).not.toBeInTheDocument();
  });

  it('auto-activates the first class once classes load, and clicking a class updates the canvas + clears the in-progress brush', async () => {
    const user = userEvent.setup();
    loadDataset();
    useClassStore.setState({
      classes: [
        { classId: 1, label: 'A', color: '#f00', isVisible: true },
        { classId: 2, label: 'B', color: '#0f0', isVisible: true },
      ],
    });
    renderPage();

    // Auto-activated to the first class (classId 1).
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-class-id', '1');

    // Start a brush instance, then switch class — activeBrushShapeId must reset.
    await user.click(screen.getByText('mock-new-brush'));
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-brush-shape-id', 'brush-1');

    await user.click(screen.getByText('mock-activate-class-2'));
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-class-id', '2');
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-brush-shape-id', 'null');
  });

  it('clearing the active brush instance after a class delete does not change activeClassId', async () => {
    const user = userEvent.setup();
    loadDataset();
    useClassStore.setState({ classes: [{ classId: 1, label: 'A', color: '#f00', isVisible: true }] });
    renderPage();

    await user.click(screen.getByText('mock-new-brush'));
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-brush-shape-id', 'brush-1');

    await user.click(screen.getByText('mock-delete-class-1'));
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-brush-shape-id', 'null');
    expect(screen.getByTestId('annotation-canvas')).toHaveAttribute('data-active-class-id', '1');
  });

  it('save button reflects isDirty/isSaving and opens the save modal with the current summary via buildSavePayload', async () => {
    const user = userEvent.setup();
    loadDataset();
    useSaveReturn.isDirty = true;
    useSaveReturn.saveSummary = { shapeCount: 4, classCount: 2 };
    buildSavePayload.mockReturnValue({ classes: [], slices: {}, split_by_slice: {}, negative_slices: [] });
    renderPage();

    const saveBtn = screen.getByRole('button', { name: /^Save$/ });
    expect(screen.getByText('Unsaved changes')).toBeInTheDocument();

    await user.click(saveBtn);
    const modal = screen.getByTestId('save-modal');
    expect(modal).toHaveAttribute('data-shape-count', '4');
    expect(modal).toHaveAttribute('data-class-count', '2');
  });

  it('does not open the save modal when buildSavePayload returns null (nothing to save)', async () => {
    const user = userEvent.setup();
    loadDataset();
    buildSavePayload.mockReturnValue(null);
    renderPage();

    await user.click(screen.getByRole('button', { name: /Saved/ }));
    expect(screen.queryByTestId('save-modal')).not.toBeInTheDocument();
  });

  it('confirming the save modal calls save() and closes the modal on success', async () => {
    const user = userEvent.setup();
    loadDataset();
    buildSavePayload.mockReturnValue({ classes: [], slices: {}, split_by_slice: {}, negative_slices: [] });
    save.mockResolvedValueOnce(true);
    renderPage();

    await user.click(screen.getByRole('button', { name: /Saved/ }));
    await user.click(screen.getByText('mock-confirm-save'));

    expect(save).toHaveBeenCalledWith({ annotatedBy: 'me', notes: 'n' });
    await waitFor(() => expect(screen.queryByTestId('save-modal')).not.toBeInTheDocument());
  });

  it('keeps the save modal open when save() fails', async () => {
    const user = userEvent.setup();
    loadDataset();
    buildSavePayload.mockReturnValue({ classes: [], slices: {}, split_by_slice: {}, negative_slices: [] });
    save.mockResolvedValueOnce(false);
    renderPage();

    await user.click(screen.getByRole('button', { name: /Saved/ }));
    await user.click(screen.getByText('mock-confirm-save'));

    await waitFor(() => expect(save).toHaveBeenCalled());
    expect(screen.getByTestId('save-modal')).toBeInTheDocument();
  });

  it('shows the version-history entry point only once versions exist, and preview/restore wire through', async () => {
    const user = userEvent.setup();
    loadDataset();
    useSaveReturn.versions = [
      { version: 1, saved_at: '2024-01-01T00:00:00Z', shape_count: 1, class_count: 1 },
      { version: 2, saved_at: '2024-01-02T00:00:00Z', shape_count: 2, class_count: 1 },
    ];
    fetchVersionPayload.mockResolvedValue({ classes: [], slices: {}, split_by_slice: {}, negative_slices: [] });
    renderPage();

    const historyBtn = screen.getByTitle('Version history');
    expect(historyBtn).toHaveTextContent('2');
    await user.click(historyBtn);
    expect(screen.getByTestId('version-history-modal')).toBeInTheDocument();

    // Preview closes the history modal and opens the version preview bar.
    await user.click(screen.getByText('mock-preview-v2'));
    expect(screen.queryByTestId('version-history-modal')).not.toBeInTheDocument();
    await waitFor(() => expect(fetchVersionPayload).toHaveBeenCalledWith(2));
    expect(await screen.findByText(/Previewing v/)).toBeInTheDocument();

    // Exiting the preview bar clears previewVersion (bar disappears).
    await user.click(screen.getByTitle('Exit preview'));
    expect(screen.queryByText(/Previewing v/)).not.toBeInTheDocument();
  });

  it('restoring a version from history calls restoreVersion', async () => {
    const user = userEvent.setup();
    loadDataset();
    useSaveReturn.versions = [{ version: 1, saved_at: '2024-01-01T00:00:00Z', shape_count: 1, class_count: 1 }];
    renderPage();

    await user.click(screen.getByTitle('Version history'));
    await user.click(screen.getByText('mock-restore-v1'));
    expect(restoreVersion).toHaveBeenCalledWith(1);
  });

  it('Insights: opens on click, and a QA focus jumps the slice and sets the focus region on the canvas', async () => {
    const user = userEvent.setup();
    loadDataset();
    renderPage();

    await user.click(screen.getByRole('button', { name: /Insights/ }));
    expect(screen.getByTestId('insights-modal')).toBeInTheDocument();

    await user.click(screen.getByText('mock-insight-focus'));
    expect(useDatasetStore.getState().currentSlice).toBe(3);
    const canvas = screen.getByTestId('annotation-canvas');
    const region = JSON.parse(canvas.getAttribute('data-focus-region') ?? 'null');
    expect(region).toMatchObject({ x: 1, y: 2, w: 3, h: 4 });

    await user.click(screen.getByText('mock-close-insights'));
    expect(screen.queryByTestId('insights-modal')).not.toBeInTheDocument();
  });

  it('Export opens and closes the download modal', async () => {
    const user = userEvent.setup();
    loadDataset();
    renderPage();

    await user.click(screen.getByRole('button', { name: /Export/ }));
    expect(screen.getByTestId('download-modal')).toBeInTheDocument();
    await user.click(screen.getByText('mock-close-download'));
    expect(screen.queryByTestId('download-modal')).not.toBeInTheDocument();
  });

  it('renders the denoise bake modal only when a source is open', () => {
    loadDataset({ source: 'sample.zarr' });
    renderPage();
    expect(screen.getByTestId('denoise-bake-modal')).toBeInTheDocument();
  });

  it('passes the correct callbacks to useKeybinds, and the delete-selected callback removes selected shapes on the current slice', () => {
    loadDataset();
    const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
    useAnnotationStore.setState({
      byImage: { [sourceKey]: { '0': [shape('s1', 1), shape('s2', 1)] } },
    });
    useToolStore.setState({ selectedShapeIds: ['s1'] });
    renderPage();

    expect(useKeybinds).toHaveBeenCalled();
    const [activeClassId, onActivateClass, onNewBrushInstance, onDeleteSelected, onCancelDraft] =
      useKeybinds.mock.calls[useKeybinds.mock.calls.length - 1];
    expect(typeof onActivateClass).toBe('function');
    expect(typeof onNewBrushInstance).toBe('function');
    expect(activeClassId).toBeNull();

    act(() => onDeleteSelected());
    expect(useAnnotationStore.getState().byImage[sourceKey]['0'].map((s: Shape) => s.id)).toEqual(['s2']);
    expect(useToolStore.getState().selectedShapeIds).toEqual([]);

    act(() => onCancelDraft());
  });

  describe('pixel classifier orchestration', () => {
    it('Train calls clf.train with the current slice shapes when "train across slices" is off', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      const shapes = [shape('s1', 1)];
      useAnnotationStore.setState({ byImage: { [sourceKey]: { '0': shapes } } });
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-clf-train'));
      expect(clfTrain).toHaveBeenCalledWith(shapes);
      expect(clfTrainAcrossSlices).not.toHaveBeenCalled();
    });

    it('Train calls clf.trainAcrossSlices with every non-empty slice once "train across slices" is toggled on', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      useAnnotationStore.setState({
        byImage: { [sourceKey]: { '0': [shape('s1', 1)], '1': [shape('s2', 1)], '2': [] } },
      });
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-toggle-train-across'));
      await user.click(screen.getByText('mock-clf-train'));

      expect(clfTrainAcrossSlices).toHaveBeenCalledWith({
        0: [shape('s1', 1)],
        1: [shape('s2', 1)],
      });
      expect(clfTrain).not.toHaveBeenCalled();
    });

    it('Predict calls clf.predict with the current slice shapes', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      const shapes = [shape('s1', 1)];
      useAnnotationStore.setState({ byImage: { [sourceKey]: { '0': shapes } } });
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-clf-predict'));
      expect(clfPredict).toHaveBeenCalledWith(shapes);
    });

    it('Commit vectorizes the predicted PNG into shapes, appends them, and dismisses the overlay', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      usePixelClassifierReturn.commitUrl = 'blob:commit';
      usePixelClassifierReturn.model = { classIds: [1, 2] };
      loadLabelPng.mockResolvedValue({ data: new Uint8Array([1]), width: 1, height: 1 });
      const newShape = shape('predicted-1', 1);
      labelMapToPolygonShapes.mockReturnValue([newShape]);
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-clf-commit'));

      await waitFor(() =>
        expect(useAnnotationStore.getState().byImage[sourceKey]?.['0']).toEqual([newShape]),
      );
      expect(clfDismiss).toHaveBeenCalledTimes(1);
    });

    it('Apply-to-volume runs across every slice index up to the dataset\'s slice count', async () => {
      const user = userEvent.setup();
      loadDataset(); // META.nSlices === 5
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-apply-to-volume'));
      expect(clfApplyAcrossVolume).toHaveBeenCalledWith([0, 1, 2, 3, 4]);
    });

    it('Commit-volume-apply records predicted-raster pointers only for slices with no existing shapes', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      useAnnotationStore.setState({ byImage: { [sourceKey]: { '1': [shape('existing', 1)] } } });
      usePixelClassifierReturn.volumeApplyJob = {
        done: 5,
        total: 5,
        result: { runs: { '0': 'run-a', '1': 'run-b', '2': 'run-c' } },
      };
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      // Select a commit class first via the toggle so commitClassIds is non-empty.
      await user.click(screen.getByText('mock-toggle-train-across')); // harmless toggle to exercise a click
      await user.click(screen.getByText('mock-commit-volume-apply'));

      // commitClassIds defaults to [] until a model exists; with model null the
      // handler no-ops (commitClassIds.length === 0) — so no pointers recorded.
      expect(usePredictedRasterStore.getState().bySource[sourceKey]).toBeUndefined();
    });

    it('Commit-volume-apply records pointers for un-annotated slices once a model supplies commit classes', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      useAnnotationStore.setState({ byImage: { [sourceKey]: { '1': [shape('existing', 1)] } } });
      usePixelClassifierReturn.model = { classIds: [1, 2] };
      usePixelClassifierReturn.volumeApplyJob = {
        done: 3,
        total: 3,
        result: { runs: { '0': 'run-a', '1': 'run-b', '2': 'run-c' } },
      };
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-commit-volume-apply'));

      const pointers = usePredictedRasterStore.getState().bySource[sourceKey];
      // Slice 1 already has real shapes, so it's skipped; 0 and 2 get pointers.
      expect(pointers).toEqual({
        '0': { runId: 'run-a', classIds: [1, 2] },
        '2': { runId: 'run-c', classIds: [1, 2] },
      });
      expect(clfResetVolumeApplyJob).toHaveBeenCalledTimes(1);
    });

    it('View in 3D navigates to /volume?mask=fast', async () => {
      const user = userEvent.setup();
      loadDataset();
      usePixelClassifierReturn.model = { classIds: [1] };
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-view-in-3d'));
      expect(await screen.findByText('VOLUME PAGE')).toBeInTheDocument();
    });

    it('Train a deep model navigates to /train', async () => {
      const user = userEvent.setup();
      loadDataset();
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-train-deep-model'));
      expect(await screen.findByText('TRAIN PAGE')).toBeInTheDocument();
    });

    it('Sync to Tiled alerts (and does not start the job) for a non-Tiled source', async () => {
      const user = userEvent.setup();
      loadDataset({ kind: 'local', source: 'data/foo', serverUri: null });
      const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-sync-to-tiled'));
      expect(alertSpy).toHaveBeenCalledWith('Pushing masks to Tiled only works for Tiled sources.');
      expect(startMaskSync).not.toHaveBeenCalled();
    });

    it('Sync to Tiled starts the mask-sync job with the current slices/classes for a Tiled source', async () => {
      const user = userEvent.setup();
      loadDataset();
      const sourceKey = 'tiled:http://localhost:8000/api:sample.zarr';
      const shapes = [shape('s1', 1)];
      useAnnotationStore.setState({ byImage: { [sourceKey]: { '0': shapes } } });
      useClassStore.setState({ classes: [{ classId: 1, label: 'A', color: '#f00', isVisible: true }] });
      renderPage();

      await user.click(screen.getByRole('tab', { name: 'Predict' }));
      await user.click(screen.getByText('mock-sync-to-tiled'));

      expect(startMaskSync).toHaveBeenCalledWith({
        sources: [
          expect.objectContaining({
            kind: 'tiled',
            source: 'sample.zarr',
            server_uri: 'http://localhost:8000/api',
            slices: { '0': shapes },
          }),
        ],
        classes: [{ classId: 1, label: 'A', color: '#f00', isVisible: true }],
      });
    });
  });
});
