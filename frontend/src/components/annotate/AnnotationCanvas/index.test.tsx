import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AnnotationCanvas from './index';
import { useDatasetStore, type ImageMeta } from '@/stores/datasetStore';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useToolStore } from '@/stores/toolStore';
import { useClassStore, type AnnotationClass } from '@/stores/classStore';
import { useLayerVisibilityStore } from '@/stores/layerVisibilityStore';
import { useClipboardStore } from '@/stores/clipboardStore';

/**
 * Shallow smoke tests for the 3636-line react-konva Stage that is the Annotate
 * page's whole editing surface. These deliberately do NOT drive tool
 * interactions (drawing, panning, brush strokes, SAM, livewire, …) — only that
 * the component mounts and re-renders across its main branches (no dataset
 * loaded / dataset + shapes loaded / different active tools / preview mode)
 * without throwing.
 *
 * jsdom has no real 2D canvas, so Konva (which every Stage/Layer/shape
 * ultimately talks to) needs `HTMLCanvasElement.getContext` stubbed before
 * anything renders — otherwise Konva's own canvas setup throws. The stub is a
 * Proxy over explicitly-typed methods/properties (the ones this component's
 * own code calls directly — willReadFrequently getImageData/putImageData for
 * the histogram and threshold-overlay repaint — plus everything Konva itself
 * needs) that falls back to a `vi.fn()` for anything unlisted.
 */
function createMockContext(canvas: HTMLCanvasElement) {
  const base: Record<string, unknown> = {
    canvas,
    fillStyle: '#000000',
    strokeStyle: '#000000',
    lineWidth: 1,
    lineCap: 'butt',
    lineJoin: 'miter',
    globalAlpha: 1,
    globalCompositeOperation: 'source-over',
    font: '10px sans-serif',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    imageSmoothingEnabled: true,
    filter: 'none',
    fillRect: vi.fn(),
    clearRect: vi.fn(),
    strokeRect: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4), width: 1, height: 1 })),
    createImageData: vi.fn((w: number, h: number) => ({
      data: new Uint8ClampedArray(Math.max(1, w) * Math.max(1, h) * 4),
      width: w,
      height: h,
    })),
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    transform: vi.fn(),
    setTransform: vi.fn(),
    resetTransform: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    bezierCurveTo: vi.fn(),
    quadraticCurveTo: vi.fn(),
    arc: vi.fn(),
    arcTo: vi.fn(),
    ellipse: vi.fn(),
    rect: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    clip: vi.fn(),
    isPointInPath: vi.fn(() => false),
    measureText: vi.fn(() => ({ width: 0 })),
    fillText: vi.fn(),
    strokeText: vi.fn(),
    createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
    createRadialGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
    createPattern: vi.fn(() => ({})),
    setLineDash: vi.fn(),
    getLineDash: vi.fn(() => []),
  };
  return new Proxy(base, {
    get(target, prop) {
      if (prop in target) return (target as Record<string, unknown>)[prop as string];
      return vi.fn();
    },
    set(target, prop, value) {
      (target as Record<string, unknown>)[prop as string] = value;
      return true;
    },
  });
}

// Snapshot the zustand stores' initial state once at module load so every test
// can restore a clean slate, per this session's established convention (these
// four stores have no built-in `reset`; annotationStore's own `reset()` is used
// where available).
const initialToolState = useToolStore.getState();
const initialLayerVisibilityState = useLayerVisibilityStore.getState();
const initialClipboardState = useClipboardStore.getState();

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = vi.fn(function (this: HTMLCanvasElement) {
    return createMockContext(this) as unknown as CanvasRenderingContext2D;
  }) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.toDataURL = vi.fn(() => 'data:,');
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  // useImageSlice's queryFn calls fetch when a dataset is loaded; keep it from
  // ever hitting the network (jsdom's <img>.src load never resolves either way,
  // so this only avoids an unhandled real request, never actual image data).
  global.fetch = vi.fn(() => Promise.reject(new Error('network disabled in tests')));

  useDatasetStore.getState().reset();
  useAnnotationStore.getState().reset();
  useClassStore.setState({ classes: [] });
  useToolStore.setState(initialToolState, true);
  useLayerVisibilityStore.setState(initialLayerVisibilityState, true);
  useClipboardStore.setState(initialClipboardState, true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const CLASS_A: AnnotationClass = { classId: 1, label: 'A', color: '#ff0000', isVisible: true };
const CLASS_B: AnnotationClass = { classId: 2, label: 'B', color: '#00ff00', isVisible: true };

const META: ImageMeta = {
  nSlices: 5,
  height: 256,
  width: 256,
  dtype: 'uint8',
  isRgb: false,
  valueRange: [0, 255],
};

function baseProps(overrides: Partial<React.ComponentProps<typeof AnnotationCanvas>> = {}) {
  return {
    brightness: 0,
    contrast: 0,
    levelsLo: 0,
    levelsHi: 255,
    activeClassId: null,
    activeBrushShapeId: null,
    onNewBrushInstance: vi.fn(),
    ...overrides,
  };
}

/** Loads a dataset (source/kind/meta) into datasetStore, as ConnectPage/BrowsePage would. */
function loadDataset() {
  useDatasetStore.getState().setDataset('local', 'sample.tif', null, META);
}

function renderCanvas(overrides: Partial<React.ComponentProps<typeof AnnotationCanvas>> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AnnotationCanvas {...baseProps(overrides)} />
    </QueryClientProvider>,
  );
}

describe('AnnotationCanvas', () => {
  it('renders without crashing when no dataset is loaded', () => {
    const { container } = renderCanvas();
    // The image layer always mounts (even empty), so at least one canvas exists.
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });

  it('renders without crashing once a dataset (meta) is loaded, with no shapes', () => {
    loadDataset();
    const { container } = renderCanvas();
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });

  it('renders without crashing with committed shapes present on the current slice', () => {
    loadDataset();
    useClassStore.setState({ classes: [CLASS_A, CLASS_B] });
    useAnnotationStore.getState().replaceClassShapesOnSlice('local:sample.tif', 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20 },
      { id: 's2', classId: 2, kind: 'ellipse', cx: 100, cy: 100, rx: 15, ry: 10 },
    ]);
    const { container } = renderCanvas({ activeClassId: 1 });
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });

  it('renders without crashing for each drawing tool (pan/select/polygon/rectangle/ellipse/brush/eraser)', () => {
    loadDataset();
    useClassStore.setState({ classes: [CLASS_A] });
    const tools = ['pan', 'select', 'polygon', 'magnetic', 'rectangle', 'ellipse', 'brush', 'eraser', 'threshold', 'sampler', 'fill'] as const;
    for (const tool of tools) {
      useToolStore.setState({ tool });
      expect(() => {
        const { unmount } = renderCanvas({ activeClassId: 1 });
        unmount();
      }, `tool=${tool}`).not.toThrow();
    }
  });

  it('renders the select tool with a selection without crashing (Transformer + toolbar overlay)', () => {
    loadDataset();
    useClassStore.setState({ classes: [CLASS_A] });
    useAnnotationStore.getState().replaceClassShapesOnSlice('local:sample.tif', 0, 1, [
      { id: 's1', classId: 1, kind: 'rectangle', x: 10, y: 10, w: 20, h: 20 },
    ]);
    useToolStore.setState({ tool: 'select', selectedShapeIds: ['s1'] });
    expect(() => renderCanvas({ activeClassId: 1 })).not.toThrow();
  });

  it('renders in read-only preview mode (previewShapes/previewClasses) without crashing', () => {
    loadDataset();
    const { container } = renderCanvas({
      previewShapes: [{ id: 'p1', classId: 9, kind: 'rectangle', x: 0, y: 0, w: 5, h: 5 }],
      previewClasses: [{ classId: 9, label: 'Preview', color: '#123456', isVisible: true }],
    });
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });

  it('renders with iPred overlay props set (proba/predictions/manifold) without crashing', () => {
    loadDataset();
    const { container } = renderCanvas({
      probaOverlayUrl: 'blob:proba',
      clfCommitUrl: 'blob:commit',
      clfStatusUrl: 'blob:status',
      predictionClassColorById: new Map([[1, '#ff0000']]),
      manifoldHeatmapUrl: 'blob:manifold',
      manifoldMarkers: [{ x: 5, y: 5, cluster: 0 }],
    });
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });

  it('renders across a display-adjustment prop sweep (brightness/contrast/levels/colormap/gamma/blur/upscale) without crashing', () => {
    loadDataset();
    expect(() =>
      renderCanvas({
        brightness: 30,
        contrast: -20,
        levelsLo: 10,
        levelsHi: 240,
        colormap: 'viridis',
        gamma: 1.4,
        clahe: true,
        sharpen: true,
        blur: 2,
        upscale: 2,
      }),
    ).not.toThrow();
  });

  it('calls onHistogram only after a slice image has actually decoded (never synchronously with no dataset)', () => {
    const onHistogram = vi.fn();
    renderCanvas({ onHistogram });
    // No dataset loaded => no image => the histogram effect bails out early.
    expect(onHistogram).not.toHaveBeenCalled();
  });

  it('re-renders across a focusRegion prop change without crashing', () => {
    loadDataset();
    const { rerender, container } = renderCanvas({ focusRegion: null });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    rerender(
      <QueryClientProvider client={queryClient}>
        <AnnotationCanvas
          {...baseProps({ focusRegion: { x: 5, y: 5, w: 10, h: 10, nonce: 1 } })}
        />
      </QueryClientProvider>,
    );
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });
});
