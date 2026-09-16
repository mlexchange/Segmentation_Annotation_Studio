import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { Stage } from 'react-konva';
import OverlaysLayer, { type OverlaysLayerProps } from './OverlaysLayer';
import type { ManifoldPoint } from '@/lib/featureManifold';

/**
 * Shallow smoke tests: jsdom has no real 2D canvas, so Konva (which every
 * Stage/Layer/shape ultimately talks to) needs `HTMLCanvasElement.getContext`
 * stubbed before anything renders — otherwise Konva's own canvas setup throws.
 * The stub is a Proxy over a handful of explicitly-typed methods/properties
 * (fillStyle, drawImage, save/restore, path building, gradients, …) that falls
 * back to a `vi.fn()` for anything unlisted, so an internal Konva call this
 * suite didn't anticipate returns a harmless mock instead of `undefined is not
 * a function`.
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
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function baseProps(overrides: Partial<OverlaysLayerProps> = {}): OverlaysLayerProps {
  return {
    width: 256,
    height: 256,
    showFeatures: false,
    featureChannelUrl: null,
    showProba: false,
    probaOverlayUrl: null,
    probaOpacity: 0.5,
    showPredictions: false,
    clfCommitUrl: null,
    clfStatusUrl: null,
    predictionsOpacity: 0.5,
    classColorById: new Map(),
    predictionClassVisible: {},
    showPredictionMulti: true,
    showPredictionAbstain: true,
    showManifold: false,
    manifoldHeatmapUrl: null,
    manifoldHeatmapOpacity: 0.45,
    manifoldShowHeatmap: true,
    manifoldMarkers: [],
    manifoldShowMarkers: true,
    manifoldBoxSize: 64,
    ...overrides,
  };
}

/** Every Konva shape/image ultimately needs a Stage ancestor. */
function renderInStage(props: OverlaysLayerProps) {
  return render(
    <Stage width={props.width} height={props.height}>
      <OverlaysLayer {...props} />
    </Stage>,
  );
}

describe('OverlaysLayer', () => {
  it('renders no Konva layer (and no canvas) when every overlay is off', () => {
    const { container } = renderInStage(baseProps());
    // OverlaysLayer renders nothing with every flag off — no <Layer> means no
    // canvas gets created at all, which is a real, assertable fact here.
    expect(container.querySelectorAll('canvas').length).toBe(0);
  });

  it('renders without throwing when every overlay flag is on but no data has loaded yet', () => {
    // urls are set but jsdom's <img>/Image never fires onload, so the feature/
    // proba/prediction overlays stay un-rendered — this only proves the pending
    // state doesn't crash. The manifold Layer, unlike those, is gated on
    // `showManifold` alone (its heatmap/marker children are conditional inside
    // it), so it mounts as one empty canvas even before anything decodes.
    const { container } = renderInStage(
      baseProps({
        showFeatures: true,
        featureChannelUrl: 'blob:feature',
        showProba: true,
        probaOverlayUrl: 'blob:proba',
        showPredictions: true,
        clfCommitUrl: 'blob:commit',
        clfStatusUrl: 'blob:status',
        showManifold: true,
        manifoldHeatmapUrl: 'blob:manifold',
        manifoldShowMarkers: false,
      }),
    );
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('renders manifold marker rects immediately (they need no image decode)', () => {
    const { container } = renderInStage(
      baseProps({
        showManifold: true,
        manifoldShowHeatmap: false,
        manifoldShowMarkers: true,
        manifoldMarkers: [
          { x: 10, y: 10, cluster: 0 },
          { x: 50, y: 60, cluster: 1 },
        ] satisfies ManifoldPoint[],
      }),
    );
    // The markers are drawn on a non-listening Layer, which Konva backs with
    // exactly one scene canvas (no hit canvas since nothing needs hit-testing).
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('adding an active layer increases the number of canvases Konva maintains', () => {
    const off = renderInStage(baseProps());
    const offCount = off.container.querySelectorAll('canvas').length;
    cleanup();

    const on = renderInStage(
      baseProps({
        showManifold: true,
        manifoldShowMarkers: true,
        manifoldMarkers: [{ x: 1, y: 1, cluster: 0 }] satisfies ManifoldPoint[],
      }),
    );
    const onCount = on.container.querySelectorAll('canvas').length;
    expect(onCount).toBeGreaterThan(offCount);
  });

  it('does not throw across width/height prop changes (re-render)', () => {
    const { rerender, container } = renderInStage(baseProps({ width: 100, height: 100 }));
    rerender(
      <Stage width={300} height={200}>
        <OverlaysLayer {...baseProps({ width: 300, height: 200, showManifold: true })} />
      </Stage>,
    );
    expect(container.querySelectorAll('canvas').length).toBeGreaterThan(0);
  });
});
