import { createRef } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { Stage } from 'react-konva';
import type Konva from 'konva';
import ShapesLayer, { type ShapesLayerProps } from './ShapesLayer';
import type { Shape } from '@/stores/annotationStore';
import type { AnnotationClass } from '@/stores/classStore';

/**
 * Shallow smoke tests: jsdom has no real 2D canvas, so Konva (which every
 * Stage/Layer/shape ultimately talks to) needs `HTMLCanvasElement.getContext`
 * stubbed before anything renders — otherwise Konva's own canvas setup throws.
 * The stub is a Proxy over explicitly-typed methods/properties (fillStyle,
 * drawImage, path building, `fill('evenodd')` used by the polygon-with-holes
 * sceneFunc, …) that falls back to a `vi.fn()` for anything unlisted, so a
 * Konva call this suite didn't anticipate returns a harmless mock instead of
 * throwing "undefined is not a function".
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

const classA: AnnotationClass = { classId: 1, label: 'A', color: '#ff0000', isVisible: true };
const classB: AnnotationClass = { classId: 2, label: 'B', color: '#00ff00', isVisible: true };
const hiddenClass: AnnotationClass = { classId: 3, label: 'Hidden', color: '#0000ff', isVisible: false };

const polygonShape: Shape = { id: 'poly-1', classId: 1, kind: 'polygon', points: [0, 0, 10, 0, 10, 10, 0, 10] };
const polygonWithHoles: Shape = {
  id: 'poly-2',
  classId: 1,
  kind: 'polygon',
  points: [0, 0, 20, 0, 20, 20, 0, 20],
  holes: [[5, 5, 15, 5, 15, 15, 5, 15]],
};
const rectShape: Shape = { id: 'rect-1', classId: 2, kind: 'rectangle', x: 1, y: 1, w: 5, h: 5 };
const ellipseShape: Shape = { id: 'ellipse-1', classId: 2, kind: 'ellipse', cx: 10, cy: 10, rx: 4, ry: 3 };
const brushShape: Shape = {
  id: 'brush-1',
  classId: 1,
  kind: 'brush',
  strokes: [
    { points: [0, 0, 5, 5, 10, 0], radius: 2, mode: 'paint' },
    { points: [2, 2, 6, 6], radius: 1, mode: 'erase' },
  ],
};
const predictedShape: Shape = { id: 'rect-predicted', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2, origin: 'predicted' };
const hiddenClassShape: Shape = { id: 'rect-hidden', classId: 3, kind: 'rectangle', x: 0, y: 0, w: 2, h: 2 };
const shapeWithErase: Shape = {
  id: 'rect-erased',
  classId: 2,
  kind: 'rectangle',
  x: 0,
  y: 0,
  w: 8,
  h: 8,
  erased: [{ points: [1, 1, 2, 2], radius: 1 }],
};

function baseProps(overrides: Partial<ShapesLayerProps> = {}): ShapesLayerProps {
  return {
    layerRef: createRef<Konva.Layer>(),
    shapes: [],
    classMap: new Map([[1, classA], [2, classB], [3, hiddenClass]]),
    fillOpacity: 0.5,
    scaleX: 1,
    selectedShapeIds: [],
    activeBrushShapeId: null,
    activeClassId: null,
    imageWidth: 256,
    imageHeight: 256,
    originVisible: { human: true, predicted: true },
    ...overrides,
  };
}

function renderInStage(props: ShapesLayerProps) {
  return render(
    <Stage width={256} height={256}>
      <ShapesLayer {...props} />
    </Stage>,
  );
}

describe('ShapesLayer', () => {
  it('renders an empty layer without crashing when there are no shapes', () => {
    const { container } = renderInStage(baseProps());
    // The Layer itself still mounts (it always exists so the parent's ref +
    // caching logic has something to grab), giving exactly one scene canvas.
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('renders without crashing across every shape kind (polygon, polygon-with-holes, rectangle, ellipse, brush)', () => {
    const { container } = renderInStage(
      baseProps({
        shapes: [polygonShape, polygonWithHoles, rectShape, ellipseShape, brushShape],
      }),
    );
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('renders a shape with erase carve-outs without crashing', () => {
    const { container } = renderInStage(baseProps({ shapes: [shapeWithErase] }));
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('filters out shapes belonging to a hidden class', () => {
    // Nothing to assert on the canvas pixels themselves (coarse mock), but the
    // filter runs in plain JS before any Konva node is built, so at minimum it
    // must not throw when the only shape present is on a hidden class.
    expect(() => renderInStage(baseProps({ shapes: [hiddenClassShape] }))).not.toThrow();
  });

  it('filters out shapes whose origin is toggled off', () => {
    expect(() =>
      renderInStage(
        baseProps({
          shapes: [predictedShape],
          originVisible: { human: true, predicted: false },
        }),
      ),
    ).not.toThrow();
  });

  it('recolors the active brush instance to the active class without crashing', () => {
    const brushInProgress: Shape = { ...brushShape, id: 'active-brush' };
    expect(() =>
      renderInStage(
        baseProps({
          shapes: [brushInProgress],
          activeBrushShapeId: 'active-brush',
          activeClassId: 2,
        }),
      ),
    ).not.toThrow();
  });

  it('renders selected shapes (thicker stroke path) without crashing', () => {
    expect(() =>
      renderInStage(baseProps({ shapes: [rectShape, ellipseShape], selectedShapeIds: ['rect-1'] })),
    ).not.toThrow();
  });

  it('does not throw when scaleX changes (stroke width depends on zoom)', () => {
    const props = baseProps({ shapes: [rectShape] });
    const { rerender, container } = renderInStage(props);
    rerender(
      <Stage width={256} height={256}>
        <ShapesLayer {...props} scaleX={4} />
      </Stage>,
    );
    expect(container.querySelectorAll('canvas').length).toBe(1);
  });

  it('forwards layerRef to the underlying Konva.Layer', () => {
    const layerRef = createRef<Konva.Layer>();
    renderInStage(baseProps({ layerRef }));
    expect(layerRef.current).not.toBeNull();
    // Real Konva.Layer instance, not a DOM node — spot-check its own API surface.
    expect(typeof layerRef.current?.getCanvas).toBe('function');
  });
});
