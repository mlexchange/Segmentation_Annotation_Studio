import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  applyProbaThresholdRgba,
  classProbaMaskSetName,
  colorizeConformalOverlay,
  colorizeLabelMap,
  labelMapToPolygonShapes,
  loadLabelPng,
  parseHexColor,
  thresholdProbaPngBlob,
} from './pixelClf';

/**
 * jsdom has no real 2D canvas; stub `getContext('2d')` with an in-memory
 * implementation of just the methods pixelClf.ts uses (createImageData /
 * putImageData / getImageData), so `colorizeLabelMap` / `colorizeConformalOverlay`
 * actually run their pixel-coloring loops instead of hitting the `!ctx` early
 * return that made the sibling `pixelClf.overlay.test.ts` a no-op smoke test.
 */
function stubCanvasContext() {
  const stores = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
    this: HTMLCanvasElement,
  ): any {
    const canvas = this;
    return {
      createImageData: (w: number, h: number) => {
        const data = new Uint8ClampedArray(w * h * 4);
        stores.set(canvas, data);
        return { data, width: w, height: h };
      },
      putImageData: () => {},
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: stores.get(canvas) ?? new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      drawImage: () => {},
    } as unknown as CanvasRenderingContext2D;
  });
}

beforeEach(() => stubCanvasContext());
afterEach(() => vi.restoreAllMocks());

describe('parseHexColor', () => {
  it('parses 3-digit shorthand hex by doubling each digit', () => {
    expect(parseHexColor('#0f8')).toEqual([0, 255, 136]);
  });
});

describe('classProbaMaskSetName', () => {
  it('clamps thresholds outside [0,1]', () => {
    expect(classProbaMaskSetName(1, 1.5, 'x')).toBe('x p≥100%');
    expect(classProbaMaskSetName(1, -0.5, 'x')).toBe('x p≥0%');
  });

  it('trims whitespace-only labels and falls back to the class id', () => {
    expect(classProbaMaskSetName(4, 0.2, '   ')).toBe('class 4 p≥20%');
  });
});

describe('applyProbaThresholdRgba edge thresholds', () => {
  it('threshold 0 colors every pixel (nothing is below the cut)', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 255]);
    applyProbaThresholdRgba(rgba, 0);
    expect(rgba[3]).toBe(255);
    expect(rgba[7]).toBe(255);
  });

  it('threshold 1 makes everything but a full-white pixel transparent (tiny span floor)', () => {
    const rgba = new Uint8ClampedArray([200, 200, 200, 255, 255, 255, 255, 255]);
    applyProbaThresholdRgba(rgba, 1);
    expect(rgba[3]).toBe(0); // p=200/255 < 1 -> transparent
    expect(rgba[7]).toBe(255); // p=1 -> exactly at/above cut
  });
});

describe('colorizeLabelMap', () => {
  it('colors known classes and leaves 0/unknown classes transparent', () => {
    const width = 2, height = 1;
    const labels = new Uint8Array([1, 2]); // 2 = unknown (no color registered)
    const colors = new Map([[1, '#ff0000']]);
    const canvas = colorizeLabelMap(labels, width, height, colors, 90);
    const ctx = canvas.getContext('2d')!;
    const { data } = ctx.getImageData(0, 0, width, height);
    // Pixel 0 -> class 1 -> red at alpha 90.
    expect([data[0], data[1], data[2], data[3]]).toEqual([255, 0, 0, 90]);
    // Pixel 1 -> class 2 has no registered color -> left transparent (all zero).
    expect([data[4], data[5], data[6], data[7]]).toEqual([0, 0, 0, 0]);
  });

  it('leaves class-0 (background) pixels transparent', () => {
    const labels = new Uint8Array([0]);
    const canvas = colorizeLabelMap(labels, 1, 1, new Map([[1, '#00ff00']]));
    const ctx = canvas.getContext('2d')!;
    const { data } = ctx.getImageData(0, 0, 1, 1);
    expect([...data]).toEqual([0, 0, 0, 0]);
  });
});

describe('colorizeConformalOverlay', () => {
  it('colors a singleton pixel with its class color at fixed alpha', () => {
    const commit = new Uint8Array([1]);
    const status = new Uint8Array([1]); // singleton
    const canvas = colorizeConformalOverlay(commit, status, 1, 1, new Map([[1, '#00ff00']]));
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, 1, 1);
    expect([...data]).toEqual([0, 255, 0, 120]);
  });

  it('hides a singleton pixel whose class is filtered out by classVisible', () => {
    const commit = new Uint8Array([1]);
    const status = new Uint8Array([1]);
    const canvas = colorizeConformalOverlay(commit, status, 1, 1, new Map([[1, '#00ff00']]), {
      classVisible: () => false,
    });
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, 1, 1);
    expect([...data]).toEqual([0, 0, 0, 0]);
  });

  it('leaves a singleton pixel transparent when its class has no registered color', () => {
    const commit = new Uint8Array([9]);
    const status = new Uint8Array([1]);
    const canvas = colorizeConformalOverlay(commit, status, 1, 1, new Map([[1, '#00ff00']]));
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, 1, 1);
    expect([...data]).toEqual([0, 0, 0, 0]);
  });

  it('renders a multi-class hatch pattern that alternates by (x+y) parity, and can be hidden', () => {
    // 4x1: status=2 (multi) at every pixel; (x+y)&3 < 2 is true for x=0,1 ("on")
    // and false for x=2,3 ("off"), giving both hatch colors in one row.
    const commit = new Uint8Array([0, 0, 0, 0]);
    const status = new Uint8Array([2, 2, 2, 2]);
    const canvas = colorizeConformalOverlay(commit, status, 4, 1, new Map());
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, 4, 1);
    expect([data[0], data[1], data[2], data[3]]).toEqual([245, 158, 11, 140]); // x=0: on
    expect([data[8], data[9], data[10], data[11]]).toEqual([180, 120, 40, 70]); // x=2: off

    const hidden = colorizeConformalOverlay(commit, status, 4, 1, new Map(), { showMulti: false });
    const hiddenData = hidden.getContext('2d')!.getImageData(0, 0, 4, 1).data;
    expect([...hiddenData]).toEqual(new Array(16).fill(0));
  });

  it('renders the abstain color for status 0 by default, and can hide it', () => {
    const commit = new Uint8Array([0]);
    const status = new Uint8Array([0]);
    const canvas = colorizeConformalOverlay(commit, status, 1, 1, new Map());
    const { data } = canvas.getContext('2d')!.getImageData(0, 0, 1, 1);
    expect([...data]).toEqual([30, 30, 40, 90]);

    const hidden = colorizeConformalOverlay(commit, status, 1, 1, new Map(), { showAbstain: false });
    const hiddenData = hidden.getContext('2d')!.getImageData(0, 0, 1, 1).data;
    expect([...hiddenData]).toEqual([0, 0, 0, 0]);
  });
});

describe('labelMapToPolygonShapes default options', () => {
  it('defaults origin to "predicted" when not specified', () => {
    const w = 16, h = 16;
    const labels = new Uint8Array(w * h);
    for (let y = 4; y < 12; y++) for (let x = 4; x < 12; x++) labels[y * w + x] = 1;
    const shapes = labelMapToPolygonShapes(labels, w, h, [1], { minRegion: 4, smooth: 0 });
    expect(shapes[0].origin).toBe('predicted');
  });

  it('honors an explicit "human" origin override', () => {
    const w = 16, h = 16;
    const labels = new Uint8Array(w * h);
    for (let y = 4; y < 12; y++) for (let x = 4; x < 12; x++) labels[y * w + x] = 1;
    const shapes = labelMapToPolygonShapes(labels, w, h, [1], { minRegion: 4, smooth: 0, origin: 'human' });
    expect(shapes[0].origin).toBe('human');
  });

  it('skips classes with no matching pixels entirely', () => {
    const w = 8, h = 8;
    const labels = new Uint8Array(w * h); // all zero
    const shapes = labelMapToPolygonShapes(labels, w, h, [1, 2], { minRegion: 1 });
    expect(shapes).toHaveLength(0);
  });

  it('drops tiny polygons below minRegion vertex count (<6 flat coords) even if any() was true', () => {
    // A 1-pixel "region" with default minRegion (64) is dropped by maskToPolygonsWithHoles
    // before points.length is even checked, so this also covers the minRegion path.
    const w = 32, h = 32;
    const labels = new Uint8Array(w * h);
    labels[10 * w + 10] = 1; // single pixel
    const shapes = labelMapToPolygonShapes(labels, w, h, [1]);
    expect(shapes).toHaveLength(0);
  });
});

/**
 * thresholdProbaPngBlob / loadLabelPng both decode a PNG via `new Image()` and
 * read it back through a 2D canvas — jsdom has neither a real `Image` decoder
 * nor `HTMLCanvasElement.toBlob`, so both are stubbed here. `stubCanvasContext`
 * above already covers getImageData/putImageData/drawImage; `installImageStub`
 * drives `Image`'s onload/onerror synchronously (as a microtask) so `await new
 * Promise(...)` in the source resolves without needing real image bytes.
 */
function installImageStub(opts: { fail?: boolean; width?: number; height?: number } = {}) {
  const { fail = false, width = 2, height = 1 } = opts;
  class StubImage {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    naturalWidth = 0;
    naturalHeight = 0;
    private _src = '';
    set src(v: string) {
      this._src = v;
      queueMicrotask(() => {
        if (fail) {
          this.onerror?.();
          return;
        }
        this.naturalWidth = width;
        this.naturalHeight = height;
        this.onload?.();
      });
    }
    get src() {
      return this._src;
    }
  }
  vi.stubGlobal('Image', StubImage);
}

describe('thresholdProbaPngBlob', () => {
  beforeEach(() => {
    global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = vi.fn();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('decodes, thresholds, and re-encodes as a PNG blob, then revokes the object URL', async () => {
    installImageStub({ width: 2, height: 1 });
    HTMLCanvasElement.prototype.toBlob = function (cb: BlobCallback) {
      cb(new Blob(['ok'], { type: 'image/png' }));
    };
    const out = await thresholdProbaPngBlob(new Blob(), 0.5);
    expect(out).toBeInstanceOf(Blob);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('rejects when the image fails to decode', async () => {
    installImageStub({ fail: true });
    await expect(thresholdProbaPngBlob(new Blob(), 0.5)).rejects.toThrow(
      'Failed to decode probability PNG',
    );
    // The object URL is still revoked even on failure (finally block).
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('rejects when the 2D context is unavailable', async () => {
    installImageStub();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    await expect(thresholdProbaPngBlob(new Blob(), 0.5)).rejects.toThrow('2D context unavailable');
  });

  it('rejects when canvas.toBlob yields no blob', async () => {
    installImageStub();
    HTMLCanvasElement.prototype.toBlob = function (cb: BlobCallback) {
      cb(null);
    };
    await expect(thresholdProbaPngBlob(new Blob(), 0.5)).rejects.toThrow('toBlob failed');
  });
});

describe('loadLabelPng', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('decodes a grayscale classId PNG into a Uint8Array + dimensions', async () => {
    installImageStub({ width: 2, height: 1 });
    const { data, width, height } = await loadLabelPng('http://example.test/label.png');
    expect(width).toBe(2);
    expect(height).toBe(1);
    expect(data).toBeInstanceOf(Uint8Array);
    expect(data.length).toBe(2);
  });

  it('rejects when the image fails to load', async () => {
    installImageStub({ fail: true });
    await expect(loadLabelPng('http://example.test/bad.png')).rejects.toThrow(
      'Failed to load prediction PNG',
    );
  });

  it('rejects when the 2D context is unavailable', async () => {
    installImageStub();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    await expect(loadLabelPng('http://example.test/label.png')).rejects.toThrow(
      '2D context unavailable',
    );
  });
});
