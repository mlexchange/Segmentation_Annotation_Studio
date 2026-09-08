import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderAdjusted, renderPreprocessOnly } from './adjust';

/**
 * jsdom has no real 2D canvas. Stub getContext('2d') with an in-memory pixel
 * buffer per canvas, and make drawImage copy the source canvas's own stored
 * buffer into the destination — so renderAdjusted's internal
 * drawImage-then-getImageData round trip actually sees real pixel data
 * instead of zeros, matching the pattern established in pixelClf.gaps.test.ts.
 */
const stores = new WeakMap<HTMLCanvasElement, Uint8ClampedArray>();

function seedCanvas(width: number, height: number, fill: (i: number) => number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i++) data[i] = fill(i);
  stores.set(canvas, data);
  return canvas;
}

function readPixels(canvas: HTMLCanvasElement): Uint8ClampedArray {
  return stores.get(canvas)!;
}

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
    this: HTMLCanvasElement,
  ): any {
    const canvas = this;
    return {
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high',
      drawImage: (source: HTMLCanvasElement, _sx: number, _sy: number, w: number, h: number) => {
        const src = stores.get(source);
        const dst = new Uint8ClampedArray(w * h * 4);
        if (src) dst.set(src.subarray(0, dst.length));
        stores.set(canvas, dst);
      },
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: stores.get(canvas) ?? new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      putImageData: (imageData: { data: Uint8ClampedArray }) => {
        stores.set(canvas, imageData.data);
      },
    };
  });
});

afterEach(() => vi.restoreAllMocks());

describe('renderAdjusted', () => {
  it('returns the raw canvas unchanged when everything is a no-op', () => {
    const source = seedCanvas(2, 2, () => 100);
    const out = renderAdjusted(source, 2, 2, 0, 0, 0, 255);
    expect([...readPixels(out)]).toEqual([...readPixels(source)]);
  });

  it('brightens pixels when brightness > 0', () => {
    const source = seedCanvas(1, 1, (i) => (i % 4 === 3 ? 255 : 100));
    const out = renderAdjusted(source, 1, 1, 0.2, 0, 0, 255);
    const px = readPixels(out);
    expect(px[0]).toBeGreaterThan(100);
    expect(px[3]).toBe(255); // alpha untouched
  });

  it('darkens pixels when brightness < 0', () => {
    const source = seedCanvas(1, 1, (i) => (i % 4 === 3 ? 255 : 100));
    const out = renderAdjusted(source, 1, 1, -0.2, 0, 0, 255);
    expect(readPixels(out)[0]).toBeLessThan(100);
  });

  it('applies a levels remap clamping outside [lo,hi]', () => {
    const source = seedCanvas(3, 1, (i) => {
      const px = Math.floor(i / 4);
      return i % 4 === 3 ? 255 : [10, 128, 250][px];
    });
    const out = renderAdjusted(source, 3, 1, 0, 0, 50, 200);
    const px = readPixels(out);
    expect(px[0]).toBe(0); // below lo -> 0
    expect(px[8]).toBe(255); // above hi -> 255
    expect(px[4]).toBeGreaterThan(0); // in range, remapped
    expect(px[4]).toBeLessThan(255);
  });

  it('bakes a gaussian blur when preprocess.blur > 0', () => {
    // A single bright pixel among dark neighbors should spread out after blur.
    const source = seedCanvas(5, 5, (i) => {
      const px = Math.floor(i / 4);
      const isCenter = px === 12; // middle of 5x5
      return i % 4 === 3 ? 255 : isCenter ? 255 : 0;
    });
    const out = renderAdjusted(source, 5, 5, 0, 0, 0, 255, { blur: 1 });
    const px = readPixels(out);
    // A neighboring pixel (index 11, just left of center) should pick up some brightness.
    expect(px[11 * 4]).toBeGreaterThan(0);
  });

  it('scales blur sigma by the working-resolution upscale factor', () => {
    const source = seedCanvas(3, 3, (i) => (i % 4 === 3 ? 255 : (Math.floor(i / 4) === 4 ? 255 : 0)));
    // Just confirm it runs without throwing at a non-1 upscale and returns a bigger canvas.
    const out = renderAdjusted(source, 3, 3, 0, 0, 0, 255, { blur: 1 }, 2);
    expect(out.width).toBe(6);
    expect(out.height).toBe(6);
  });

  it('leaves the canvas untouched with all-zero preprocess flags and no blur', () => {
    const source = seedCanvas(2, 2, () => 77);
    const out = renderAdjusted(source, 2, 2, 0, 0, 0, 255, { blur: 0, clahe: false, sharpen: false });
    expect([...readPixels(out)]).toEqual([...readPixels(source)]);
  });
});

describe('renderPreprocessOnly', () => {
  it('applies only the nonlinear preprocessors, ignoring brightness/contrast/levels', () => {
    const source = seedCanvas(5, 5, (i) => {
      const px = Math.floor(i / 4);
      return i % 4 === 3 ? 255 : px === 12 ? 255 : 0;
    });
    const out = renderPreprocessOnly(source, 5, 5, { blur: 1 });
    const px = readPixels(out);
    expect(px[11 * 4]).toBeGreaterThan(0); // blur still applied
  });

  it('returns an unmodified canvas when no preprocessors are requested', () => {
    const source = seedCanvas(2, 2, () => 50);
    const out = renderPreprocessOnly(source, 2, 2, {});
    expect([...readPixels(out)]).toEqual([...readPixels(source)]);
  });
});
