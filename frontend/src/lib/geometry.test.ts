import { describe, it, expect } from 'vitest';
import { toImage, toStage, normalizeRect, normalizeEllipse } from '../geometry';

const T = { scaleX: 2, scaleY: 2, x: 100, y: 50 };

describe('geometry', () => {
  it('toImage round-trips with toStage', () => {
    const img = { x: 30, y: 40 };
    const stage = toStage(img, T);
    const back = toImage(stage, T);
    expect(back.x).toBeCloseTo(img.x);
    expect(back.y).toBeCloseTo(img.y);
  });

  it('toImage converts stage coords to image pixels', () => {
    const result = toImage({ x: 100, y: 50 }, T);
    expect(result.x).toBeCloseTo(0);
    expect(result.y).toBeCloseTo(0);
  });

  it('normalizeRect fixes negative width', () => {
    const r = normalizeRect(10, 10, -5, 8);
    expect(r.x).toBe(5);
    expect(r.y).toBe(10);
    expect(r.w).toBe(5);
    expect(r.h).toBe(8);
  });

  it('normalizeRect fixes negative height', () => {
    const r = normalizeRect(10, 20, 8, -5);
    expect(r.x).toBe(10);
    expect(r.y).toBe(15);
    expect(r.w).toBe(8);
    expect(r.h).toBe(5);
  });

  it('normalizeRect all four drag directions yield same result', () => {
    const base = normalizeRect(5, 5, 10, 10);
    expect(normalizeRect(15, 5, -10, 10)).toEqual(base);
    expect(normalizeRect(5, 15, 10, -10)).toEqual(base);
    expect(normalizeRect(15, 15, -10, -10)).toEqual(base);
  });

  it('normalizeEllipse ensures positive radii', () => {
    const e = normalizeEllipse(10, 10, -5, -3);
    expect(e.rx).toBe(5);
    expect(e.ry).toBe(3);
  });
});
