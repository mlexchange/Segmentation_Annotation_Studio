import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mark, time, snapshot, resetPerf, perfEnabled, initPerf } from './perf';

/** initPerf reads the flag from URL/localStorage; force it on for these tests. */
function enable() {
  window.localStorage.setItem('perf', '1');
  initPerf();
}

beforeEach(() => {
  window.localStorage.clear();
  resetPerf();
});

afterEach(() => {
  window.localStorage.clear();
  initPerf();
  resetPerf();
});

describe('perf gating', () => {
  it('is disabled by default and records nothing', () => {
    initPerf();
    expect(perfEnabled()).toBe(false);
    mark('commit', 123);
    expect(snapshot()).toEqual([]);
  });

  it('still runs the timed function when disabled', () => {
    initPerf();
    const fn = vi.fn(() => 42);
    expect(time('commit', fn)).toBe(42);
    expect(fn).toHaveBeenCalledOnce();
    expect(snapshot()).toEqual([]);
  });

  it('enables via localStorage', () => {
    enable();
    expect(perfEnabled()).toBe(true);
    mark('commit', 5);
    expect(snapshot()).toHaveLength(1);
  });
});

describe('perf statistics', () => {
  beforeEach(enable);

  it('reports p50/p95 over recorded samples', () => {
    for (let i = 1; i <= 100; i++) mark('commit', i);
    const [stat] = snapshot();
    expect(stat.label).toBe('commit');
    expect(stat.count).toBe(100);
    expect(stat.p50).toBeGreaterThanOrEqual(50);
    expect(stat.p50).toBeLessThanOrEqual(52);
    expect(stat.p95).toBeGreaterThanOrEqual(95);
    expect(stat.last).toBe(100);
  });

  it('sorts slowest p95 first', () => {
    mark('commit', 1);
    mark('clip', 100);
    mark('merge', 50);
    expect(snapshot().map((s) => s.label)).toEqual(['clip', 'merge', 'commit']);
  });

  it('keeps a bounded window of samples', () => {
    for (let i = 0; i < 500; i++) mark('commit', i);
    expect(snapshot()[0].count).toBeLessThanOrEqual(120);
  });

  it('records the duration of a timed function and returns its value', () => {
    const out = time('merge', () => 'result');
    expect(out).toBe('result');
    const [stat] = snapshot();
    expect(stat.label).toBe('merge');
    expect(stat.count).toBe(1);
    expect(stat.last).toBeGreaterThanOrEqual(0);
  });

  it('records a sample even when the timed function throws', () => {
    expect(() => time('clip', () => { throw new Error('boom'); })).toThrow('boom');
    expect(snapshot().find((s) => s.label === 'clip')?.count).toBe(1);
  });

  it('resetPerf clears everything', () => {
    mark('commit', 10);
    resetPerf();
    expect(snapshot()).toEqual([]);
  });
});
