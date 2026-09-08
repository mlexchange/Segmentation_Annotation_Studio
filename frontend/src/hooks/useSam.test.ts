/**
 * useSam wraps samClient, a singleton that owns a real Web Worker (spawned via
 * `new Worker(new URL('./samWorker.ts', import.meta.url), ...)`). jsdom doesn't
 * implement Worker, and vendoring a fake Worker global would still leave us
 * exercising the worker's postMessage protocol rather than the hook's own logic.
 * So — matching the pattern already used in Toolbar/index.test.tsx, which mocks
 * `@/hooks/useSam` wholesale rather than deal with the worker boundary — we mock
 * `@/lib/sam/samClient` here instead and test useSam's actual public surface
 * (status subscription, ensureEncoded's caching/dedup, segment) against it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';

const subscribers = new Set<(s: string) => void>();
let currentStatus = 'idle';
const encode = vi.fn();
const decode = vi.fn();
const init = vi.fn();
const getStatus = vi.fn(() => currentStatus);
const getBackend = vi.fn(() => 'wasm');

vi.mock('@/lib/sam/samClient', () => ({
  samClient: {
    getStatus: () => getStatus(),
    getBackend: () => getBackend(),
    subscribe: (fn: (s: string) => void) => {
      subscribers.add(fn);
      fn(currentStatus);
      return () => subscribers.delete(fn);
    },
    init: (...args: unknown[]) => init(...args),
    encode: (...args: unknown[]) => encode(...args),
    decode: (...args: unknown[]) => decode(...args),
  },
  webgpuAvailable: () => false,
}));

import { useSam } from './useSam';

function setStatus(s: string) {
  currentStatus = s;
  subscribers.forEach((fn) => fn(s));
}

beforeEach(() => {
  currentStatus = 'idle';
  subscribers.clear();
  encode.mockReset().mockResolvedValue(undefined);
  decode.mockReset().mockResolvedValue({ mask: new Uint8Array([1]), width: 1, height: 1, score: 0.9 });
  init.mockReset().mockResolvedValue(undefined);
  getStatus.mockClear();
  getBackend.mockClear();
});

afterEach(() => {
  cleanup();
});

describe('useSam', () => {
  it('reflects samClient.getStatus() at mount and updates on subscribe notifications', () => {
    const { result } = renderHook(() => useSam(false));
    expect(result.current.status).toBe('idle');
    expect(result.current.supported).toBe(true);

    act(() => setStatus('ready'));
    expect(result.current.status).toBe('ready');
  });

  it('supported is false once status flips to unsupported', () => {
    const { result } = renderHook(() => useSam(false));
    act(() => setStatus('unsupported'));
    expect(result.current.supported).toBe(false);
  });

  it('calls samClient.init() when enabled and status is idle', () => {
    renderHook(() => useSam(true));
    expect(init).toHaveBeenCalledTimes(1);
  });

  it('does not call init() when disabled', () => {
    renderHook(() => useSam(false));
    expect(init).not.toHaveBeenCalled();
  });

  it('does not call init() again when status is already past idle', () => {
    currentStatus = 'ready';
    renderHook(() => useSam(true));
    expect(init).not.toHaveBeenCalled();
  });

  it('swallows an init() rejection without throwing', async () => {
    init.mockRejectedValue(new Error('no webgpu'));
    expect(() => renderHook(() => useSam(true))).not.toThrow();
    await waitFor(() => expect(init).toHaveBeenCalled());
  });

  it('ensureEncoded encodes once per key and returns true on success', async () => {
    const { result } = renderHook(() => useSam(false));
    const makeSource = vi.fn(() => ({}) as CanvasImageSource);

    let ok = false;
    await act(async () => {
      ok = await result.current.ensureEncoded('slice-0', makeSource);
    });
    expect(ok).toBe(true);
    expect(encode).toHaveBeenCalledTimes(1);
    expect(makeSource).toHaveBeenCalledTimes(1);

    // Same key again -> cached, no new encode/makeSource call.
    await act(async () => {
      ok = await result.current.ensureEncoded('slice-0', makeSource);
    });
    expect(ok).toBe(true);
    expect(encode).toHaveBeenCalledTimes(1);
    expect(makeSource).toHaveBeenCalledTimes(1);
  });

  it('ensureEncoded re-encodes when the key changes (e.g. brightness/contrast changed)', async () => {
    const { result } = renderHook(() => useSam(false));
    const makeSource = vi.fn(() => ({}) as CanvasImageSource);

    await act(async () => {
      await result.current.ensureEncoded('slice-0', makeSource);
    });
    await act(async () => {
      await result.current.ensureEncoded('slice-0:bc=1', makeSource);
    });
    expect(encode).toHaveBeenCalledTimes(2);
  });

  it('ensureEncoded returns false and sets error on a failed encode, and does not retry a known-bad key', async () => {
    encode.mockRejectedValue(new Error('encode failed'));
    const { result } = renderHook(() => useSam(false));
    const makeSource = vi.fn(() => ({}) as CanvasImageSource);

    let ok = true;
    await act(async () => {
      ok = await result.current.ensureEncoded('bad-key', makeSource);
    });
    expect(ok).toBe(false);
    expect(result.current.error).toBe('encode failed');
    expect(encode).toHaveBeenCalledTimes(1);

    // Known-bad key: no retry, no second makeSource/encode call.
    await act(async () => {
      ok = await result.current.ensureEncoded('bad-key', makeSource);
    });
    expect(ok).toBe(false);
    expect(encode).toHaveBeenCalledTimes(1);
    expect(makeSource).toHaveBeenCalledTimes(1);
  });

  it('ensureEncoded awaits a concurrent in-flight encode for the same key rather than starting a second one', async () => {
    let resolveEncode!: () => void;
    encode.mockReturnValue(new Promise<void>((resolve) => { resolveEncode = resolve; }));
    const { result } = renderHook(() => useSam(false));
    const makeSource = vi.fn(() => ({}) as CanvasImageSource);

    let p1: Promise<boolean>;
    let p2: Promise<boolean>;
    act(() => {
      p1 = result.current.ensureEncoded('k', makeSource);
      p2 = result.current.ensureEncoded('k', makeSource);
    });
    expect(encode).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveEncode();
      await Promise.all([p1, p2]);
    });
    expect(encode).toHaveBeenCalledTimes(1);
  });

  it('segment returns null immediately when there are no points and no box', async () => {
    const { result } = renderHook(() => useSam(false));
    const mask = await result.current.segment([], null, 'auto', 0);
    expect(mask).toBeNull();
    expect(decode).not.toHaveBeenCalled();
  });

  it('segment decodes points/box via samClient and returns the mask', async () => {
    const { result } = renderHook(() => useSam(false));
    const mask = await result.current.segment([{ x: 1, y: 2, label: 1 }], null, 'fine', 0.5);
    expect(decode).toHaveBeenCalledWith([{ x: 1, y: 2, label: 1 }], null, 'fine', 0.5);
    expect(mask).toEqual({ mask: new Uint8Array([1]), width: 1, height: 1, score: 0.9 });
  });

  it('segment returns null (not throw) when decode rejects', async () => {
    decode.mockRejectedValue(new Error('decode failed'));
    const { result } = renderHook(() => useSam(false));
    const mask = await result.current.segment([{ x: 1, y: 2, label: 1 }], null, 'auto', 0);
    expect(mask).toBeNull();
  });

  it('exposes backend from samClient.getBackend()', () => {
    getBackend.mockReturnValue('webgpu');
    const { result } = renderHook(() => useSam(false));
    expect(result.current.backend).toBe('webgpu');
  });
});
