import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import {
  generateGuide, loadGuide, saveGuide, useGuideLoad, useGuideSync,
} from './useGuideSync';
import { useReferenceGuideStore } from '@/stores/referenceGuideStore';

beforeEach(() => {
  useReferenceGuideStore.getState().clear();
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('loadGuide', () => {
  it('returns null on a 404 (no guide exists)', async () => {
    (fetch as any).mockResolvedValue({ status: 404, ok: false });
    expect(await loadGuide('local:x.tif')).toBeNull();
  });

  it('returns null on any fetch error (never throws)', async () => {
    (fetch as any).mockRejectedValue(new Error('network down'));
    expect(await loadGuide('local:x.tif')).toBeNull();
  });

  it('parses classes/notes from a successful response', async () => {
    (fetch as any).mockResolvedValue({
      status: 200, ok: true,
      json: async () => ({ guide: { classes: [{ label: 'Cell', color: '#f00', description: '', exampleCrops: [] }], notes: 'hi' } }),
    });
    expect(await loadGuide('local:x.tif')).toEqual({
      classes: [{ label: 'Cell', color: '#f00', description: '', exampleCrops: [] }], notes: 'hi',
    });
  });

  it('defaults to empty classes/notes when the guide object is bare', async () => {
    (fetch as any).mockResolvedValue({ status: 200, ok: true, json: async () => ({ guide: {} }) });
    expect(await loadGuide('local:x.tif')).toEqual({ classes: [], notes: '' });
  });
});

describe('saveGuide', () => {
  it('PUTs the classes/notes and returns true on success', async () => {
    (fetch as any).mockResolvedValue({ ok: true });
    const ok = await saveGuide('local:x.tif', [], 'notes');
    expect(ok).toBe(true);
    const [url, init] = (fetch as any).mock.calls[0];
    expect(url).toContain('/api/guide?source_key=');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({ classes: [], notes: 'notes' });
  });

  it('returns false without throwing on a network error', async () => {
    (fetch as any).mockRejectedValue(new Error('down'));
    expect(await saveGuide('local:x.tif', [], '')).toBe(false);
  });

  it('returns false when the server responds not-ok', async () => {
    (fetch as any).mockResolvedValue({ ok: false });
    expect(await saveGuide('local:x.tif', [], '')).toBe(false);
  });
});

describe('generateGuide', () => {
  it('POSTs the payload and returns the generated guide', async () => {
    (fetch as any).mockResolvedValue({
      ok: true, json: async () => ({ classes: [{ label: 'Pore', color: '#000', description: '', exampleCrops: [] }], notes: '' }),
    });
    const result = await generateGuide('local:x.tif', { classes: [], slices: {} });
    expect(result.classes).toHaveLength(1);
  });

  it('throws a descriptive error including the response detail on failure', async () => {
    (fetch as any).mockResolvedValue({ ok: false, status: 500, text: async () => 'boom' });
    await expect(generateGuide('local:x.tif', { classes: [], slices: {} })).rejects.toThrow(/500.*boom/);
  });
});

describe('useGuideLoad', () => {
  it('loads the guide into the store for a given sourceKey', async () => {
    (fetch as any).mockResolvedValue({
      status: 200, ok: true, json: async () => ({ guide: { classes: [{ label: 'Cell', color: '#f00', description: '', exampleCrops: [] }], notes: 'n' } }),
    });
    renderHook(() => useGuideLoad('local:x.tif'));
    await waitFor(() => expect(useReferenceGuideStore.getState().loadedFor).toBe('local:x.tif'));
    expect(useReferenceGuideStore.getState().entries).toHaveLength(1);
  });

  it('clears the store when sourceKey is null', () => {
    useReferenceGuideStore.getState().setGuide([{ label: 'x', color: '#000', description: '', exampleCrops: [] }], 'n', 'local:y.tif');
    renderHook(() => useGuideLoad(null));
    expect(useReferenceGuideStore.getState().entries).toEqual([]);
  });
});

describe('useGuideSync', () => {
  it('debounces an autosave after the guide has loaded for this source', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    (fetch as any).mockImplementation(async (url: string) => {
      if (url.includes('source_key=local%3Ax.tif') && !url.includes('generate')) {
        return { status: 200, ok: true, json: async () => ({ guide: { classes: [], notes: '' } }) };
      }
      return { ok: true };
    });
    renderHook(() => useGuideSync('local:x.tif'));

    await waitFor(() => expect(useReferenceGuideStore.getState().loadedFor).toBe('local:x.tif'));
    (fetch as any).mockClear();

    act(() => {
      useReferenceGuideStore.getState().setGuide([{ label: 'Cell', color: '#f00', description: '', exampleCrops: [] }], '', 'local:x.tif');
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/guide?source_key='),
      expect.objectContaining({ method: 'PUT' }),
    );
  });

  it('does not autosave before the guide has finished loading for this source', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    (fetch as any).mockReturnValue(new Promise(() => {})); // never resolves -> loadedFor stays unset
    renderHook(() => useGuideSync('local:x.tif'));
    await vi.advanceTimersByTimeAsync(2000);
    // Only the initial GET attempt (still pending), no PUT.
    const putCalls = (fetch as any).mock.calls.filter(([, init]: any) => init?.method === 'PUT');
    expect(putCalls).toHaveLength(0);
  });
});
