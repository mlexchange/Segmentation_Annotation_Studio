import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { useOpenInAnnotate } from './useOpenInAnnotate';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useAnnotationStore } from '@/stores/annotationStore';

const navigateMock = vi.fn();
vi.mock('react-router', () => ({
  useNavigate: () => navigateMock,
}));

function metaResponse(overrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    json: async () => ({
      n_slices: 5,
      height: 100,
      width: 200,
      dtype: 'uint8',
      is_rgb: false,
      value_range: [0, 255],
      keywords: [],
      ...overrides,
    }),
  };
}

function draft404() {
  return { status: 404, ok: false };
}

beforeEach(() => {
  useDatasetStore.getState().reset();
  useClassStore.setState({ classes: [] });
  useAnnotationStore.getState().reset();
  navigateMock.mockClear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('useOpenInAnnotate', () => {
  describe('openTiledArray', () => {
    it('fetches meta, seeds the dataset store, and navigates to /annotate', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse())
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());

      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv');
      });

      expect(useDatasetStore.getState().kind).toBe('tiled');
      expect(useDatasetStore.getState().source).toBe('browse/ds/img');
      expect(useDatasetStore.getState().serverUri).toBe('http://srv');
      expect(useDatasetStore.getState().meta?.nSlices).toBe(5);
      expect(navigateMock).toHaveBeenCalledWith('/annotate');

      const [url] = (fetch as any).mock.calls[0];
      expect(url).toContain('/api/image/meta?');
      expect(url).toContain('source=browse%2Fds%2Fimg');
      expect(url).toContain('kind=tiled');
      expect(url).toContain('server_uri=');
    });

    it('jumps to the requested initialSlice, clamped to the last slice', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse({ n_slices: 5 }))
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());

      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv', 999);
      });
      expect(useDatasetStore.getState().currentSlice).toBe(4); // clamped to n_slices-1
    });

    it('does not override slice 0 when initialSlice is 0 (default)', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse())
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());
      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv');
      });
      expect(useDatasetStore.getState().currentSlice).toBe(0);
    });

    it('seeds classes from meta.keywords when the draft has none', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse({ keywords: ['Cell', 'Pore', 'cell'] }))
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());
      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv');
      });
      const classes = useClassStore.getState().classes;
      // 'cell' is deduped case-insensitively against 'Cell'.
      expect(classes.map((c) => c.label)).toEqual(['Cell', 'Pore']);
    });

    it('adopts classes from a loaded draft instead of keyword-seeding', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse({ keywords: ['ShouldNotAppear'] }))
        .mockResolvedValueOnce({
          status: 200,
          ok: true,
          json: async () => ({
            payload: {
              classes: [{ classId: 1, label: 'FromDraft', color: '#000', isVisible: true }],
              slices: { '0': [{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1 }] },
              split_by_slice: {},
              negative_slices: [],
            },
          }),
        }));
      const { result } = renderHook(() => useOpenInAnnotate());
      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv');
      });
      expect(useClassStore.getState().classes).toEqual([
        { classId: 1, label: 'FromDraft', color: '#000', isVisible: true },
      ]);
      const sourceKey = 'tiled:http://srv:browse/ds/img';
      expect(useAnnotationStore.getState().byImage[sourceKey]?.['0']).toHaveLength(1);
    });

    it('resets to empty classes when neither the draft nor keywords supply any', async () => {
      useClassStore.setState({ classes: [{ classId: 9, label: 'Stale', color: '#fff', isVisible: true }] });
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse({ keywords: [] }))
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());
      await act(async () => {
        await result.current.openTiledArray('browse/ds/img', 'http://srv');
      });
      expect(useClassStore.getState().classes).toEqual([]);
    });

    it('throws with the response body when the meta fetch fails', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, text: async () => 'not found' }));
      const { result } = renderHook(() => useOpenInAnnotate());
      await expect(result.current.openTiledArray('browse/ds/img', 'http://srv')).rejects.toThrow('not found');
      expect(navigateMock).not.toHaveBeenCalled();
    });
  });

  describe('openLocalFile', () => {
    it('fetches meta with kind=local, seeds the dataset store, and navigates', async () => {
      vi.stubGlobal('fetch', vi.fn()
        .mockResolvedValueOnce(metaResponse())
        .mockResolvedValueOnce(draft404()));
      const { result } = renderHook(() => useOpenInAnnotate());
      await act(async () => {
        await result.current.openLocalFile('rel/path.tif');
      });
      expect(useDatasetStore.getState().kind).toBe('local');
      expect(useDatasetStore.getState().source).toBe('rel/path.tif');
      expect(useDatasetStore.getState().serverUri).toBeNull();
      expect(navigateMock).toHaveBeenCalledWith('/annotate');

      const [url] = (fetch as any).mock.calls[0];
      expect(url).toContain('kind=local');
      expect(url).not.toContain('server_uri');
    });

    it('throws with the response body when the meta fetch fails', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, text: async () => 'boom' }));
      const { result } = renderHook(() => useOpenInAnnotate());
      await expect(result.current.openLocalFile('rel/path.tif')).rejects.toThrow('boom');
      expect(navigateMock).not.toHaveBeenCalled();
    });
  });
});
