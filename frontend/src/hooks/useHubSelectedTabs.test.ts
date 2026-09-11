import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useHubSelectedTabs } from './useHubSelectedTabs';

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useHubSelectedTabs', () => {
  it('starts null with nothing stored', () => {
    const { result } = renderHook(() => useHubSelectedTabs());
    expect(result.current.selectedPaths).toBeNull();
  });

  it('loads previously stored paths on mount', () => {
    localStorage.setItem('sam3_hub_selected_tab_paths', JSON.stringify(['/connect', '/browse']));
    const { result } = renderHook(() => useHubSelectedTabs());
    expect(result.current.selectedPaths).toEqual(['/connect', '/browse']);
  });

  it('ignores malformed JSON and starts null', () => {
    localStorage.setItem('sam3_hub_selected_tab_paths', 'not json');
    const { result } = renderHook(() => useHubSelectedTabs());
    expect(result.current.selectedPaths).toBeNull();
  });

  it('ignores a non-array stored value', () => {
    localStorage.setItem('sam3_hub_selected_tab_paths', JSON.stringify({ a: 1 }));
    const { result } = renderHook(() => useHubSelectedTabs());
    expect(result.current.selectedPaths).toBeNull();
  });

  it('ignores an array with non-string items', () => {
    localStorage.setItem('sam3_hub_selected_tab_paths', JSON.stringify([1, 2, 3]));
    const { result } = renderHook(() => useHubSelectedTabs());
    expect(result.current.selectedPaths).toBeNull();
  });

  it('setSelectedPaths writes through to localStorage and updates state', () => {
    const { result } = renderHook(() => useHubSelectedTabs());
    act(() => result.current.setSelectedPaths(['/annotate']));
    expect(result.current.selectedPaths).toEqual(['/annotate']);
    expect(JSON.parse(localStorage.getItem('sam3_hub_selected_tab_paths')!)).toEqual(['/annotate']);
  });

  it('logs and keeps prior state when localStorage.setItem throws', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota exceeded');
    });
    const { result } = renderHook(() => useHubSelectedTabs());
    act(() => result.current.setSelectedPaths(['/annotate']));
    expect(result.current.selectedPaths).toBeNull();
    expect(errorSpy).toHaveBeenCalled();
    setItemSpy.mockRestore();
  });
});
