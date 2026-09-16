import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { useKeybinds } from './useKeybinds';
import { useToolStore } from '@/stores/toolStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import * as editHistory from '@/hooks/editHistory';

const INITIAL_TOOL_STATE = useToolStore.getState();
const INITIAL_DATASET_STATE = useDatasetStore.getState();

function openSample(nSlices = 5) {
  useDatasetStore.getState().setDataset('local', 'x.tif', null, {
    nSlices,
    height: 10,
    width: 10,
    dtype: 'uint8',
    isRgb: false,
    valueRange: [0, 255],
  });
}

function keydown(key: string, opts: Partial<KeyboardEventInit> = {}) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...opts }));
  });
}
function keyup(key: string, opts: Partial<KeyboardEventInit> = {}) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true, cancelable: true, ...opts }));
  });
}

describe('useKeybinds', () => {
  let onActivateClass: ReturnType<typeof vi.fn>;
  let onNewBrushInstance: ReturnType<typeof vi.fn>;
  let onDeleteSelected: ReturnType<typeof vi.fn>;
  let onCancelDraft: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    useToolStore.setState(INITIAL_TOOL_STATE, true);
    useDatasetStore.setState(INITIAL_DATASET_STATE, true);
    useClassStore.setState({ classes: [] });
    onActivateClass = vi.fn();
    onNewBrushInstance = vi.fn();
    onDeleteSelected = vi.fn();
    onCancelDraft = vi.fn();
  });

  afterEach(() => {
    cleanup();
  });

  function mount(activeClassId: number | null = null) {
    return renderHook(() =>
      useKeybinds(activeClassId, onActivateClass, onNewBrushInstance, onDeleteSelected, onCancelDraft),
    );
  }

  it('switches tools on their letter key', () => {
    mount();
    keydown('p');
    expect(useToolStore.getState().tool).toBe('polygon');
    keydown('b');
    expect(useToolStore.getState().tool).toBe('brush');
    keydown('r');
    expect(useToolStore.getState().tool).toBe('eraser');
    keydown('g');
    expect(useToolStore.getState().tool).toBe('magic');
  });

  it('ignores keys typed into an editable target (input/textarea/select)', () => {
    mount();
    const input = document.createElement('input');
    document.body.appendChild(input);
    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'p', bubbles: true }));
    });
    expect(useToolStore.getState().tool).not.toBe('polygon');
    document.body.removeChild(input);
  });

  it('activates a class 1-9 via digit keys, mapped by position', () => {
    useClassStore.setState({
      classes: [
        { classId: 10, label: 'A', color: '#f00', isVisible: true },
        { classId: 20, label: 'B', color: '#0f0', isVisible: true },
      ],
    });
    mount();
    keydown('2');
    expect(onActivateClass).toHaveBeenCalledWith(20);
  });

  it('does nothing for a digit beyond the class list length', () => {
    useClassStore.setState({ classes: [{ classId: 1, label: 'A', color: '#f00', isVisible: true }] });
    mount();
    keydown('5');
    expect(onActivateClass).not.toHaveBeenCalled();
  });

  it('holding Space switches to pan and releasing restores the previous tool', () => {
    mount();
    act(() => { useToolStore.getState().setTool('brush'); });
    keydown(' ');
    expect(useToolStore.getState().tool).toBe('pan');
    expect(useToolStore.getState().panReturnTool).toBe('brush');

    keyup(' ');
    expect(useToolStore.getState().tool).toBe('brush');
    expect(useToolStore.getState().panReturnTool).toBeNull();
  });

  it('a repeated (auto-repeat) Space keydown does not re-arm the pan-return tool', () => {
    mount();
    act(() => { useToolStore.getState().setTool('brush'); });
    keydown(' ');
    expect(useToolStore.getState().tool).toBe('pan');
    // Simulate OS key-repeat: tool is already 'pan', repeat=true should just bail out.
    keydown(' ', { repeat: true });
    expect(useToolStore.getState().panReturnTool).toBe('brush');
  });

  it('Ctrl/Cmd+Z calls editHistory.undo(); Ctrl+Shift+Z and Ctrl+Y call redo()', () => {
    const undoSpy = vi.spyOn(editHistory, 'undo').mockImplementation(() => {});
    const redoSpy = vi.spyOn(editHistory, 'redo').mockImplementation(() => {});
    mount();

    keydown('z', { ctrlKey: true });
    expect(undoSpy).toHaveBeenCalledTimes(1);
    expect(redoSpy).not.toHaveBeenCalled();

    keydown('z', { ctrlKey: true, shiftKey: true });
    expect(redoSpy).toHaveBeenCalledTimes(1);

    keydown('y', { ctrlKey: true });
    expect(redoSpy).toHaveBeenCalledTimes(2);

    keydown('z', { metaKey: true });
    expect(undoSpy).toHaveBeenCalledTimes(2);

    undoSpy.mockRestore();
    redoSpy.mockRestore();
  });

  it('"t" requests a fit-to-screen (bumps fitRequestId)', () => {
    mount();
    const before = useToolStore.getState().fitRequestId;
    keydown('t');
    expect(useToolStore.getState().fitRequestId).toBe(before + 1);
  });

  it('"x" advances to the next slice, clamped to the last slice', () => {
    openSample(3);
    mount();
    keydown('x');
    expect(useDatasetStore.getState().currentSlice).toBe(1);
    keydown('x');
    keydown('x');
    keydown('x'); // beyond the end, clamps to nSlices-1
    expect(useDatasetStore.getState().currentSlice).toBe(2);
  });

  it('ArrowLeft/ArrowRight navigate slices, clamped at 0', () => {
    openSample(3);
    mount();
    keydown('ArrowLeft'); // already at 0, clamps
    expect(useDatasetStore.getState().currentSlice).toBe(0);
    keydown('ArrowRight');
    expect(useDatasetStore.getState().currentSlice).toBe(1);
    keydown('ArrowLeft');
    expect(useDatasetStore.getState().currentSlice).toBe(0);
  });

  it('slice navigation is a no-op with no dataset loaded (meta is null)', () => {
    mount();
    keydown('x');
    expect(useDatasetStore.getState().currentSlice).toBe(0);
  });

  it('"n" triggers onNewBrushInstance, Delete/Backspace trigger onDeleteSelected, Escape triggers onCancelDraft', () => {
    mount();
    keydown('n');
    expect(onNewBrushInstance).toHaveBeenCalledTimes(1);
    keydown('Delete');
    expect(onDeleteSelected).toHaveBeenCalledTimes(1);
    keydown('Backspace');
    expect(onDeleteSelected).toHaveBeenCalledTimes(2);
    keydown('Escape');
    expect(onCancelDraft).toHaveBeenCalledTimes(1);
  });

  it('removes its listeners on unmount', () => {
    const { unmount } = mount();
    unmount();
    keydown('p');
    expect(useToolStore.getState().tool).not.toBe('polygon');
  });
});
