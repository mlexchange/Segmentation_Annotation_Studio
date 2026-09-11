import { beforeEach, describe, expect, it } from 'vitest';
import { useClipboardStore } from './clipboardStore';

beforeEach(() => {
  useClipboardStore.setState({ shapes: [] });
});

describe('clipboardStore', () => {
  it('starts empty', () => {
    expect(useClipboardStore.getState().shapes).toEqual([]);
  });

  it('copy stores a deep copy of the given shapes', () => {
    const shape = { id: 's1', classId: 1, kind: 'rectangle' as const, x: 0, y: 0, w: 2, h: 2 };
    useClipboardStore.getState().copy([shape]);
    const stored = useClipboardStore.getState().shapes;
    expect(stored).toEqual([shape]);
    expect(stored[0]).not.toBe(shape); // deep copy, not the same reference

    shape.x = 99;
    const restored = useClipboardStore.getState().shapes[0];
    expect(restored.kind === 'rectangle' && restored.x).toBe(0); // unaffected by later mutation
  });

  it('clear empties the clipboard', () => {
    useClipboardStore.getState().copy([{ id: 's1', classId: 1, kind: 'rectangle', x: 0, y: 0, w: 1, h: 1 }]);
    useClipboardStore.getState().clear();
    expect(useClipboardStore.getState().shapes).toEqual([]);
  });
});
