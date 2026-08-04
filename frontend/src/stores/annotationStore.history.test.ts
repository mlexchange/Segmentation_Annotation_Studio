import { beforeEach, describe, expect, it } from 'vitest';
import { useAnnotationStore } from './annotationStore';

describe('annotation temporal history', () => {
  beforeEach(() => {
    useAnnotationStore.getState().reset();
    useAnnotationStore.temporal.getState().clear();
  });

  it('undoes and redoes split assignments', () => {
    useAnnotationStore.getState().setSplitForSlice('sample', 2, 'valid');
    expect(useAnnotationStore.getState().splitBySlice.sample['2']).toBe('valid');

    useAnnotationStore.temporal.getState().undo();
    expect(useAnnotationStore.getState().splitBySlice.sample).toBeUndefined();

    useAnnotationStore.temporal.getState().redo();
    expect(useAnnotationStore.getState().splitBySlice.sample['2']).toBe('valid');
  });

  it('undoes and redoes negative-slice flags', () => {
    useAnnotationStore.getState().toggleNegativeSlice('sample', 4);
    expect(useAnnotationStore.getState().negativeSlices.sample).toEqual(['4']);

    useAnnotationStore.temporal.getState().undo();
    expect(useAnnotationStore.getState().negativeSlices.sample).toBeUndefined();

    useAnnotationStore.temporal.getState().redo();
    expect(useAnnotationStore.getState().negativeSlices.sample).toEqual(['4']);
  });
});
