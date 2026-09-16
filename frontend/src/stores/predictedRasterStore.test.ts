import { describe, it, expect, beforeEach } from 'vitest';
import { usePredictedRasterStore } from './predictedRasterStore';

const reset = () => usePredictedRasterStore.setState({ bySource: {} });

describe('predictedRasterStore', () => {
  beforeEach(reset);

  it('setPointers merges into an existing source without clobbering other slices', () => {
    const { setPointers } = usePredictedRasterStore.getState();
    setPointers('sampleA', { '0': { runId: 'run-0', classIds: [1, 2] } });
    setPointers('sampleA', { '1': { runId: 'run-1', classIds: [1, 2] } });
    expect(usePredictedRasterStore.getState().bySource.sampleA).toEqual({
      '0': { runId: 'run-0', classIds: [1, 2] },
      '1': { runId: 'run-1', classIds: [1, 2] },
    });
  });

  it('setPointers for the same slice overwrites the prior pointer', () => {
    const { setPointers } = usePredictedRasterStore.getState();
    setPointers('sampleA', { '0': { runId: 'run-old', classIds: [1] } });
    setPointers('sampleA', { '0': { runId: 'run-new', classIds: [1, 2] } });
    expect(usePredictedRasterStore.getState().bySource.sampleA['0']).toEqual({
      runId: 'run-new',
      classIds: [1, 2],
    });
  });

  it('clearSlice removes only the targeted slice', () => {
    const { setPointers, clearSlice } = usePredictedRasterStore.getState();
    setPointers('sampleA', {
      '0': { runId: 'run-0', classIds: [1] },
      '1': { runId: 'run-1', classIds: [1] },
    });
    clearSlice('sampleA', '0');
    const bySource = usePredictedRasterStore.getState().bySource.sampleA;
    expect(bySource['0']).toBeUndefined();
    expect(bySource['1']).toEqual({ runId: 'run-1', classIds: [1] });
  });

  it('clearSlice on an unknown sample or slice is a no-op', () => {
    const { clearSlice } = usePredictedRasterStore.getState();
    expect(() => clearSlice('unknown', '0')).not.toThrow();
    expect(usePredictedRasterStore.getState().bySource).toEqual({});
  });

  it('clearSource drops every pointer for that sample, leaving others intact', () => {
    const { setPointers, clearSource } = usePredictedRasterStore.getState();
    setPointers('sampleA', { '0': { runId: 'run-0', classIds: [1] } });
    setPointers('sampleB', { '0': { runId: 'run-b0', classIds: [1] } });
    clearSource('sampleA');
    const state = usePredictedRasterStore.getState();
    expect(state.bySource.sampleA).toBeUndefined();
    expect(state.bySource.sampleB).toEqual({ '0': { runId: 'run-b0', classIds: [1] } });
  });
});
