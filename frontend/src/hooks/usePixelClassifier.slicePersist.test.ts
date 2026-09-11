/**
 * A trained model must survive a slice change (only sample/composition identity
 * resets it), and predict() must target the CURRENT slice's feature bank rather
 * than the bank the model happened to be trained on.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { usePixelClassifier } from './usePixelClassifier';
import { useIpredStore } from '@/stores/ipredStore';

vi.mock('@/lib/pixelClf', () => ({
  thresholdProbaPngBlob: vi.fn(async () => new Blob()),
}));

vi.mock('@/lib/ipredApi', async () => {
  const actual = await vi.importActual<typeof import('@/lib/ipredApi')>('@/lib/ipredApi');
  return {
    ...actual,
    openIpredSession: vi.fn(async () => ({ session_id: 's1', project_id: 'p1' })),
    ipredPreprocess: vi.fn(async () => {
      throw new Error('ensureFeatureBank should not need to preprocess in this test');
    }),
    ipredTrain: vi.fn(async () => ({
      model_id: 'model-1',
      feature_id: 'bank-A',
      trainer_id: 'catboost',
      class_ids: [1, 2],
      train_accuracy: 0.9,
      n_train: 100,
      n_cal: 20,
      n_samples: 120,
      params: { iterations: 50 },
      feature_importances: [],
    })),
    ipredInfer: vi.fn(async (payload: { feature_id?: string | null }) => ({
      run_id: 'run-1',
      model_id: 'model-1',
      feature_id: payload.feature_id ?? '',
      alpha: 0.05,
      class_ids: [1, 2],
      counts: { singleton: 1, multi: 0, abstain: 0 },
    })),
  };
});

import { ipredInfer } from '@/lib/ipredApi';

describe('usePixelClassifier — model persists across slice changes', () => {
  beforeEach(() => {
    useIpredStore.getState().reset();
    vi.mocked(ipredInfer).mockClear();
    global.fetch = vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(),
    })) as unknown as typeof fetch;
    global.URL.createObjectURL = vi.fn(() => 'blob:mock');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('keeps the model when only featureJobId changes, and predict() uses the current bank', async () => {
    const { result, rerender } = renderHook(
      (props: { featureJobId: string | null; sliceIndex: number }) =>
        usePixelClassifier({
          featureJobId: props.featureJobId,
          resetKey: 'sample-1', // sourceKey only — does NOT change across slices
          source: 'sample.tif',
          kind: 'local',
          serverUri: null,
          sliceIndex: props.sliceIndex,
        }),
      { initialProps: { featureJobId: 'bank-A', sliceIndex: 5 } },
    );

    await act(async () => {
      await result.current.train([]);
    });
    await waitFor(() => expect(result.current.model?.modelId).toBe('model-1'));
    expect(result.current.model?.featureId).toBe('bank-A');

    // Simulate switching to a different slice: featureJobId changes to that
    // slice's own feature bank, resetKey (sourceKey) does not change.
    rerender({ featureJobId: 'bank-B', sliceIndex: 8 });

    // The model must still be there — only the run/preview state resets.
    expect(result.current.model?.modelId).toBe('model-1');

    await act(async () => {
      await result.current.predict([]);
    });
    await waitFor(() => expect(result.current.commitUrl).not.toBeNull());

    // predict() must have targeted the CURRENT slice's bank (bank-B), not the
    // bank the model was originally trained against (bank-A).
    expect(ipredInfer).toHaveBeenCalledWith(
      expect.objectContaining({ model_id: 'model-1', feature_id: 'bank-B' }),
    );
  });
});
