/**
 * usePixelClassifier — covers branches NOT already exercised by
 * usePixelClassifier.slicePersist.test.ts (which only covers: model persists
 * across a slice change, and predict() targets the current slice's bank).
 *
 * Here: train() error paths (no composition, unknown-feature-bank recovery,
 * generic error), predict() guards + error path, proba class cycling +
 * threshold clamping, saveThresholdedClass, dismiss, trainAcrossSlices /
 * applyAcrossVolume (via the real useExportJob hook, driven through mocked
 * fetch responses so no fake timers are needed — the first status poll
 * always resolves 'done'/'error' immediately), the live volume-apply preview
 * effect keyed off predictedRasterStore, and ensureSession reuse.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { usePixelClassifier } from './usePixelClassifier';
import { useIpredStore } from '@/stores/ipredStore';
import { usePredictedRasterStore } from '@/stores/predictedRasterStore';

vi.mock('@/lib/pixelClf', () => ({
  thresholdProbaPngBlob: vi.fn(async () => new Blob()),
}));

vi.mock('@/lib/ipredApi', async () => {
  const actual = await vi.importActual<typeof import('@/lib/ipredApi')>('@/lib/ipredApi');
  return {
    ...actual,
    openIpredSession: vi.fn(async () => ({ session_id: 's1', project_id: 'p1' })),
    ipredPreprocess: vi.fn(async () => ({
      feature_id: 'bank-A',
      project_id: 'p1',
      setup_id: 'setup-1',
      slice_index: 0,
      n_channels: 3,
      height: 10,
      width: 10,
      labels: ['a'],
      cache_hit: false,
    })),
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
      feature_importances: [{ label: 'intensity', importance: 3 }],
    })),
    ipredInfer: vi.fn(async () => ({
      run_id: 'run-1',
      model_id: 'model-1',
      feature_id: 'bank-A',
      alpha: 0.05,
      class_ids: [1, 2],
      counts: { singleton: 1, multi: 0, abstain: 0 },
    })),
    ipredThresholdClass: vi.fn(async () => ({
      run_id: 'run-1',
      class_id: 1,
      class_index: 0,
      threshold: 0.5,
      width: 10,
      height: 10,
      n_positive: 5,
      label_map_b64: 'AAAA',
    })),
  };
});

import {
  openIpredSession,
  ipredPreprocess,
  ipredTrain,
  ipredInfer,
  ipredThresholdClass,
} from '@/lib/ipredApi';

const BASE_ARGS = {
  featureJobId: 'bank-A' as string | null,
  resetKey: 'sample-1' as string | null,
  source: 'sample.tif' as string | null,
  kind: 'local' as string | null,
  serverUri: null as string | null,
  sliceIndex: 0,
};

/** Routes fetch calls used by ipredRunCommitUrl/StatusUrl/ProbaUrl (blob PNGs)
 *  and by useExportJob's batch-train/apply start + status-poll routes. */
function makeFetchRouter(opts?: {
  batchTrainResult?: Record<string, unknown> | null;
  batchTrainError?: string;
  batchApplyResult?: Record<string, unknown> | null;
  batchApplyError?: string;
  pngOk?: boolean;
}) {
  const {
    batchTrainResult = null,
    batchTrainError,
    batchApplyResult = null,
    batchApplyError,
    pngOk = true,
  } = opts ?? {};
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes('/api/ipred/batch/train') && init?.method === 'POST') {
      return { ok: true, json: async () => ({ job_id: 'job-train-1' }) } as unknown as Response;
    }
    if (url.includes('/api/ipred/batch/apply') && init?.method === 'POST') {
      return { ok: true, json: async () => ({ job_id: 'job-apply-1' }) } as unknown as Response;
    }
    if (url.includes('/api/export/status/job-train-1')) {
      return {
        ok: true,
        json: async () =>
          batchTrainError
            ? { state: 'error', error: batchTrainError }
            : { state: 'done', result: batchTrainResult },
      } as unknown as Response;
    }
    if (url.includes('/api/export/status/job-apply-1')) {
      return {
        ok: true,
        json: async () =>
          batchApplyError
            ? { state: 'error', error: batchApplyError }
            : { state: 'done', result: batchApplyResult },
      } as unknown as Response;
    }
    // commit.png / status.png / proba/N.png
    return { ok: pngOk, blob: async () => new Blob() } as unknown as Response;
  });
}

beforeEach(() => {
  useIpredStore.getState().reset();
  usePredictedRasterStore.setState({ bySource: {} });
  vi.mocked(openIpredSession).mockClear();
  vi.mocked(ipredPreprocess).mockClear();
  vi.mocked(ipredTrain).mockClear();
  vi.mocked(ipredInfer).mockClear();
  vi.mocked(ipredThresholdClass).mockClear();
  global.fetch = makeFetchRouter();
  global.URL.createObjectURL = vi.fn(() => 'blob:mock');
  global.URL.revokeObjectURL = vi.fn();
});

describe('usePixelClassifier — train()', () => {
  it('sets an error and does not train without a preferred composition', async () => {
    useIpredStore.setState({ preferredCompositionId: '' });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(result.current.error).toBe('Select a composition first.');
    expect(ipredTrain).not.toHaveBeenCalled();
  });

  it('recovers from an "unknown feature" training error by clearing the model and flagging expiry', async () => {
    vi.mocked(ipredTrain).mockRejectedValueOnce(new Error('unknown feature bank xyz'));
    const onFeatureJobExpired = vi.fn();
    const { result } = renderHook(() =>
      usePixelClassifier({ ...BASE_ARGS, onFeatureJobExpired }),
    );
    await act(async () => {
      await result.current.train([]);
    });
    expect(onFeatureJobExpired).toHaveBeenCalledTimes(1);
    expect(result.current.model).toBeNull();
    expect(result.current.error).toMatch(/Feature bank missing/);
  });

  it('surfaces a generic training error message verbatim', async () => {
    vi.mocked(ipredTrain).mockRejectedValueOnce(new Error('trainer exploded'));
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(result.current.error).toBe('trainer exploded');
    expect(result.current.model).toBeNull();
  });

  it('trains successfully, mapping the ipred response into ClfTrainResult', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(result.current.model).toMatchObject({
      modelId: 'model-1',
      featureId: 'bank-A',
      nSamples: 120,
      nTrain: 100,
      nCal: 20,
      classIds: [1, 2],
      trainAccuracy: 0.9,
      nTrees: 50,
      usesSam: false,
      trainerId: 'catboost',
      compositionId: 'comp-skimage-slimsam',
    });
    expect(result.current.model?.featureImportances).toEqual([
      { label: 'intensity', importance: 3 },
    ]);
  });

  it('reuses an existing feature bank instead of calling ipredPreprocess', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(ipredPreprocess).not.toHaveBeenCalled();
    expect(ipredTrain).toHaveBeenCalledWith(
      expect.objectContaining({ feature_id: 'bank-A', session_id: 's1' }),
    );
  });

  it('auto-preprocesses and reports the new feature bank when no featureJobId is set', async () => {
    const onFeatureReady = vi.fn();
    const { result } = renderHook(() =>
      usePixelClassifier({ ...BASE_ARGS, featureJobId: null, onFeatureReady }),
    );
    await act(async () => {
      await result.current.train([]);
    });
    expect(ipredPreprocess).toHaveBeenCalledWith(
      expect.objectContaining({ session_id: 's1', composition_id: 'comp-skimage-slimsam' }),
    );
    expect(onFeatureReady).toHaveBeenCalledWith(
      expect.objectContaining({ featureId: 'bank-A', width: 10, height: 10 }),
    );
    expect(result.current.model?.modelId).toBe('model-1');
  });

  it('reuses an already-open ipred session rather than opening a new one', async () => {
    useIpredStore.getState().setIpredSession({ sessionId: 'existing-session', projectId: 'p9' });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(openIpredSession).not.toHaveBeenCalled();
    expect(ipredTrain).toHaveBeenCalledWith(
      expect.objectContaining({ session_id: 'existing-session' }),
    );
  });
});

describe('usePixelClassifier — predict()', () => {
  it('does nothing without a trained model', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.predict([]);
    });
    expect(ipredInfer).not.toHaveBeenCalled();
    expect(result.current.commitUrl).toBeNull();
  });

  it('predicts, publishes commit/status urls, counts, and the first proba channel', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
    expect(result.current.commitUrl).toBe('blob:mock');
    expect(result.current.statusUrl).toBe('blob:mock');
    expect(result.current.predictCounts).toEqual({ singleton: 1, multi: 0, abstain: 0 });
    expect(result.current.runId).toBe('run-1');
    expect(result.current.probaUrl).toBe('blob:mock');
    expect(result.current.activeProbaClassId).toBe(1);
  });

  it('revokes any partial preview and sets an error when the commit/status fetch fails', async () => {
    global.fetch = makeFetchRouter({ pngOk: false });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
    expect(result.current.error).toMatch(/Failed to fetch conformal prediction PNGs/);
    expect(result.current.commitUrl).toBeNull();
    expect(result.current.runId).toBeNull();
  });
});

describe('usePixelClassifier — proba class cycling + thresholds', () => {
  async function trainAndPredict(result: { current: ReturnType<typeof usePixelClassifier> }) {
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
  }

  it('selectProbaClass does nothing without a run/model', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.selectProbaClass(1);
    });
    expect(result.current.probaClassIndex).toBe(0);
  });

  it('cycleProbaClass wraps around the class list in both directions', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await trainAndPredict(result);
    expect(result.current.probaClassIndex).toBe(0);

    await act(async () => {
      result.current.cycleProbaClass(-1);
    });
    await waitFor(() => expect(result.current.probaClassIndex).toBe(1));
    expect(result.current.activeProbaClassId).toBe(2);

    await act(async () => {
      result.current.cycleProbaClass(1);
    });
    await waitFor(() => expect(result.current.probaClassIndex).toBe(0));
    expect(result.current.activeProbaClassId).toBe(1);
  });

  it('setProbaThreshold clamps to [0,1] and republishes the preview for the active class', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await trainAndPredict(result);

    await act(async () => {
      result.current.setProbaThreshold(1.5);
    });
    expect(result.current.activeProbaThreshold).toBe(1);

    await act(async () => {
      result.current.setProbaThreshold(-0.5);
    });
    expect(result.current.activeProbaThreshold).toBe(0);
  });

  it('setProbaThreshold is a no-op without a model', () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    act(() => {
      result.current.setProbaThreshold(0.7);
    });
    expect(result.current.activeProbaThreshold).toBe(0.5);
  });
});

describe('usePixelClassifier — saveThresholdedClass / dismiss', () => {
  it('returns null and does not call the API without an active run', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    let out;
    await act(async () => {
      out = await result.current.saveThresholdedClass();
    });
    expect(out).toBeNull();
    expect(ipredThresholdClass).not.toHaveBeenCalled();
  });

  it('calls ipredThresholdClass with the active class/threshold once predicted', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
    let out;
    await act(async () => {
      out = await result.current.saveThresholdedClass();
    });
    expect(ipredThresholdClass).toHaveBeenCalledWith('run-1', { class_id: 1, threshold: 0.5 });
    expect(out).toMatchObject({ run_id: 'run-1', class_id: 1 });
    expect(result.current.savingClass).toBe(false);
  });

  it('surfaces an error from a failed save without throwing', async () => {
    vi.mocked(ipredThresholdClass).mockRejectedValueOnce(new Error('save failed'));
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
    await act(async () => {
      await result.current.saveThresholdedClass();
    });
    expect(result.current.error).toBe('save failed');
  });

  it('dismiss() revokes the current prediction preview', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.predict([]);
    });
    expect(result.current.commitUrl).not.toBeNull();
    act(() => {
      result.current.dismiss();
    });
    expect(result.current.commitUrl).toBeNull();
    expect(result.current.runId).toBeNull();
  });
});

describe('usePixelClassifier — identity resets', () => {
  it('clears the trained model when resetKey (sample identity) changes', async () => {
    const { result, rerender } = renderHook(
      (props: { resetKey: string }) => usePixelClassifier({ ...BASE_ARGS, resetKey: props.resetKey }),
      { initialProps: { resetKey: 'sample-1' } },
    );
    await act(async () => {
      await result.current.train([]);
    });
    expect(result.current.model?.modelId).toBe('model-1');

    rerender({ resetKey: 'sample-2' });
    expect(result.current.model).toBeNull();
  });

  it('clears the trained model when preferredCompositionId changes', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    expect(result.current.model?.modelId).toBe('model-1');

    act(() => {
      useIpredStore.getState().setPreferredCompositionId('comp-other');
    });
    expect(result.current.model).toBeNull();
  });
});

describe('usePixelClassifier — trainAcrossSlices (multi-slice batch train)', () => {
  it('errors when there are no annotated slices', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.trainAcrossSlices({});
    });
    expect(result.current.error).toBe('No annotated slices to train on.');
  });

  it('errors without a preferred composition', async () => {
    useIpredStore.setState({ preferredCompositionId: '' });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.trainAcrossSlices({ 0: [] });
    });
    expect(result.current.error).toBe('Select a composition first.');
  });

  it('starts the batch-train job and adopts the completed result as the model', async () => {
    global.fetch = makeFetchRouter({
      batchTrainResult: {
        model_id: 'model-multi',
        feature_id: 'bank-multi',
        trainer_id: 'catboost',
        class_ids: [1, 2, 3],
        train_accuracy: 0.8,
        n_train: 300,
        n_cal: 60,
        n_samples: 360,
        params: { iterations: 200 },
        feature_importances: [],
      },
    });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.trainAcrossSlices({ 0: [], 5: [] });
    });
    await waitFor(() => expect(result.current.model?.modelId).toBe('model-multi'));
    expect(result.current.model?.classIds).toEqual([1, 2, 3]);
  });

  it('surfaces a batch-train job error', async () => {
    global.fetch = makeFetchRouter({ batchTrainError: 'multi-train blew up' });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.trainAcrossSlices({ 0: [], 5: [] });
    });
    await waitFor(() => expect(result.current.error).toBe('multi-train blew up'));
    expect(result.current.model).toBeNull();
  });
});

describe('usePixelClassifier — applyAcrossVolume (batch apply)', () => {
  it('errors without a trained model', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.applyAcrossVolume([0, 1, 2]);
    });
    expect(result.current.error).toBe('Train a model first.');
  });

  it('is a no-op with an empty slice list even with a model', async () => {
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.applyAcrossVolume([]);
    });
    expect(result.current.error).toBeNull();
    expect(result.current.volumeApplyJob.status).toBe('idle');
  });

  it('surfaces a batch-apply job error', async () => {
    global.fetch = makeFetchRouter({ batchApplyError: 'apply blew up' });
    const { result } = renderHook(() => usePixelClassifier(BASE_ARGS));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.applyAcrossVolume([0, 1]);
    });
    await waitFor(() => expect(result.current.error).toBe('apply blew up'));
  });

  it('fetches and shows the live per-slice preview once the job result names a run for the current slice', async () => {
    global.fetch = makeFetchRouter({ batchApplyResult: { runs: { '0': 'run-vol-1' } } });
    const { result } = renderHook(() => usePixelClassifier({ ...BASE_ARGS, sliceIndex: 0 }));
    await act(async () => {
      await result.current.train([]);
    });
    await act(async () => {
      await result.current.applyAcrossVolume([0, 1]);
    });
    await waitFor(() => expect(result.current.commitUrl).toBe('blob:mock'));
    expect(result.current.statusUrl).toBe('blob:mock');
  });

  it('shows the committed pointer preview from predictedRasterStore once the job map is gone', async () => {
    const { result } = renderHook(() => usePixelClassifier({ ...BASE_ARGS, resetKey: 'sample-9', sliceIndex: 3 }));
    await act(async () => {
      await result.current.train([]);
    });
    act(() => {
      usePredictedRasterStore.getState().setPointers('sample-9', {
        '3': { runId: 'run-committed-1', classIds: [1, 2] },
      });
    });
    await waitFor(() => expect(result.current.commitUrl).toBe('blob:mock'));
  });
});
