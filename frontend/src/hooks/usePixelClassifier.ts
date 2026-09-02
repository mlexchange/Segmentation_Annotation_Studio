/**
 * usePixelClassifier — train / conformal predict via ipred.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useConnectionStore } from '@/stores/connectionStore';
import { useIpredStore } from '@/stores/ipredStore';
import type { Shape } from '@/stores/annotationStore';
import { useExportJob } from '@/hooks/useExportJob';
import {
  ipredInfer,
  ipredPreprocess,
  ipredRunCommitUrl,
  ipredRunProbaUrl,
  ipredRunStatusUrl,
  ipredThresholdClass,
  ipredTrain,
  openIpredSession,
  type IpredThresholdClassResult,
  type IpredTrainResult,
} from '@/lib/ipredApi';
import { thresholdProbaPngBlob } from '@/lib/pixelClf';

export interface ClfParams {
  iterations: number;
  depth: number;
  learningRate: number;
  /** Misfire level α for conformal sets (e.g. 0.05 = 5%). */
  alpha: number;
}

export const DEFAULT_CLF_PARAMS: ClfParams = {
  iterations: 200,
  depth: 6,
  learningRate: 0.1,
  alpha: 0.05,
};

export interface ClfFeatureImportance {
  label: string;
  importance: number;
}

export interface ClfPredictCounts {
  singleton: number;
  multi: number;
  abstain: number;
}

export interface ClfTrainResult {
  modelId: string;
  featureId: string;
  nSamples: number;
  nTrain: number;
  nCal: number;
  classIds: number[];
  trainAccuracy: number;
  params: ClfParams;
  nTrees: number;
  usesSam: boolean;
  featureImportances: ClfFeatureImportance[];
  trainerId: string;
  compositionId: string | null;
}

export interface UsePixelClassifierArgs {
  /** Current ipred feature bank id (from Preprocess compute), if any. */
  featureJobId: string | null;
  /** Sample/composition identity ONLY (e.g. sourceKey) — must NOT include the
   *  slice index, or a trained model gets wiped on every slice change. */
  resetKey: string | null;
  source: string | null;
  kind: string | null;
  serverUri: string | null;
  sliceIndex: number;
  onFeatureJobExpired?: () => void;
  /** Called when train auto-runs preprocess and gets a new feature bank. */
  onFeatureReady?: (info: {
    featureId: string;
    width: number;
    height: number;
    labels: string[];
    nChannels: number;
    setupId: string;
    cacheHit: boolean;
  }) => void;
}

const DEFAULT_PROBA_THRESHOLD = 0.5;

/** Maps an ipred train response (single- or multi-slice — same shape, plus an
 *  optional `trained_slice_indices`) into the UI's ClfTrainResult. Shared so
 *  the multi-slice path doesn't duplicate `train()`'s mapping. */
function toClfTrainResult(
  data: IpredTrainResult,
  params: ClfParams,
  compositionId: string | null,
): ClfTrainResult {
  return {
    modelId: data.model_id,
    featureId: data.feature_id,
    nSamples: data.n_samples,
    nTrain: data.n_train,
    nCal: data.n_cal,
    classIds: data.class_ids,
    trainAccuracy: data.train_accuracy,
    params: { ...params },
    nTrees: Number(data.params?.iterations ?? params.iterations),
    usesSam: !!(data.params as { uses_sam?: boolean } | undefined)?.uses_sam,
    featureImportances: (data.feature_importances ?? []).map((fi) => ({
      label: fi.label,
      importance: fi.importance,
    })),
    trainerId: data.trainer_id,
    compositionId,
  };
}

export function usePixelClassifier({
  featureJobId,
  resetKey,
  source,
  kind,
  serverUri,
  sliceIndex,
  onFeatureJobExpired,
  onFeatureReady,
}: UsePixelClassifierArgs) {
  const preferredCompositionId = useIpredStore((s) => s.preferredCompositionId);
  const preferredTrainerId = useIpredStore((s) => s.preferredTrainerId);
  const preferredTrainerConfig = useIpredStore((s) => s.preferredTrainerConfig);
  const ipredSessionId = useIpredStore((s) => s.ipredSessionId);
  const setIpredSession = useIpredStore((s) => s.setIpredSession);
  const localRoot = useConnectionStore((s) => s.localRoot);

  const [params, setParams] = useState<ClfParams>(() => ({
    ...DEFAULT_CLF_PARAMS,
    iterations: preferredTrainerConfig.iterations,
    depth: preferredTrainerConfig.depth,
    learningRate: preferredTrainerConfig.learning_rate,
  }));
  const [model, setModel] = useState<ClfTrainResult | null>(null);
  const [commitUrl, setCommitUrl] = useState<string | null>(null);
  const [statusUrl, setStatusUrl] = useState<string | null>(null);
  const [predictCounts, setPredictCounts] = useState<ClfPredictCounts | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [probaClassIndex, setProbaClassIndex] = useState(0);
  const [probaThresholds, setProbaThresholds] = useState<Record<number, number>>({});
  const [probaUrl, setProbaUrl] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [savingClass, setSavingClass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const commitUrlRef = useRef<string | null>(null);
  const statusUrlRef = useRef<string | null>(null);
  const probaUrlRef = useRef<string | null>(null);
  /** run_id currently shown via the volume-apply live preview (see the effect
   * below) — cleared in revokePredict() so a slice revisit after any revoke
   * (slice change, sample change, job restart) always re-fetches rather than
   * skipping because "we already showed this run_id once" while commitUrl
   * itself has since gone back to null. */
  const volumeApplyPreviewRunIdRef = useRef<string | null>(null);
  /** Raw softmax PNG per class (before threshold preview). */
  const rawProbaBlobRef = useRef<Blob | null>(null);
  const probaThresholdsRef = useRef(probaThresholds);
  probaThresholdsRef.current = probaThresholds;
  const modelRef = useRef(model);
  modelRef.current = model;
  const onExpiredRef = useRef(onFeatureJobExpired);
  onExpiredRef.current = onFeatureJobExpired;
  const onFeatureReadyRef = useRef(onFeatureReady);
  onFeatureReadyRef.current = onFeatureReady;

  // Keep Train knobs aligned with ipred trainer defaults when they change.
  useEffect(() => {
    setParams((p) => ({
      ...p,
      iterations: preferredTrainerConfig.iterations,
      depth: preferredTrainerConfig.depth,
      learningRate: preferredTrainerConfig.learning_rate,
    }));
  }, [preferredTrainerConfig]);

  const revokeProba = useCallback(() => {
    if (probaUrlRef.current) {
      URL.revokeObjectURL(probaUrlRef.current);
      probaUrlRef.current = null;
    }
    setProbaUrl(null);
    rawProbaBlobRef.current = null;
  }, []);

  const publishProbaPreview = useCallback(async (raw: Blob, threshold: number) => {
    const preview = await thresholdProbaPngBlob(raw, threshold);
    const url = URL.createObjectURL(preview);
    if (probaUrlRef.current) URL.revokeObjectURL(probaUrlRef.current);
    probaUrlRef.current = url;
    setProbaUrl(url);
  }, []);

  const revokePredict = useCallback(() => {
    if (commitUrlRef.current) {
      URL.revokeObjectURL(commitUrlRef.current);
      commitUrlRef.current = null;
    }
    if (statusUrlRef.current) {
      URL.revokeObjectURL(statusUrlRef.current);
      statusUrlRef.current = null;
    }
    setCommitUrl(null);
    setStatusUrl(null);
    setPredictCounts(null);
    setRunId(null);
    setProbaClassIndex(0);
    setProbaThresholds({});
    volumeApplyPreviewRunIdRef.current = null;
    revokeProba();
  }, [revokeProba]);

  // Sample or composition identity changed — the trained model no longer applies.
  // `resetKey` is the sample's sourceKey alone (no slice index baked in), so a
  // plain slice change does NOT land here; see the effect below for that case.
  useEffect(() => {
    setModel(null);
    revokePredict();
    setError(null);
  }, [resetKey, preferredCompositionId, revokePredict]);

  // Feature bank changed (new slice, or a recompute) — any in-flight prediction
  // preview is tied to the OLD bank and must go, but the trained model itself
  // stays valid: it can be applied to whichever slice is on screen now (see
  // `predict()`, which always resolves the CURRENT slice's bank via
  // `ensureFeatureBank()` rather than the model's original training bank).
  useEffect(() => {
    revokePredict();
    setError(null);
  }, [featureJobId, revokePredict]);

  useEffect(
    () => () => {
      if (commitUrlRef.current) URL.revokeObjectURL(commitUrlRef.current);
      if (statusUrlRef.current) URL.revokeObjectURL(statusUrlRef.current);
      if (probaUrlRef.current) URL.revokeObjectURL(probaUrlRef.current);
    },
    [],
  );

  const ensureSession = useCallback(async (): Promise<string> => {
    if (ipredSessionId) return ipredSessionId;
    if (!source || !kind) throw new Error('No sample open');
    const session = await openIpredSession({
      kind,
      source,
      server_uri: serverUri,
      root: kind === 'local' ? localRoot : null,
    });
    setIpredSession({
      sessionId: session.session_id,
      projectId: session.project_id,
    });
    return session.session_id;
  }, [ipredSessionId, source, kind, serverUri, localRoot, setIpredSession]);

  const ensureFeatureBank = useCallback(async (): Promise<string> => {
    if (featureJobId) return featureJobId;
    if (!preferredCompositionId) {
      throw new Error('Select a composition first.');
    }
    const sessionId = await ensureSession();
    const bank = await ipredPreprocess({
      session_id: sessionId,
      composition_id: preferredCompositionId,
      slice_index: sliceIndex,
    });
    onFeatureReadyRef.current?.({
      featureId: bank.feature_id,
      width: bank.width,
      height: bank.height,
      labels: bank.labels ?? [],
      nChannels: bank.n_channels,
      setupId: bank.setup_id,
      cacheHit: bank.cache_hit,
    });
    return bank.feature_id;
  }, [featureJobId, preferredCompositionId, ensureSession, sliceIndex]);

  const loadProbaChannel = useCallback(
    async (run: string, classIndex: number) => {
      const res = await fetch(ipredRunProbaUrl(run, classIndex));
      if (!res.ok) throw new Error(`Failed to load class ${classIndex} probability map`);
      const blob = await res.blob();
      rawProbaBlobRef.current = blob;
      setProbaClassIndex(classIndex);
      const m = modelRef.current;
      const cid = m?.classIds[classIndex];
      const t =
        cid !== undefined
          ? (probaThresholdsRef.current[cid] ?? DEFAULT_PROBA_THRESHOLD)
          : DEFAULT_PROBA_THRESHOLD;
      await publishProbaPreview(blob, t);
    },
    [publishProbaPreview],
  );

  const train = useCallback(
    async (shapes: Shape[]) => {
      if (training) return;
      if (!preferredCompositionId) {
        setError('Select a composition first.');
        return;
      }
      setTraining(true);
      setError(null);
      revokePredict();
      try {
        const sessionId = await ensureSession();
        const featureId = await ensureFeatureBank();
        const data = await ipredTrain({
          session_id: sessionId,
          shapes,
          feature_id: featureId,
          trainer_id: preferredTrainerId,
          config: {
            iterations: params.iterations,
            depth: params.depth,
            learning_rate: params.learningRate,
          },
        });
        setModel(toClfTrainResult(data, params, preferredCompositionId));
      } catch (e) {
        setModel(null);
        const msg = e instanceof Error ? e.message : String(e);
        if (/unknown feature|not found/i.test(msg)) {
          onExpiredRef.current?.();
          setError('Feature bank missing. Compute or Train again (auto-preprocesses).');
        } else {
          setError(msg);
        }
      } finally {
        setTraining(false);
      }
    },
    [
      training,
      preferredCompositionId,
      preferredTrainerId,
      params,
      revokePredict,
      ensureSession,
      ensureFeatureBank,
    ],
  );

  // ---- Batch operations: multi-slice train, whole-volume apply ----
  const multiTrainJobHook = useExportJob();
  const volumeApplyJobHook = useExportJob();
  // Guards against re-applying an already-handled job result on every render
  // (the job's `state` object is recreated each poll tick even once done).
  const multiTrainHandledRef = useRef<string | null>(null);
  const volumeApplyHandledRef = useRef<string | null>(null);

  /** Train one model pooling labeled pixels across every slice in `perSliceShapes`. */
  const trainAcrossSlices = useCallback(
    async (perSliceShapes: Record<number, Shape[]>) => {
      if (Object.keys(perSliceShapes).length === 0) {
        setError('No annotated slices to train on.');
        return;
      }
      if (!preferredCompositionId) {
        setError('Select a composition first.');
        return;
      }
      setError(null);
      revokePredict();
      const sessionId = await ensureSession();
      multiTrainHandledRef.current = null;
      await multiTrainJobHook.startIpredBatchTrain({
        session_id: sessionId,
        slices: perSliceShapes,
        composition_id: preferredCompositionId,
        trainer_id: preferredTrainerId,
        config: {
          iterations: params.iterations,
          depth: params.depth,
          learning_rate: params.learningRate,
        },
      });
    },
    [
      preferredCompositionId,
      preferredTrainerId,
      params,
      revokePredict,
      ensureSession,
      multiTrainJobHook,
    ],
  );

  // Adopt the completed multi-train job's result the same way `train()` does.
  useEffect(() => {
    const { status, result, error: jobError, jobId } = multiTrainJobHook.state;
    if (!jobId || multiTrainHandledRef.current === jobId) return;
    if (status === 'done' && result) {
      multiTrainHandledRef.current = jobId;
      setModel(toClfTrainResult(result as unknown as IpredTrainResult, params, preferredCompositionId));
    } else if (status === 'error') {
      multiTrainHandledRef.current = jobId;
      setModel(null);
      setError(jobError ?? 'Multi-slice training failed.');
    }
  }, [multiTrainJobHook.state, params, preferredCompositionId]);

  /** Run inference across many slices (e.g. the whole volume). Does not commit —
   *  turning the result's per-slice runs into shapes stays client-side in
   *  AnnotatePage, reusing the same PNG-vectorize path as the single-slice commit. */
  const applyAcrossVolume = useCallback(
    async (sliceIndices: number[]) => {
      if (!model) {
        setError('Train a model first.');
        return;
      }
      if (sliceIndices.length === 0) return;
      setError(null);
      // Clear any prior job's frozen preview (single-slice OR a previous
      // volume apply) before starting — otherwise switching slices right
      // after kicking off a new run could briefly show a stale overlay left
      // over from before this job's own results start landing.
      revokePredict();
      const sessionId = await ensureSession();
      volumeApplyHandledRef.current = null;
      await volumeApplyJobHook.startIpredBatchApply({
        session_id: sessionId,
        model_id: model.modelId,
        slice_indices: sliceIndices,
        composition_id: preferredCompositionId,
        alpha: params.alpha,
      });
    },
    [model, preferredCompositionId, params.alpha, ensureSession, volumeApplyJobHook, revokePredict],
  );

  useEffect(() => {
    const { status, error: jobError, jobId } = volumeApplyJobHook.state;
    if (!jobId || volumeApplyHandledRef.current === jobId) return;
    if (status === 'error') {
      volumeApplyHandledRef.current = jobId;
      setError(jobError ?? 'Volume apply failed.');
    }
  }, [volumeApplyJobHook.state]);

  // Live per-slice preview during (or after) a volume-apply job: as soon as
  // ipred_batch_jobs.py's result.runs has an entry for whichever slice is
  // currently on screen, fetch and show that slice's commit/status overlay —
  // reusing the exact same commitUrl/statusUrl the single-slice "Predict"
  // button already drives, so AnnotationCanvas needs no new prop. Before this,
  // switching slices during/after a volume apply showed nothing at all until
  // the explicit "Commit" step vectorized everything into permanent shapes —
  // there was no cheap way to just look at a slice's predicted result first.
  //
  // NOT a proba-channel preview: batch-apply runs are created with
  // store_probabilities=false (ipred_batch_jobs.py's `_apply_one_slice`), so
  // there is no proba.npy to load for these run ids — only commit/status.
  useEffect(() => {
    const result = volumeApplyJobHook.state.result as { runs?: Record<string, string> } | null;
    const targetRunId = result?.runs?.[String(sliceIndex)] ?? null;
    if (!targetRunId || targetRunId === volumeApplyPreviewRunIdRef.current) return;
    let cancelled = false;
    volumeApplyPreviewRunIdRef.current = targetRunId;
    void (async () => {
      try {
        const [commitRes, statusRes] = await Promise.all([
          fetch(ipredRunCommitUrl(targetRunId)),
          fetch(ipredRunStatusUrl(targetRunId)),
        ]);
        if (!commitRes.ok || !statusRes.ok || cancelled) return;
        const [commitBlob, statusBlob] = await Promise.all([commitRes.blob(), statusRes.blob()]);
        if (cancelled) return;
        const cUrl = URL.createObjectURL(commitBlob);
        const sUrl = URL.createObjectURL(statusBlob);
        if (commitUrlRef.current) URL.revokeObjectURL(commitUrlRef.current);
        if (statusUrlRef.current) URL.revokeObjectURL(statusUrlRef.current);
        commitUrlRef.current = cUrl;
        statusUrlRef.current = sUrl;
        setCommitUrl(cUrl);
        setStatusUrl(sUrl);
      } catch {
        // Best-effort live preview only — a fetch hiccup here shouldn't
        // surface a hard error; the explicit Commit step remains the
        // authoritative path regardless of whether this preview loaded.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sliceIndex, volumeApplyJobHook.state.result]);

  const predict = useCallback(
    async (_shapes: Shape[]) => {
      if (!model || predicting) return;
      setPredicting(true);
      setError(null);
      try {
        const sessionId = await ensureSession();
        // Always the CURRENT slice's bank, not model.featureId (the slice the model
        // happened to be trained on) — the model persists across slices, so predict
        // must target whichever slice is on screen, computing a bank if missing.
        const featureId = await ensureFeatureBank();
        const run = await ipredInfer({
          session_id: sessionId,
          model_id: model.modelId,
          feature_id: featureId,
          alpha: params.alpha,
        });
        const [commitRes, statusRes] = await Promise.all([
          fetch(ipredRunCommitUrl(run.run_id)),
          fetch(ipredRunStatusUrl(run.run_id)),
        ]);
        if (!commitRes.ok || !statusRes.ok) {
          throw new Error('Failed to fetch conformal prediction PNGs');
        }
        const [commitBlob, statusBlob] = await Promise.all([commitRes.blob(), statusRes.blob()]);
        const cUrl = URL.createObjectURL(commitBlob);
        const sUrl = URL.createObjectURL(statusBlob);
        if (commitUrlRef.current) URL.revokeObjectURL(commitUrlRef.current);
        if (statusUrlRef.current) URL.revokeObjectURL(statusUrlRef.current);
        commitUrlRef.current = cUrl;
        statusUrlRef.current = sUrl;
        setCommitUrl(cUrl);
        setStatusUrl(sUrl);
        setPredictCounts(run.counts);
        setRunId(run.run_id);
        const classIds = run.class_ids?.length ? run.class_ids : model.classIds;
        setModel((m) => (m ? { ...m, classIds } : m));
        const thresholds: Record<number, number> = {};
        for (const cid of classIds) thresholds[cid] = DEFAULT_PROBA_THRESHOLD;
        setProbaThresholds(thresholds);
        await loadProbaChannel(run.run_id, 0);
      } catch (e) {
        revokePredict();
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPredicting(false);
      }
    },
    [model, predicting, params.alpha, ensureSession, ensureFeatureBank, revokePredict, loadProbaChannel],
  );

  const selectProbaClass = useCallback(
    async (index: number) => {
      if (!runId || !model) return;
      const n = model.classIds.length;
      if (n < 1) return;
      const next = ((index % n) + n) % n;
      setError(null);
      try {
        await loadProbaChannel(runId, next);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [runId, model, loadProbaChannel],
  );

  const cycleProbaClass = useCallback(
    (delta: number) => {
      void selectProbaClass(probaClassIndex + delta);
    },
    [selectProbaClass, probaClassIndex],
  );

  const setProbaThreshold = useCallback(
    (threshold: number) => {
      if (!model) return;
      const classId = model.classIds[probaClassIndex];
      if (classId === undefined) return;
      const t = Math.min(1, Math.max(0, threshold));
      setProbaThresholds((prev) => ({ ...prev, [classId]: t }));
      const raw = rawProbaBlobRef.current;
      if (raw) {
        void publishProbaPreview(raw, t).catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
        });
      }
    },
    [model, probaClassIndex, publishProbaPreview],
  );

  const activeProbaClassId = model?.classIds[probaClassIndex] ?? null;
  const activeProbaThreshold =
    activeProbaClassId !== null
      ? (probaThresholds[activeProbaClassId] ?? DEFAULT_PROBA_THRESHOLD)
      : DEFAULT_PROBA_THRESHOLD;

  const saveThresholdedClass = useCallback(async (): Promise<IpredThresholdClassResult | null> => {
    if (!runId || activeProbaClassId === null || savingClass) return null;
    setSavingClass(true);
    setError(null);
    try {
      return await ipredThresholdClass(runId, {
        class_id: activeProbaClassId,
        threshold: activeProbaThreshold,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setSavingClass(false);
    }
  }, [runId, activeProbaClassId, activeProbaThreshold, savingClass]);

  const dismiss = useCallback(() => {
    revokePredict();
  }, [revokePredict]);

  return {
    params,
    setParams,
    model,
    commitUrl,
    statusUrl,
    predictUrl: commitUrl,
    predictCounts,
    runId,
    probaUrl,
    probaClassIndex,
    activeProbaClassId,
    activeProbaThreshold,
    selectProbaClass,
    cycleProbaClass,
    setProbaThreshold,
    saveThresholdedClass,
    savingClass,
    training,
    predicting,
    error,
    train,
    predict,
    dismiss,
    compositionId: preferredCompositionId,
    trainerId: preferredTrainerId,
    canTrainWithoutJob: !!preferredCompositionId && !!source && !!kind,
    // ---- Batch operations ----
    trainAcrossSlices,
    multiTrainJob: multiTrainJobHook.state,
    multiTraining: multiTrainJobHook.state.status === 'running',
    resetMultiTrainJob: multiTrainJobHook.reset,
    applyAcrossVolume,
    volumeApplyJob: volumeApplyJobHook.state,
    volumeApplying: volumeApplyJobHook.state.status === 'running',
    resetVolumeApplyJob: volumeApplyJobHook.reset,
  };
}
