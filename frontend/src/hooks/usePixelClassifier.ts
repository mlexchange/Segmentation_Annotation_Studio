/**
 * usePixelClassifier — train / conformal predict via ipred.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useConnectionStore } from '@/stores/connectionStore';
import { useIpredStore } from '@/stores/ipredStore';
import type { Shape } from '@/stores/annotationStore';
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
    revokeProba();
  }, [revokeProba]);

  useEffect(() => {
    setModel(null);
    revokePredict();
    setError(null);
  }, [featureJobId, resetKey, preferredCompositionId, revokePredict]);

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
        const result: ClfTrainResult = {
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
          compositionId: preferredCompositionId,
        };
        setModel(result);
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

  const predict = useCallback(
    async (_shapes: Shape[]) => {
      if (!model || predicting) return;
      setPredicting(true);
      setError(null);
      try {
        const sessionId = await ensureSession();
        const featureId = model.featureId || (await ensureFeatureBank());
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
  };
}
