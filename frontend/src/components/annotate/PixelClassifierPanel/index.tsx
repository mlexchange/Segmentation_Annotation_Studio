/**
 * PixelClassifierPanel — train / conformal-predict / commit, with the
 * Suggest-Labels manifold loop folded in and conformal coverage surfaced as
 * headline guidance (not a legend entry).
 */
import { CaretLeft, CaretRight, CircleNotch, Compass, TreeStructure } from '@phosphor-icons/react';
import type { ClfParams, ClfPredictCounts, ClfTrainResult } from '@/hooks/usePixelClassifier';
import type { ManifoldParams } from '@/hooks/useFeatureManifold';
import { cn } from '@/lib/utils';
import CollapsibleSection from '@/components/common/CollapsibleSection';

/** Shared progress-bar treatment (see useExportJob / DownloadModal). */
function BusyBar({ label }: { label: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-[10px] text-gray-500">
        <span>{label}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
        <div className="h-full w-1/3 animate-pulse rounded-full bg-sky-500" />
      </div>
    </div>
  );
}

/** Real done/total progress bar for batch jobs (see DownloadModal's job bar). */
function JobProgressBar({ label, done, total }: { label: string; done: number; total: number }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-[10px] text-gray-500">
        <span>{label}</span>
        {total > 0 && <span className="tabular-nums">{done}/{total}</span>}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-sky-500 transition-all"
          style={{ width: `${Math.max(5, pct)}%` }}
        />
      </div>
    </div>
  );
}

/** Coverage guidance derived from conformal set counts — the headline, not a legend line. */
function coverageHeadline(counts: ClfPredictCounts, alpha: number): { text: string; tone: 'good' | 'warn' } {
  const total = counts.singleton + counts.multi + counts.abstain;
  if (total === 0) return { text: 'No predicted pixels yet.', tone: 'warn' };
  const confidentPct = Math.round((counts.singleton / total) * 100);
  const alphaPct = Math.round(alpha * 100);
  if (confidentPct >= 70) {
    return {
      text: `${confidentPct}% of pixels are confidently labeled at ${100 - alphaPct}% coverage.`,
      tone: 'good',
    };
  }
  return {
    text: `Only ${confidentPct}% confidently labeled — most pixels need more training data or a lower α.`,
    tone: 'warn',
  };
}

export interface PixelClassifierPanelProps {
  hasFeatureJob: boolean;
  canAutoPreprocess?: boolean;
  hasShapes: boolean;
  training: boolean;
  predicting: boolean;
  params: ClfParams;
  onParamsChange: (p: ClfParams) => void;
  model: ClfTrainResult | null;
  hasPrediction: boolean;
  predictCounts: ClfPredictCounts | null;
  probaClassIndex?: number;
  activeProbaClassId?: number | null;
  activeProbaThreshold?: number;
  classLabelForId?: (classId: number) => string;
  onCycleProbaClass?: (delta: number) => void;
  onProbaThresholdChange?: (threshold: number) => void;
  error: string | null;
  onTrain: () => void;
  onPredict: () => void;
  onCommit: () => void;
  onDismiss: () => void;
  commitLabel?: string;
  featureSetupId?: string | null;
  trainerId?: string | null;

  // Multi-slice train: pool labeled pixels across every annotated slice of the sample.
  annotatedSliceCount: number;
  trainAcrossSlices: boolean;
  onTrainAcrossSlicesChange: (v: boolean) => void;
  multiTraining: boolean;
  multiTrainProgress: { done: number; total: number } | null;

  // Apply a trained model across many slices at once, then commit selected classes.
  totalSliceCount: number;
  commitClassIds: number[];
  onToggleCommitClassId: (classId: number) => void;
  volumeApplying: boolean;
  volumeApplyProgress: { done: number; total: number } | null;
  volumeApplyResult: { runCount: number; errorCount: number; cancelled: boolean } | null;
  volumeApplyCommitting: boolean;
  onApplyToVolume: () => void;
  onCommitVolumeApply: () => void;
  onCancelVolumeApply: () => void;
  onDismissVolumeApply: () => void;

  // Suggest-labels (manifold), folded into the same loop instead of a separate panel.
  manifoldParams: ManifoldParams;
  onManifoldParamsChange: (p: ManifoldParams) => void;
  manifoldSampling: boolean;
  manifoldHasSample: boolean;
  manifoldShowHeatmap: boolean;
  onManifoldShowHeatmapChange: (v: boolean) => void;
  manifoldShowMarkers: boolean;
  onManifoldShowMarkersChange: (v: boolean) => void;
  manifoldHeatmapOpacity: number;
  onManifoldHeatmapOpacityChange: (v: number) => void;
  manifoldMeta: { nPicked: number; nSubsample: number; explainedVariance: number } | null;
  manifoldError: string | null;
  onManifoldSample: () => void;
  onManifoldDismiss: () => void;
  manifoldRoiShapeCount: number;
  canCaptureManifoldRoi: boolean;
  onCaptureManifoldRoi: () => void;
  onClearManifoldRoi: () => void;
}

/** Sidebar controls for CatBoost + split-conformal sets, with suggest-labels folded in. */
export default function PixelClassifierPanel({
  hasFeatureJob,
  canAutoPreprocess = false,
  hasShapes,
  training,
  predicting,
  params,
  onParamsChange,
  model,
  hasPrediction,
  predictCounts,
  probaClassIndex = 0,
  activeProbaClassId = null,
  activeProbaThreshold = 0.5,
  classLabelForId,
  onCycleProbaClass,
  onProbaThresholdChange,
  error,
  onTrain,
  onPredict,
  onCommit,
  onDismiss,
  commitLabel = 'Commit singletons',
  featureSetupId = null,
  trainerId = null,
  annotatedSliceCount,
  trainAcrossSlices,
  onTrainAcrossSlicesChange,
  multiTraining,
  multiTrainProgress,
  totalSliceCount,
  commitClassIds,
  onToggleCommitClassId,
  volumeApplying,
  volumeApplyProgress,
  volumeApplyResult,
  volumeApplyCommitting,
  onApplyToVolume,
  onCommitVolumeApply,
  onCancelVolumeApply,
  onDismissVolumeApply,
  manifoldParams,
  onManifoldParamsChange,
  manifoldSampling,
  manifoldHasSample,
  manifoldShowHeatmap,
  onManifoldShowHeatmapChange,
  manifoldShowMarkers,
  onManifoldShowMarkersChange,
  manifoldHeatmapOpacity,
  onManifoldHeatmapOpacityChange,
  manifoldMeta,
  manifoldError,
  onManifoldSample,
  onManifoldDismiss,
  manifoldRoiShapeCount,
  canCaptureManifoldRoi,
  onCaptureManifoldRoi,
  onClearManifoldRoi,
}: PixelClassifierPanelProps) {
  const busy = training || predicting || multiTraining || volumeApplying;
  const canTrain =
    (trainAcrossSlices ? annotatedSliceCount > 0 : (hasFeatureJob || canAutoPreprocess) && hasShapes) &&
    !busy;
  // ensureFeatureBank() (inside predict()) computes a bank for the current slice
  // when one isn't ready yet, so predicting doesn't require hasFeatureJob up front —
  // only that a model exists to apply.
  const canPredict = !!model && !busy;
  const maxImp = model?.featureImportances[0]?.importance ?? 1;
  const alphaPct = Math.round(params.alpha * 100);
  const nClasses = model?.classIds.length ?? 0;
  const threshPct = Math.round(activeProbaThreshold * 100);
  const probaLabel =
    activeProbaClassId !== null
      ? (classLabelForId?.(activeProbaClassId) ?? `class ${activeProbaClassId}`)
      : '—';
  const headline = predictCounts ? coverageHeadline(predictCounts, params.alpha) : null;

  return (
    <CollapsibleSection
      title="Classifier"
      headerRight={
        <span className="text-[10px] text-gray-400" title="Trainer + Mondrian conformal sets">
          {trainerId ?? 'catboost'} · conformal
        </span>
      }
    >
      <p className="text-[11px] text-gray-600 leading-snug">
        Setup:{' '}
        <span className="font-mono text-gray-800">
          {featureSetupId ?? 'none — pick a recipe under Features'}
        </span>
        {!hasFeatureJob && canAutoPreprocess ? (
          <span className="text-gray-500"> · Train will compute features</span>
        ) : null}
      </p>

      <div className="grid grid-cols-3 gap-1 text-[10px] text-gray-600">
        <label className="flex flex-col gap-0.5">
          Trees
          <input
            type="number"
            min={10}
            max={2000}
            step={10}
            value={params.iterations}
            disabled={busy}
            onChange={(e) =>
              onParamsChange({ ...params, iterations: Math.max(1, Number(e.target.value) || 200) })
            }
            className="rounded border border-gray-200 px-1 py-0.5 text-gray-800"
          />
        </label>
        <label className="flex flex-col gap-0.5">
          Depth
          <input
            type="number"
            min={1}
            max={10}
            step={1}
            value={params.depth}
            disabled={busy}
            onChange={(e) =>
              onParamsChange({ ...params, depth: Math.min(10, Math.max(1, Number(e.target.value) || 6)) })
            }
            className="rounded border border-gray-200 px-1 py-0.5 text-gray-800"
          />
        </label>
        <label className="flex flex-col gap-0.5">
          LR
          <input
            type="number"
            min={0.01}
            max={1}
            step={0.01}
            value={params.learningRate}
            disabled={busy}
            onChange={(e) =>
              onParamsChange({
                ...params,
                learningRate: Math.min(1, Math.max(0.01, Number(e.target.value) || 0.1)),
              })
            }
            className="rounded border border-gray-200 px-1 py-0.5 text-gray-800"
          />
        </label>
      </div>

      <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
        <span className="flex justify-between">
          <span>α (misfire)</span>
          <span className="text-gray-800 font-medium">{alphaPct}%</span>
        </span>
        <input
          type="range"
          min={1}
          max={20}
          step={1}
          value={alphaPct}
          disabled={busy}
          onChange={(e) =>
            onParamsChange({
              ...params,
              alpha: Math.min(0.2, Math.max(0.01, Number(e.target.value) / 100)),
            })
          }
          className="w-full accent-sky-600"
        />
      </label>

      <label
        className={cn(
          'flex items-center gap-1.5 text-[10px]',
          annotatedSliceCount > 1 ? 'text-gray-600' : 'text-gray-300 cursor-not-allowed',
        )}
        title={
          annotatedSliceCount > 1
            ? `Pool labeled pixels across all ${annotatedSliceCount} annotated slices into one model`
            : 'Annotate more than one slice to enable this'
        }
      >
        <input
          type="checkbox"
          disabled={annotatedSliceCount <= 1 || busy}
          checked={trainAcrossSlices && annotatedSliceCount > 1}
          onChange={(e) => onTrainAcrossSlicesChange(e.target.checked)}
        />
        Train across all annotated slices ({annotatedSliceCount})
      </label>

      <button
        type="button"
        disabled={!canTrain}
        title={
          trainAcrossSlices
            ? 'Train one model pooling every annotated slice'
            : !hasFeatureJob ? 'Compute features first' : !hasShapes ? 'Annotate at least two classes' : 'Train classifier'
        }
        onClick={onTrain}
        className={cn(
          'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
          canTrain
            ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
        )}
      >
        <TreeStructure size={14} />
        {trainAcrossSlices ? 'Train across slices' : 'Train classifier'}
      </button>
      {training && <BusyBar label="Training…" />}
      {multiTraining && (
        <JobProgressBar
          label="Training across slices…"
          done={multiTrainProgress?.done ?? 0}
          total={multiTrainProgress?.total ?? 0}
        />
      )}

      <button
        type="button"
        disabled={!canPredict}
        onClick={onPredict}
        className={cn(
          'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
          canPredict
            ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
        )}
      >
        Predict
      </button>
      {predicting && <BusyBar label="Predicting…" />}

      {model && (
        <div className="flex flex-col gap-1.5 rounded-md border border-gray-100 bg-gray-50 p-1.5">
          <p className="text-[10px] text-gray-700">
            Acc {(model.trainAccuracy * 100).toFixed(1)}% · train {model.nTrain.toLocaleString()} /
            cal {model.nCal.toLocaleString()} · {model.nTrees} trees · classes{' '}
            {model.classIds.join(', ')}
            {model.usesSam ? ' · +SlimSAM' : ''}
          </p>

          {model.featureImportances.length > 0 && (
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] font-medium text-gray-600">Feature importance</span>
              {model.featureImportances.slice(0, 8).map((fi) => (
                <div key={fi.label} className="flex items-center gap-1 text-[10px]">
                  <span className="w-[42%] truncate text-gray-600" title={fi.label}>{fi.label}</span>
                  <div className="flex-1 h-1.5 rounded bg-gray-200 overflow-hidden">
                    <div
                      className="h-full bg-sky-500"
                      style={{ width: `${Math.max(2, (fi.importance / maxImp) * 100)}%` }}
                    />
                  </div>
                  <span className="w-8 text-right text-gray-500">{fi.importance.toFixed(1)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {model && (
        <div className="flex flex-col gap-1.5 rounded-md border border-gray-100 bg-gray-50 p-1.5">
          <span className="text-[10px] font-medium text-gray-600">Classes to commit</span>
          <div className="flex flex-wrap gap-1">
            {model.classIds.map((cid) => {
              const on = commitClassIds.includes(cid);
              return (
                <button
                  key={cid}
                  type="button"
                  onClick={() => onToggleCommitClassId(cid)}
                  className={cn(
                    'rounded border px-1.5 py-0.5 text-[9px]',
                    on
                      ? 'border-sky-300 bg-sky-50 text-sky-900'
                      : 'border-gray-200 bg-white text-gray-400',
                  )}
                >
                  {classLabelForId?.(cid) ?? `class ${cid}`}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            disabled={busy || totalSliceCount <= 1 || commitClassIds.length === 0}
            onClick={onApplyToVolume}
            title={`Predict across all ${totalSliceCount} slices, then choose which classes to commit`}
            className={cn(
              'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
              !busy && totalSliceCount > 1 && commitClassIds.length > 0
                ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
                : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
            )}
          >
            Apply across volume ({totalSliceCount} slices)
          </button>

          {volumeApplying && (
            <div className="flex flex-col gap-1">
              <JobProgressBar
                label="Applying across volume…"
                done={volumeApplyProgress?.done ?? 0}
                total={volumeApplyProgress?.total ?? 0}
              />
              <button
                type="button"
                onClick={onCancelVolumeApply}
                className="self-start text-[10px] text-gray-500 hover:text-red-600"
              >
                Cancel
              </button>
            </div>
          )}

          {volumeApplyResult && !volumeApplying && (
            <div className="flex flex-col gap-1 rounded border border-sky-100 bg-sky-50/60 p-1.5 text-[10px] text-gray-700">
              <p>
                {volumeApplyResult.cancelled ? 'Cancelled — ' : ''}
                Predicted {volumeApplyResult.runCount} slice(s)
                {volumeApplyResult.errorCount > 0 ? `, ${volumeApplyResult.errorCount} failed` : ''}.
              </p>
              <div className="flex gap-1">
                <button
                  type="button"
                  disabled={volumeApplyCommitting || volumeApplyResult.runCount === 0}
                  onClick={onCommitVolumeApply}
                  className="flex-1 py-1 rounded-md text-[10px] font-medium bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-50"
                >
                  {volumeApplyCommitting
                    ? 'Committing…'
                    : `Commit predicted shapes (${commitClassIds.length} class${commitClassIds.length === 1 ? '' : 'es'})`}
                </button>
                <button
                  type="button"
                  onClick={onDismissVolumeApply}
                  className="flex-1 py-1 rounded-md text-[10px] border border-gray-200 text-gray-700 hover:bg-gray-50"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {hasPrediction && headline && (
        <div
          className={cn(
            'rounded-md border p-2 text-[11px] font-medium leading-snug',
            headline.tone === 'good'
              ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
              : 'border-amber-200 bg-amber-50 text-amber-800',
          )}
        >
          {headline.text}
          {predictCounts && (
            <p className="mt-0.5 text-[10px] font-normal opacity-80">
              singleton {predictCounts.singleton.toLocaleString()} · multi{' '}
              {predictCounts.multi.toLocaleString()} · abstain {predictCounts.abstain.toLocaleString()}
            </p>
          )}
        </div>
      )}

      {hasPrediction && nClasses > 0 && onCycleProbaClass && onProbaThresholdChange && (
        <div className="flex flex-col gap-1.5 rounded-md border border-sky-100 bg-sky-50/60 p-1.5">
          <div className="flex items-center justify-between gap-1">
            <span className="text-[10px] font-medium text-gray-700">Class probability</span>
            <span className="text-[10px] text-gray-500">
              {probaClassIndex + 1}/{nClasses}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={busy || nClasses < 2}
              onClick={() => onCycleProbaClass(-1)}
              className="p-1 rounded border border-gray-200 bg-white text-gray-700 hover:bg-sky-50 disabled:opacity-40"
              title="Previous class"
            >
              <CaretLeft size={14} />
            </button>
            <span className="flex-1 text-center text-[11px] font-medium text-gray-800 truncate" title={probaLabel}>
              {probaLabel}
            </span>
            <button
              type="button"
              disabled={busy || nClasses < 2}
              onClick={() => onCycleProbaClass(1)}
              className="p-1 rounded border border-gray-200 bg-white text-gray-700 hover:bg-sky-50 disabled:opacity-40"
              title="Next class"
            >
              <CaretRight size={14} />
            </button>
          </div>
          <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
            <span className="flex justify-between">
              <span>Threshold</span>
              <span className="text-gray-800 font-medium">{threshPct}%</span>
            </span>
            <input
              type="range"
              min={5}
              max={95}
              step={1}
              value={threshPct}
              disabled={busy}
              onChange={(e) => onProbaThresholdChange(Number(e.target.value) / 100)}
              className="w-full accent-sky-600"
            />
          </label>
        </div>
      )}

      {hasPrediction && (
        <div className="flex gap-1">
          <button
            type="button"
            onClick={onCommit}
            title="Add singleton polygons only; keep existing annotations"
            className="flex-1 py-1.5 rounded-md text-xs font-medium bg-sky-600 text-white hover:bg-sky-700"
          >
            {commitLabel}
          </button>
          <button
            type="button"
            onClick={onDismiss}
            className="flex-1 py-1.5 rounded-md text-xs border border-gray-200 text-gray-700 hover:bg-gray-50"
          >
            Dismiss
          </button>
        </div>
      )}

      {error && <p className="text-[10px] text-red-600 leading-snug break-words">{error}</p>}

      {/* Suggest labels — folded into the same train/infer loop instead of its own panel. */}
      <div className="border-t border-gray-100 pt-1.5">
        <CollapsibleSection
          title="Suggest labels (manifold coverage)"
          icon={<Compass size={12} />}
          defaultOpen={false}
          disabled={!hasFeatureJob}
        >
          <div className="flex flex-col gap-1.5 rounded-md border border-gray-100 bg-gray-50 p-1.5">
            <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-600">
              <label className="flex flex-col gap-0.5">
                K boxes
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={manifoldParams.k}
                  disabled={manifoldSampling}
                  onChange={(e) =>
                    onManifoldParamsChange({ ...manifoldParams, k: Math.max(1, Number(e.target.value) || 24) })
                  }
                  className="rounded border border-gray-200 px-1 py-0.5 text-gray-800"
                />
              </label>
              <label className="flex flex-col gap-0.5">
                Box size
                <input
                  type="number"
                  min={8}
                  max={512}
                  step={8}
                  value={manifoldParams.boxSize}
                  disabled={manifoldSampling}
                  onChange={(e) =>
                    onManifoldParamsChange({
                      ...manifoldParams,
                      boxSize: Math.max(8, Number(e.target.value) || 64),
                    })
                  }
                  className="rounded border border-gray-200 px-1 py-0.5 text-gray-800"
                />
              </label>
            </div>

            <div className="flex items-center gap-1 text-[10px] text-gray-600">
              <button
                type="button"
                disabled={!canCaptureManifoldRoi}
                onClick={onCaptureManifoldRoi}
                className={cn(
                  'flex-1 rounded border px-1.5 py-1',
                  canCaptureManifoldRoi
                    ? 'border-gray-200 bg-white hover:bg-sky-50'
                    : 'border-gray-200 bg-gray-50 text-gray-300 cursor-not-allowed',
                )}
              >
                Restrict to selection ({manifoldRoiShapeCount})
              </button>
              {manifoldRoiShapeCount > 0 && (
                <button
                  type="button"
                  onClick={onClearManifoldRoi}
                  className="rounded border border-gray-200 bg-white px-1.5 py-1 hover:bg-sky-50"
                >
                  Clear
                </button>
              )}
            </div>

            <button
              type="button"
              disabled={!hasFeatureJob || manifoldSampling}
              onClick={onManifoldSample}
              className={cn(
                'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
                !hasFeatureJob || manifoldSampling
                  ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
                  : 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300',
              )}
            >
              {manifoldSampling ? <CircleNotch size={14} className="animate-spin" /> : <Compass size={14} />}
              {manifoldSampling ? 'Sampling…' : 'Suggest regions to label'}
            </button>
            {manifoldSampling && <BusyBar label="Sampling manifold coverage…" />}

            {manifoldMeta && (
              <p className="text-[10px] text-gray-600 leading-snug">
                {manifoldMeta.nPicked} boxes from {manifoldMeta.nSubsample.toLocaleString()} px ·{' '}
                {(manifoldMeta.explainedVariance * 100).toFixed(0)}% variance explained
              </p>
            )}

            {manifoldHasSample && (
              <div className="flex flex-col gap-1 text-[10px] text-gray-600">
                <label className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={manifoldShowHeatmap}
                    onChange={(e) => onManifoldShowHeatmapChange(e.target.checked)}
                  />
                  Show coverage heatmap
                </label>
                <label className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={manifoldShowMarkers}
                    onChange={(e) => onManifoldShowMarkersChange(e.target.checked)}
                  />
                  Show suggested boxes
                </label>
                <label className="flex flex-col gap-0.5">
                  <span className="flex justify-between">
                    <span>Heatmap opacity</span>
                    <span className="text-gray-800 font-medium">
                      {Math.round(manifoldHeatmapOpacity * 100)}%
                    </span>
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={Math.round(manifoldHeatmapOpacity * 100)}
                    onChange={(e) => onManifoldHeatmapOpacityChange(Number(e.target.value) / 100)}
                    className="w-full accent-sky-600"
                  />
                </label>
                <button
                  type="button"
                  onClick={onManifoldDismiss}
                  className="self-start text-[10px] text-gray-500 hover:text-sky-700"
                >
                  Dismiss suggestions
                </button>
              </div>
            )}

            {manifoldError && (
              <p className="text-[10px] text-red-600 leading-snug break-words">{manifoldError}</p>
            )}
          </div>
        </CollapsibleSection>
      </div>
    </CollapsibleSection>
  );
}
