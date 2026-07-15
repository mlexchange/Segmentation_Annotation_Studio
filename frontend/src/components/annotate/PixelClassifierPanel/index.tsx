/**
 * PixelClassifierPanel — CatBoost train / conformal predict / commit under Features.
 */
import { CaretLeft, CaretRight, CircleNotch, FloppyDisk, TreeStructure } from '@phosphor-icons/react';
import type {
  ClfParams,
  ClfPredictCounts,
  ClfTrainResult,
  ClfTreeView,
} from '@/hooks/usePixelClassifier';
import { cn } from '@/lib/utils';

export interface PixelClassifierPanelProps {
  hasFeatureJob: boolean;
  /** True when Train can auto-preprocess via preferred Feature Setup. */
  canAutoPreprocess?: boolean;
  hasShapes: boolean;
  training: boolean;
  predicting: boolean;
  params: ClfParams;
  onParamsChange: (p: ClfParams) => void;
  model: ClfTrainResult | null;
  treeView: ClfTreeView | null;
  treeIndex: number;
  onTreeIndexChange: (index: number) => void;
  hasPrediction: boolean;
  predictCounts: ClfPredictCounts | null;
  /** Softmax class heatmap browsing (after Predict). */
  probaClassIndex?: number;
  activeProbaClassId?: number | null;
  activeProbaThreshold?: number;
  classLabelForId?: (classId: number) => string;
  onCycleProbaClass?: (delta: number) => void;
  onProbaThresholdChange?: (threshold: number) => void;
  onSaveClassMap?: () => void;
  savingClass?: boolean;
  error: string | null;
  onTrain: () => void;
  onPredict: () => void;
  onCommit: () => void;
  onDismiss: () => void;
  /** Button label for Commit (differs in Cleanup vs Draw mode). */
  commitLabel?: string;
  /** Active ipred Feature Setup id (from Ipred tab). */
  featureSetupId?: string | null;
  /** Active trainer plugin id. */
  trainerId?: string | null;
}

/** Sidebar controls for ilastik-style CatBoost + split-conformal sets. */
export default function PixelClassifierPanel({
  hasFeatureJob,
  canAutoPreprocess = false,
  hasShapes,
  training,
  predicting,
  params,
  onParamsChange,
  model,
  treeView,
  treeIndex,
  onTreeIndexChange,
  hasPrediction,
  predictCounts,
  probaClassIndex = 0,
  activeProbaClassId = null,
  activeProbaThreshold = 0.5,
  classLabelForId,
  onCycleProbaClass,
  onProbaThresholdChange,
  onSaveClassMap,
  savingClass = false,
  error,
  onTrain,
  onPredict,
  onCommit,
  onDismiss,
  commitLabel = 'Commit singletons',
  featureSetupId = null,
  trainerId = null,
}: PixelClassifierPanelProps) {
  const busy = training || predicting || savingClass;
  const canTrain = (hasFeatureJob || canAutoPreprocess) && hasShapes && !busy;
  const canPredict = !!model && (hasFeatureJob || !!model.featureId) && !busy;
  const maxImp = model?.featureImportances[0]?.importance ?? 1;
  const alphaPct = Math.round(params.alpha * 100);
  const nClasses = model?.classIds.length ?? 0;
  const threshPct = Math.round(activeProbaThreshold * 100);
  const probaLabel =
    activeProbaClassId !== null
      ? (classLabelForId?.(activeProbaClassId) ?? `class ${activeProbaClassId}`)
      : '—';

  return (
    <div className="flex flex-col gap-2 border-t border-gray-100 pt-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
          Classifier
        </span>
        <span className="text-[10px] text-gray-400" title="Trainer + Mondrian conformal sets">
          {trainerId ?? 'catboost'} · conformal
        </span>
      </div>

      <p className="text-[11px] text-gray-600 leading-snug">
        Setup:{' '}
        <span className="font-mono text-gray-800">
          {featureSetupId ?? 'none — pick on Ipred tab'}
        </span>
        {!hasFeatureJob && canAutoPreprocess ? (
          <span className="text-gray-500"> · Train will compute features</span>
        ) : null}
      </p>

      <p className="text-[10px] text-gray-500 leading-snug">
        Sparse labels. Conformal sets at α: solid = singleton (commit), hatch = multi,
        dark = abstain. Commit keeps scribbles and only adds singletons. Browse softmax
        class maps below to threshold each class (live preview) and optionally
        cache a mask set.
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
        <span className="text-[9px] text-gray-400 leading-snug">
          Lower α → larger sets (more multi). Predict reuses model; no retrain.
        </span>
      </label>

      <button
        type="button"
        disabled={!canTrain}
        title={!hasFeatureJob ? 'Compute features first' : !hasShapes ? 'Annotate at least two classes' : 'Train CatBoost'}
        onClick={onTrain}
        className={cn(
          'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
          canTrain
            ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
        )}
      >
        {training ? (
          <CircleNotch size={14} className="animate-spin" />
        ) : (
          <TreeStructure size={14} />
        )}
        {training ? 'Training…' : 'Train classifier'}
      </button>

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
        {predicting ? (
          <CircleNotch size={14} className="animate-spin" />
        ) : null}
        {predicting ? 'Predicting…' : 'Predict'}
      </button>

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

          {model.nTrees > 0 && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-1">
                <span className="text-[10px] font-medium text-gray-600">Tree</span>
                <input
                  type="number"
                  min={0}
                  max={Math.max(0, model.nTrees - 1)}
                  value={treeIndex}
                  onChange={(e) => onTreeIndexChange(Number(e.target.value) || 0)}
                  className="w-14 rounded border border-gray-200 px-1 py-0.5 text-[10px]"
                />
              </div>
              {treeView && (
                <div className="max-h-28 overflow-y-auto text-[10px] text-gray-600 space-y-0.5">
                  <p>depth {treeView.depth} · {treeView.n_leaves} leaves</p>
                  {treeView.splits.map((sp, i) => (
                    <p key={`${sp.feature_index}-${i}`} className="truncate" title={sp.feature_label}>
                      L{i}: {sp.feature_label} ≤ {sp.threshold.toPrecision(3)}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {hasPrediction && predictCounts && (
        <p className="text-[10px] text-gray-600 leading-snug">
          Sets · singleton {predictCounts.singleton.toLocaleString()} · multi{' '}
          {predictCounts.multi.toLocaleString()} · abstain {predictCounts.abstain.toLocaleString()}
        </p>
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
            <span className="text-[9px] text-gray-400 leading-snug">
              Viridis; P(class) below threshold is clipped (transparent).
            </span>
          </label>
          {onSaveClassMap && (
            <button
              type="button"
              disabled={busy}
              onClick={onSaveClassMap}
              className={cn(
                'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
                busy
                  ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
                  : 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300',
              )}
            >
              {savingClass ? (
                <CircleNotch size={14} className="animate-spin" />
              ) : (
                <FloppyDisk size={14} />
              )}
              {savingClass ? 'Saving…' : 'Save class → mask cache'}
            </button>
          )}
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

      {error && (
        <p className="text-[10px] text-red-600 leading-snug break-words">{error}</p>
      )}
    </div>
  );
}
