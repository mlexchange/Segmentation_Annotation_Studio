/**
 * LayersPanel — always-on toggles for canvas function groups + per-class visibility.
 *
 * Per-class annotation visibility lives in ClassManager already (it owns the
 * class list); this panel only exposes the whole-group toggle plus the
 * per-class breakdown for predictions, which has no other home.
 */
import { useEffect } from 'react';
import { Eye, EyeSlash, Stack } from '@phosphor-icons/react';
import { useClassStore } from '@/stores/classStore';
import {
  LAYER_GROUP_META,
  type LayerGroupId,
  useLayerVisibilityStore,
  isPredictionClassVisible,
} from '@/stores/layerVisibilityStore';
import { cn } from '@/lib/utils';

const GROUP_ORDER: LayerGroupId[] = [
  'image',
  'denoise',
  'features',
  'proba',
  'predictions',
  'annotations',
  'manifold',
];

export interface LayersPanelProps {
  /** Feature channel available (preprocess job + selection). */
  hasFeatures?: boolean;
  /** Softmax probability URL loaded. */
  hasProba?: boolean;
  /** Conformal prediction overlays loaded. */
  hasPredictions?: boolean;
  /** Prediction class ids (from trained model). */
  predictionClassIds?: number[];
  /** Manifold sample active. */
  hasManifold?: boolean;
}

function GroupToggle({
  id,
  disabled,
}: {
  id: LayerGroupId;
  disabled?: boolean;
}) {
  const on = useLayerVisibilityStore((s) => s.groups[id]);
  const toggle = useLayerVisibilityStore((s) => s.toggleGroup);
  const meta = LAYER_GROUP_META[id];
  return (
    <button
      type="button"
      disabled={disabled}
      title={disabled ? `${meta.label} (nothing loaded)` : meta.hint}
      onClick={() => toggle(id)}
      className={cn(
        'flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-[11px] transition-colors',
        disabled
          ? 'cursor-not-allowed text-gray-400'
          : on
            ? 'text-gray-800 hover:bg-sky-50'
            : 'text-gray-500 hover:bg-gray-50',
      )}
    >
      {on && !disabled ? (
        <Eye size={14} className="shrink-0 text-sky-700" />
      ) : (
        <EyeSlash size={14} className="shrink-0 text-gray-400" />
      )}
      <span className="font-medium">{meta.label}</span>
      {disabled ? (
        <span className="ml-auto text-[9px] text-gray-400">—</span>
      ) : null}
    </button>
  );
}

/** Sidebar layer stack controls. */
export default function LayersPanel({
  hasFeatures = false,
  hasProba = false,
  hasPredictions = false,
  predictionClassIds = [],
  hasManifold = false,
}: LayersPanelProps) {
  const classes = useClassStore((s) => s.classes);
  const groups = useLayerVisibilityStore((s) => s.groups);
  const probaOpacity = useLayerVisibilityStore((s) => s.probaOpacity);
  const setProbaOpacity = useLayerVisibilityStore((s) => s.setProbaOpacity);
  const predictionsOpacity = useLayerVisibilityStore((s) => s.predictionsOpacity);
  const setPredictionsOpacity = useLayerVisibilityStore((s) => s.setPredictionsOpacity);
  const predictionClassVisible = useLayerVisibilityStore((s) => s.predictionClassVisible);
  const togglePredictionClass = useLayerVisibilityStore((s) => s.togglePredictionClass);
  const showPredictionMulti = useLayerVisibilityStore((s) => s.showPredictionMulti);
  const showPredictionAbstain = useLayerVisibilityStore((s) => s.showPredictionAbstain);
  const setShowPredictionMulti = useLayerVisibilityStore((s) => s.setShowPredictionMulti);
  const setShowPredictionAbstain = useLayerVisibilityStore((s) => s.setShowPredictionAbstain);
  const ensurePredictionClasses = useLayerVisibilityStore((s) => s.ensurePredictionClasses);

  useEffect(() => {
    if (predictionClassIds.length) ensurePredictionClasses(predictionClassIds);
  }, [predictionClassIds, ensurePredictionClasses]);

  const classLabel = (id: number) =>
    classes.find((c) => c.classId === id)?.label?.trim() || `class ${id}`;
  const classColor = (id: number) =>
    classes.find((c) => c.classId === id)?.color ?? '#64748b';

  const disabledFor = (id: LayerGroupId): boolean => {
    if (id === 'features') return !hasFeatures;
    if (id === 'proba') return !hasProba;
    if (id === 'predictions') return !hasPredictions;
    if (id === 'manifold') return !hasManifold;
    if (id === 'annotations') return classes.length === 0;
    return false;
  };

  return (
    <div
      className="flex flex-col gap-1.5 rounded-md border border-gray-200 bg-white p-2"
      data-testid="layers-panel"
    >
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-gray-500">
        <Stack size={13} /> Layers
      </div>

      <div className="flex flex-col gap-0.5">
        {GROUP_ORDER.map((id) => (
          <GroupToggle key={id} id={id} disabled={disabledFor(id)} />
        ))}
      </div>

      {groups.proba && hasProba && (
        <label className="flex flex-col gap-0.5 border-t border-gray-100 pt-1.5 text-[10px] text-gray-600">
          <span className="flex justify-between">
            <span>Probability opacity</span>
            <span className="font-medium text-gray-800">{Math.round(probaOpacity * 100)}%</span>
          </span>
          <input
            type="range"
            min={10}
            max={100}
            step={5}
            value={Math.round(probaOpacity * 100)}
            onChange={(e) => setProbaOpacity(Number(e.target.value) / 100)}
            className="w-full accent-sky-600"
          />
        </label>
      )}

      {groups.predictions && hasPredictions && (
        <div className="flex flex-col gap-1 border-t border-gray-100 pt-1.5">
          <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
            <span className="flex justify-between">
              <span>Prediction opacity</span>
              <span className="font-medium text-gray-800">
                {Math.round(predictionsOpacity * 100)}%
              </span>
            </span>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={Math.round(predictionsOpacity * 100)}
              onChange={(e) => setPredictionsOpacity(Number(e.target.value) / 100)}
              className="w-full accent-sky-600"
            />
          </label>
          <div className="flex flex-wrap gap-1">
            <button
              type="button"
              onClick={() => setShowPredictionMulti(!showPredictionMulti)}
              className={cn(
                'rounded border px-1.5 py-0.5 text-[9px]',
                showPredictionMulti
                  ? 'border-amber-300 bg-amber-50 text-amber-900'
                  : 'border-gray-200 bg-gray-50 text-gray-400',
              )}
            >
              Multi
            </button>
            <button
              type="button"
              onClick={() => setShowPredictionAbstain(!showPredictionAbstain)}
              className={cn(
                'rounded border px-1.5 py-0.5 text-[9px]',
                showPredictionAbstain
                  ? 'border-slate-300 bg-slate-100 text-slate-800'
                  : 'border-gray-200 bg-gray-50 text-gray-400',
              )}
            >
              Abstain
            </button>
          </div>
          <span className="text-[9px] font-medium uppercase tracking-wide text-gray-500">
            Prediction classes
          </span>
          {(predictionClassIds.length ? predictionClassIds : classes.map((c) => c.classId)).map(
            (cid) => {
              const on = isPredictionClassVisible(predictionClassVisible, cid);
              return (
                <button
                  key={`pred-${cid}`}
                  type="button"
                  onClick={() => togglePredictionClass(cid)}
                  className="flex items-center gap-1.5 rounded px-1 py-0.5 text-left text-[10px] hover:bg-sky-50"
                >
                  {on ? <Eye size={12} className="text-sky-700" /> : <EyeSlash size={12} className="text-gray-400" />}
                  <span
                    className="inline-block h-2 w-2 shrink-0 rounded-sm"
                    style={{ background: classColor(cid), opacity: on ? 1 : 0.3 }}
                  />
                  <span className={on ? 'text-gray-800' : 'text-gray-400'}>{classLabel(cid)}</span>
                </button>
              );
            },
          )}
        </div>
      )}
    </div>
  );
}
