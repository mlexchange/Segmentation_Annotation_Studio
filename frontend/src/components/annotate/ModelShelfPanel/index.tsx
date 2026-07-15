/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 *
 * ModelShelfPanel — TEMPORARY scaffold to save / list / apply CatBoost models.
 * Replace with your own model registry UX, then delete this panel.
 */
import { useEffect, useState } from 'react';
import { FloppyDisk, Lightning, Trash } from '@phosphor-icons/react';
import { KillSwitchBanner } from '@/components/KillSwitchBanner';
import { useModelShelf } from '@/hooks/useModelShelf';
import type { FeatureParams } from '@/hooks/useFeatureChannels';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useConnectionStore } from '@/stores/connectionStore';
import { formatFeatureRecipe } from '@/lib/featureRecipe';
import { cn } from '@/lib/utils';

interface ModelShelfPanelProps {
  sourceKey: string | null;
  trainedModelId: string | null;
  alpha: number;
  /** Current feature params used when saving a freshly trained model. */
  featureParams: FeatureParams;
}

const btn =
  'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
  'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

/**
 * # REMOVE THIS AND USE YOUR OWN STUFF
 * Persist classifiers and re-apply them on the current image into a mask set.
 */
export default function ModelShelfPanel({
  sourceKey,
  trainedModelId,
  alpha,
  featureParams,
}: ModelShelfPanelProps) {
  const shelf = useModelShelf();
  const preferredShelfModelId = useConnectionStore((s) => s.preferredShelfModelId);
  const preferredShelfModelName = useConnectionStore((s) => s.preferredShelfModelName);
  const [name, setName] = useState('my classifier');
  const [status, setStatus] = useState<string | null>(null);
  const { currentSlice } = useDatasetStore();
  const byImage = useAnnotationStore((s) => s.byImage);
  const sliceShapes = sourceKey
    ? (byImage[sourceKey]?.[String(currentSlice)] ?? [])
    : [];

  useEffect(() => {
    if (preferredShelfModelName) {
      setStatus(`Session model ready: “${preferredShelfModelName}” — click Apply`);
    }
  }, [preferredShelfModelName]);

  const sortedModels = [...shelf.models].sort((a, b) => {
    if (a.id === preferredShelfModelId) return -1;
    if (b.id === preferredShelfModelId) return 1;
    return 0;
  });

  return (
    <div
      className="flex flex-col gap-2 rounded-md border-2 border-amber-500/80 p-1.5"
      data-testid="model-shelf-panel-scaffold"
    >
      <KillSwitchBanner detail="Scaffold: CatBoost model shelf (save / apply). Wire your own model store and delete this." />
      <span className="flex items-center gap-1.5 text-xs font-semibold uppercase text-gray-500 tracking-wide">
        <FloppyDisk size={13} /> Model shelf
        <span className="ml-auto font-mono text-[9px] font-bold normal-case text-amber-700">
          TEMP / REMOVE
        </span>
      </span>
      <p className="text-[10px] text-gray-400 -mt-1 leading-snug">
        Save a trained classifier, then Apply on other images (recomputes its feature recipe).
        {preferredShelfModelId
          ? ' A model was selected on Connect/Browse (highlighted).'
          : ''}
      </p>

      <div className="flex gap-1">
        <input
          className="flex-1 rounded border border-gray-200 px-1 py-0.5 text-[10px]"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Model name"
        />
        <button
          type="button"
          className={btn}
          disabled={!trainedModelId || shelf.busy}
          title={!trainedModelId ? 'Train a classifier first' : 'Save to disk'}
          onClick={() => {
            if (!trainedModelId) return;
            void shelf.saveModel(trainedModelId, name, featureParams).then((ok) => {
              setStatus(ok ? `Saved “${name}”` : 'Save failed');
            });
          }}
        >
          Save
        </button>
      </div>

      <ul className="flex flex-col gap-0.5 max-h-32 overflow-y-auto">
        {sortedModels.length === 0 && (
          <li className="text-[10px] text-gray-400">No saved models yet.</li>
        )}
        {sortedModels.map((m) => (
          <li
            key={m.id}
            className={cn(
              'flex items-center gap-1 text-[10px] rounded px-1 py-0.5 hover:bg-gray-50',
              m.id === preferredShelfModelId && 'bg-sky-50 ring-1 ring-sky-200',
            )}
          >
            <div className="flex-1 min-w-0">
              <p className="truncate font-medium text-gray-700">
                {m.name}
                {m.id === preferredShelfModelId ? ' · session' : ''}
              </p>
              <p className="text-gray-400 truncate">
                train {m.n_train} / cal {m.n_cal}
                {m.uses_sam ? ' · SAM' : ''} · classes {(m.class_ids ?? []).join(',')}
              </p>
              <p className="text-gray-400 truncate" title={formatFeatureRecipe(m.feature_recipe)}>
                {formatFeatureRecipe(m.feature_recipe)}
              </p>
            </div>
            <button
              type="button"
              className={cn(btn, 'px-1.5')}
              disabled={!sourceKey || shelf.busy}
              title="Predict → new mask set"
              onClick={() => {
                if (!sourceKey) return;
                void shelf
                  .applyToCurrent(m.id, {
                    sourceKey,
                    alpha,
                    preserveShapes: sliceShapes,
                    name: `${m.name} α=${Math.round(alpha * 100)}%`,
                  })
                  .then((id) => setStatus(id ? `Applied → mask set` : 'Apply failed'));
              }}
            >
              <Lightning size={12} />
            </button>
            <button
              type="button"
              className={cn(btn, 'px-1.5')}
              disabled={shelf.busy}
              onClick={() => {
                void shelf.deleteModel(m.id).then(() => setStatus('Deleted'));
              }}
            >
              <Trash size={12} />
            </button>
          </li>
        ))}
      </ul>

      {(status || shelf.error) && (
        <p className={cn('text-[10px] leading-snug', shelf.error ? 'text-red-600' : 'text-gray-500')}>
          {shelf.error ?? status}
        </p>
      )}
    </div>
  );
}
