/**
 * SuggestLabelsPanel — manifold-coverage sampling ("Suggest regions to
 * label"), as its own sidebar section.
 *
 * Previously folded into PixelClassifierPanel, near the bottom of the
 * sidebar — but useFeatureManifold only ever depends on a feature bank
 * (featureJobId), never on a trained classifier, so it was buried behind
 * classifier training for no functional reason. Pulled out to stand next to
 * FeatureChannelsPanel instead: the earliest point it actually has what it
 * needs, and well before a user is asked to train anything.
 */
import { CircleNotch, Compass } from '@phosphor-icons/react';
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

export interface SuggestLabelsPanelProps {
  hasFeatureJob: boolean;
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

/** Sidebar section for manifold-coverage label suggestions. */
export default function SuggestLabelsPanel({
  hasFeatureJob,
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
}: SuggestLabelsPanelProps) {
  return (
    <CollapsibleSection
      title="Suggest labels (manifold coverage)"
      icon={<Compass size={13} />}
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
  );
}
