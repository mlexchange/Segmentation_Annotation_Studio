/**
 * ManifoldSuggestPanel — Suggest Labels (variance boxes + ROI).
 */
import { CircleNotch, Crosshair, MapTrifold, Selection } from '@phosphor-icons/react';
import type { ManifoldParams } from '@/hooks/useFeatureManifold';
import { cn } from '@/lib/utils';

export interface ManifoldSuggestPanelProps {
  hasFeatureJob: boolean;
  params: ManifoldParams;
  onParamsChange: (p: ManifoldParams) => void;
  sampling: boolean;
  hasSample: boolean;
  showHeatmap: boolean;
  onShowHeatmapChange: (v: boolean) => void;
  showMarkers: boolean;
  onShowMarkersChange: (v: boolean) => void;
  heatmapOpacity: number;
  onHeatmapOpacityChange: (v: number) => void;
  meta: {
    k: number;
    nPicked: number;
    nSubsample: number;
    explainedVariance: number;
    radius: number;
    boxSize: number;
    hasMask?: boolean;
    maskPixels?: number;
  } | null;
  error: string | null;
  onSample: () => void;
  onDismiss: () => void;
  /** Number of shapes in the active suggest ROI (0 = full image). */
  roiShapeCount: number;
  canCaptureRoi: boolean;
  onCaptureRoi: () => void;
  onClearRoi: () => void;
}

const btn =
  'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors';

/** Controls for greedy variance-box sampling overlay. */
export default function ManifoldSuggestPanel({
  hasFeatureJob,
  params,
  onParamsChange,
  sampling,
  hasSample,
  showHeatmap,
  onShowHeatmapChange,
  showMarkers,
  onShowMarkersChange,
  heatmapOpacity,
  onHeatmapOpacityChange,
  meta,
  error,
  onSample,
  onDismiss,
  roiShapeCount,
  canCaptureRoi,
  onCaptureRoi,
  onClearRoi,
}: ManifoldSuggestPanelProps) {
  const canSample = hasFeatureJob && !sampling;

  return (
    <div className="flex flex-col gap-2 border-t border-gray-100 pt-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
          Suggest labels
        </span>
        <span
          className="text-[10px] text-gray-400"
          title="Sliding-box feature variance with spatial + feature exclusion"
        >
          variance boxes
        </span>
      </div>
      <p className="text-[10px] text-gray-500 leading-snug">
        Optionally select a region of interest, then Suggest. Boxes are placed only fully inside
        the ROI. Guidance only — not saved.
      </p>

      <div className="flex flex-col gap-1">
        <button
          type="button"
          disabled={!canCaptureRoi || sampling}
          title={
            canCaptureRoi
              ? 'Use currently selected shape(s) as the suggest ROI'
              : 'Select one or more shapes first'
          }
          onClick={onCaptureRoi}
          className={cn(
            btn,
            canCaptureRoi && !sampling
              ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
              : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
          )}
        >
          <Selection size={14} />
          Use selection as ROI
        </button>
        <div className="flex items-center justify-between text-[10px] text-gray-600 px-0.5">
          <span>
            {roiShapeCount > 0
              ? `ROI: ${roiShapeCount} shape${roiShapeCount === 1 ? '' : 's'}`
              : 'ROI: none (full image)'}
          </span>
          {roiShapeCount > 0 && (
            <button
              type="button"
              onClick={onClearRoi}
              disabled={sampling}
              className="text-sky-700 hover:underline disabled:text-gray-400"
            >
              Clear ROI
            </button>
          )}
        </div>
      </div>

      <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
        <span className="flex justify-between">
          <span>K (boxes)</span>
          <span className="font-medium text-gray-800">{params.k}</span>
        </span>
        <input
          type="range"
          min={8}
          max={128}
          step={1}
          value={params.k}
          disabled={sampling}
          onChange={(e) =>
            onParamsChange({ ...params, k: Math.min(128, Math.max(8, Number(e.target.value))) })
          }
          className="w-full accent-sky-600"
        />
      </label>

      <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
        <span className="flex justify-between">
          <span>Box size (px)</span>
          <span className="font-medium text-gray-800">{params.boxSize}</span>
        </span>
        <input
          type="range"
          min={16}
          max={256}
          step={2}
          value={params.boxSize}
          disabled={sampling}
          onChange={(e) =>
            onParamsChange({
              ...params,
              boxSize: Math.min(256, Math.max(16, Number(e.target.value) || 64)),
            })
          }
          className="w-full accent-sky-600"
        />
      </label>

      <button
        type="button"
        disabled={!canSample}
        title={!hasFeatureJob ? 'Compute features first' : 'Suggest annotation boxes'}
        onClick={onSample}
        className={cn(
          btn,
          canSample
            ? 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300'
            : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed',
        )}
      >
        {sampling ? (
          <CircleNotch size={14} className="animate-spin" />
        ) : (
          <MapTrifold size={14} />
        )}
        {sampling ? 'Sampling…' : 'Suggest labels'}
      </button>

      {hasSample && (
        <div className="flex flex-col gap-1.5 rounded-md border border-gray-100 bg-gray-50 p-1.5">
          {meta && (
            <p className="text-[10px] text-gray-600">
              {meta.nPicked}/{meta.k} boxes · {meta.boxSize}px side · exclude r≈
              {Math.round(meta.radius)}px
              {meta.hasMask ? ` · ROI` : ''} · PCA{' '}
              {(meta.explainedVariance * 100).toFixed(0)}% var
            </p>
          )}
          {sampling && (
            <p className="text-[10px] text-sky-700 leading-snug">Re-placing boxes…</p>
          )}
          {meta && (meta.boxSize !== params.boxSize || meta.k !== params.k) && !sampling && (
            <p className="text-[10px] text-amber-700 leading-snug">
              Params changed — waiting to re-suggest…
            </p>
          )}
          <label className="flex items-center gap-1.5 text-[10px] text-gray-700">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => onShowHeatmapChange(e.target.checked)}
            />
            Residual heatmap
          </label>
          <label className="flex items-center gap-1.5 text-[10px] text-gray-700">
            <input
              type="checkbox"
              checked={showMarkers}
              onChange={(e) => onShowMarkersChange(e.target.checked)}
            />
            <Crosshair size={12} /> Boxes
          </label>
          {showHeatmap && (
            <label className="flex flex-col gap-0.5 text-[10px] text-gray-600">
              <span className="flex justify-between">
                <span>Opacity</span>
                <span>{Math.round(heatmapOpacity * 100)}%</span>
              </span>
              <input
                type="range"
                min={10}
                max={80}
                step={5}
                value={Math.round(heatmapOpacity * 100)}
                onChange={(e) => onHeatmapOpacityChange(Number(e.target.value) / 100)}
                className="w-full accent-sky-600"
              />
            </label>
          )}
          <button
            type="button"
            onClick={onDismiss}
            className="text-[11px] text-sky-700 hover:underline self-start"
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
