/**
 * FeatureChannelsPanel — compute / cycle feature channels via ipred Feature Setup.
 */
import { useState, type ReactNode } from 'react';
import { CaretLeft, CaretRight, CircleNotch, Stack } from '@phosphor-icons/react';
import type { FeatureParams, FeatureJobInfo } from '@/hooks/useFeatureChannels';
import { cn } from '@/lib/utils';

export interface FeatureChannelsPanelProps {
  params: FeatureParams;
  onParamsChange: (p: FeatureParams) => void;
  job: FeatureJobInfo | null;
  channelIndex: number | null;
  computing: boolean;
  error: string | null;
  onCompute: () => void;
  onSelectChannel: (index: number | null) => void;
  onCycle: (delta: number) => void;
  onOriginal: () => void;
  disabled?: boolean;
  /** Backend has SlimSAM vision_encoder.onnx (legacy hint). */
  samAvailable?: boolean;
  /** Active ipred Feature Setup id. */
  featureSetupId?: string | null;
  /** Optional CatBoost panel rendered under channel controls. */
  classifierSlot?: ReactNode;
  /** Optional manifold / Suggest Labels panel. */
  manifoldSlot?: ReactNode;
}

/** Sidebar panel for feature bank compute + channel selection. */
export default function FeatureChannelsPanel({
  params,
  onParamsChange,
  job,
  channelIndex,
  computing,
  error,
  onCompute,
  onSelectChannel,
  onCycle,
  onOriginal,
  disabled = false,
  samAvailable = false,
  featureSetupId = null,
  classifierSlot,
  manifoldSlot,
}: FeatureChannelsPanelProps) {
  const [showLegacy, setShowLegacy] = useState(false);
  const n = job?.channels.length ?? 0;
  const activeLabel =
    channelIndex !== null && job
      ? job.channels[channelIndex]?.label ?? `Channel ${channelIndex}`
      : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
          Features
        </span>
        {n > 0 && (
          <span className="text-[10px] text-gray-400">
            {n} ch{job?.hasSam ? ' +SAM' : ''}
            {job?.cacheHit ? ' · cache' : ''}
          </span>
        )}
      </div>

      <p className="text-[11px] text-gray-600 leading-snug">
        Composition:{' '}
        <span className="font-mono text-gray-800">
          {featureSetupId ?? 'none — pick on Ipred tab'}
        </span>
      </p>

      <button
        type="button"
        disabled={computing || disabled || !featureSetupId}
        onClick={onCompute}
        className={cn(
          'flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs border transition-colors',
          computing || disabled || !featureSetupId
            ? 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'
            : 'bg-white text-gray-800 border-gray-200 hover:bg-sky-50 hover:border-sky-300',
        )}
      >
        {computing ? (
          <CircleNotch size={14} className="animate-spin" />
        ) : (
          <Stack size={14} />
        )}
        {computing ? 'Computing…' : 'Compute'}
      </button>

      <button
        type="button"
        className="text-[10px] text-gray-500 hover:text-sky-700 self-start"
        onClick={() => setShowLegacy((v) => !v)}
      >
        {showLegacy ? 'Hide' : 'Show'} legacy recipe knobs
      </button>

      {showLegacy && (
        <>
          <div className="grid grid-cols-2 gap-1.5 text-xs">
            <label className="flex flex-col gap-0.5 text-gray-600">
              σ min
              <input
                type="number"
                min={0.5}
                step={1}
                value={params.sigmaMin}
                disabled={computing || disabled}
                onChange={(e) =>
                  onParamsChange({ ...params, sigmaMin: Number(e.target.value) || 1 })
                }
                className="rounded border border-gray-200 px-1.5 py-1 text-gray-800"
              />
            </label>
            <label className="flex flex-col gap-0.5 text-gray-600">
              σ max
              <input
                type="number"
                min={1}
                step={1}
                value={params.sigmaMax}
                disabled={computing || disabled}
                onChange={(e) =>
                  onParamsChange({ ...params, sigmaMax: Number(e.target.value) || 8 })
                }
                className="rounded border border-gray-200 px-1.5 py-1 text-gray-800"
              />
            </label>
          </div>
          <div className="flex flex-wrap gap-x-2 gap-y-1 text-xs text-gray-700">
            {(
              [
                ['intensity', 'Intensity'],
                ['edges', 'Edges'],
                ['texture', 'Texture'],
                ['clahe', 'CLAHE'],
                ['includeSam', 'SlimSAM'],
              ] as const
            ).map(([key, label]) => (
              <label
                key={key}
                className={cn(
                  'inline-flex items-center gap-1',
                  key === 'includeSam' && !samAvailable ? 'opacity-50' : 'cursor-pointer',
                )}
              >
                <input
                  type="checkbox"
                  checked={params[key]}
                  disabled={computing || disabled || (key === 'includeSam' && !samAvailable)}
                  onChange={(e) => onParamsChange({ ...params, [key]: e.target.checked })}
                  className="rounded border-gray-300"
                />
                {label}
              </label>
            ))}
          </div>
          <p className="text-[10px] text-amber-700 leading-snug">
            Legacy knobs no longer drive Compute — edit the Feature Setup on the Ipred tab.
          </p>
        </>
      )}

      {error && (
        <p className="text-[10px] text-red-600 leading-snug break-words">{error}</p>
      )}

      {job && n > 0 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="Previous channel"
              title="Previous channel"
              disabled={channelIndex === null}
              onClick={() => onCycle(-1)}
              className="p-1 rounded border border-gray-200 hover:bg-sky-50 disabled:opacity-40"
            >
              <CaretLeft size={14} />
            </button>
            <select
              className="flex-1 min-w-0 rounded border border-gray-200 px-1 py-1 text-xs text-gray-800"
              value={channelIndex === null ? '' : String(channelIndex)}
              onChange={(e) => {
                const v = e.target.value;
                onSelectChannel(v === '' ? null : Number(v));
              }}
            >
              <option value="">Original</option>
              {job.channels.map((ch) => (
                <option key={ch.index} value={ch.index}>
                  {ch.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              aria-label="Next channel"
              title="Next channel"
              disabled={channelIndex === null}
              onClick={() => onCycle(1)}
              className="p-1 rounded border border-gray-200 hover:bg-sky-50 disabled:opacity-40"
            >
              <CaretRight size={14} />
            </button>
          </div>
          {activeLabel && (
            <p className="text-[10px] text-gray-500 truncate" title={activeLabel}>
              {activeLabel}
            </p>
          )}
          <button
            type="button"
            onClick={onOriginal}
            className="text-[11px] text-sky-700 hover:underline self-start"
          >
            Show original
          </button>
        </div>
      )}

      {manifoldSlot}
      {classifierSlot}
    </div>
  );
}
