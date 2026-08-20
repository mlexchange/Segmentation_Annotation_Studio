/**
 * DenoisePanel — classical denoising for the open sample.
 *
 * Unlike CLAHE/Sharpen (client-side display bakes in `lib/sam/adjust.ts`), the
 * filters here run SERVER-side on the raw slice before normalization. Three
 * reasons, all deliberate: noise statistics live in the source's own intensity
 * units rather than the 8-bit display range; the preview then uses the exact
 * same code as the "save as a new dataset" bake, so the two cannot drift; and
 * NLM/TV are impractical to implement and run in the browser at 3232².
 *
 * A consequence worth knowing: because the denoised pixels arrive inside the
 * fetched PNG, the magic wand / SAM / lasso all operate on the denoised image
 * too (usually an improvement), and their caches invalidate automatically
 * because the slice URL changed.
 *
 * These filters do not reach a model by default. The one exception is opt-in:
 * "Train on denoised input" (TrainPage / ApplyModelPanel's fine-tune) reads
 * THIS setting out of `stores/denoiseStore` and records it on the resulting
 * run, after which inference reapplies it automatically — see
 * `lib/trainDenoiseOption`. That's also why the panel's state is a store
 * rather than AnnotatePage-local `useState`.
 *
 * Cost is real, so the panel offers a 1:1 centre-crop preview for the slow
 * filters (measured at 3232²: bilateral 7.5 s full-slice vs 0.71 s cropped).
 * It crops rather than downscales on purpose — downscaling is itself a
 * denoiser, so a downscaled preview cannot be used to judge denoising.
 *
 * A "Learned denoiser" mode sits alongside these classical filters: train a
 * self-supervised Noise2Noise/Noise2Void model (no annotations needed) and
 * preview it on the current slice. That flow needs a scheme picker, a
 * training-scope selector, a training job, and a run picker/preview of its
 * own, so it lives in the sibling LearnedDenoiserPanel rather than growing
 * this file further — the classical-filter UI below is unchanged from before
 * that mode existed.
 */
import { useMemo, useState } from 'react';
import { Sparkle, Archive } from '@phosphor-icons/react';
import { API_BASE } from '@/config';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import { useTrainCapability, type DenoiseMethodInfo } from '@/hooks/useTrainCapability';
import { buildSliceUrl, type DenoiseOpts } from '@/hooks/useImageSlice';
import LearnedDenoiserPanel from './LearnedDenoiserPanel';
import type { DenoiseBakeTarget } from './DenoiseBakeModal';
import type { RenderOpts } from '@/stores/datasetStore';

export const CROP_PREVIEW_SIZE = 768;

type DenoiseMode = 'classical' | 'learned';

export interface DenoisePanelProps {
  source: string | null;
  kind: string | null;
  serverUri: string | null;
  sliceIndex: number;
  /** Total slices in the currently-open sample — bounds the Learned
   *  Denoiser mode's training-scope range inputs. */
  nSlices: number;
  renderOpts: RenderOpts;
  denoise: DenoiseOpts;
  onDenoiseChange: (next: DenoiseOpts) => void;
  /** Opens the "denoise & save as a new dataset" flow (Tiled sources only).
   *  Receives what to bake, so the Learned-denoiser mode can bake a trained
   *  run rather than a classical filter. */
  onBake?: (target: DenoiseBakeTarget) => void;
}

export default function DenoisePanel({
  source, kind, serverUri, sliceIndex, nSlices, renderOpts, denoise, onDenoiseChange, onBake,
}: DenoisePanelProps) {
  const { capability } = useTrainCapability();
  const [mode, setMode] = useState<DenoiseMode>('classical');
  const [showCrop, setShowCrop] = useState(false);
  const [autoBusy, setAutoBusy] = useState(false);

  const methods: DenoiseMethodInfo[] = capability.denoise.methods;
  const active = methods.find((m) => m.method === denoise.method);
  const isOn = denoise.method !== 'none';
  const isSlow = active?.cost === 'slow' || active?.cost === 'moderate';

  const cropUrl = useMemo(() => {
    if (!source || !kind || !isOn || !showCrop) return null;
    return buildSliceUrl(source, kind, sliceIndex, renderOpts, serverUri, {
      ...denoise,
      crop: CROP_PREVIEW_SIZE,
    });
  }, [source, kind, sliceIndex, renderOpts, serverUri, denoise, isOn, showCrop]);

  /** Ask the server what strength suits this slice's measured noise level.
   *  Runs through the same estimator the bake uses, so "Auto" here and a baked
   *  result agree. */
  const handleAuto = async () => {
    if (!source || !kind || !isOn) return;
    setAutoBusy(true);
    try {
      const params = new URLSearchParams({
        source, kind, slice_index: String(sliceIndex), method: denoise.method,
      });
      if (serverUri) params.set('server_uri', serverUri);
      const res = await fetch(`${API_BASE}/api/denoise/auto?${params}`);
      if (res.ok) {
        const data = await res.json();
        if (typeof data.strength === 'number') {
          onDenoiseChange({ ...denoise, strength: data.strength });
        }
      }
    } finally {
      setAutoBusy(false);
    }
  };

  if (!source) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5">
        <Sparkle size={14} className="text-gray-500" />
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Denoise</span>
      </div>

      <div className="flex rounded-md border border-gray-300 overflow-hidden text-xs font-medium">
        {([
          ['classical', 'Classical filter'],
          ['learned', 'Learned denoiser'],
        ] as const).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => {
              // Leaving Learned mode with a model still applied would leave the
              // classical <select> bound to a value ('model') that isn't in its
              // option list, rendering blank. Drop back to no denoising.
              if (value === 'classical' && denoise.method === 'model') {
                onDenoiseChange({ ...denoise, method: 'none', runId: undefined });
              }
              setMode(value);
            }}
            className={`flex-1 px-2 py-1.5 transition-colors ${
              mode === value ? 'bg-sky-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === 'classical' && (
        <>
          <select
            value={denoise.method}
            onChange={(e) => onDenoiseChange({ ...denoise, method: e.target.value })}
            className="w-full rounded-md border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-sky-500"
          >
            {methods.map((m) => (
              <option key={m.method} value={m.method} disabled={!m.available}>
                {m.label}
                {m.cost === 'slow' ? ' — slow' : ''}
                {!m.available ? ' (unavailable)' : ''}
              </option>
            ))}
          </select>

          {active && active.method !== 'none' && (
            <p className="text-[11px] leading-snug text-gray-400">{active.description}</p>
          )}

          {isOn && (
            <>
              <div className="flex items-end gap-1.5">
                <DebouncedSlider
                  label="Strength"
                  value={denoise.strength}
                  min={0}
                  max={1}
                  step={0.02}
                  // 400ms, unlike DisplayControls' 0: every commit here is a network
                  // round-trip plus real CPU on the server, not a free GPU filter.
                  debounceMs={400}
                  onChange={(v) => onDenoiseChange({ ...denoise, strength: v })}
                  format={(v) => `${Math.round(v * 100)}%`}
                />
                <button
                  type="button"
                  onClick={handleAuto}
                  disabled={autoBusy}
                  title="Pick a strength from this slice's measured noise level"
                  className="shrink-0 rounded-md border border-gray-300 px-2 py-1 text-[11px] text-gray-600 hover:bg-gray-50 transition-colors disabled:opacity-50"
                >
                  {autoBusy ? '…' : 'Auto'}
                </button>
              </div>

              {isSlow && (
                <label className="flex items-start gap-1.5 text-[11px] text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showCrop}
                    onChange={(e) => setShowCrop(e.target.checked)}
                    className="mt-0.5 accent-sky-600"
                  />
                  <span>
                    Fast 1:1 preview
                    <span className="text-gray-400">
                      {' '}— filters a {CROP_PREVIEW_SIZE}px centre crop instead of the whole slice,
                      for quick tuning. The full slice still updates on the canvas.
                    </span>
                  </span>
                </label>
              )}

              {cropUrl && (
                <div className="rounded-md border border-gray-200 overflow-hidden bg-black">
                  <img src={cropUrl} alt="Denoise preview (centre crop, 1:1)" className="block w-full" />
                </div>
              )}

              <p className="text-[11px] leading-snug text-amber-700">
                Changes the canvas, and the magic wand and SAM do see the denoised image.
                Annotations and exports still use the original pixels. A model only ever sees
                denoised input if you tick “Train on denoised input” when you start training —
                that setting is then recorded on the run and reapplied at inference.
              </p>
            </>
          )}

          {onBake && (
            <button
              type="button"
              onClick={() => onBake({
                method: denoise.method,
                label: active?.label ?? denoise.method,
                strength: denoise.strength,
              })}
              disabled={!isOn}
              title={
                isOn
                  ? 'Apply this filter to every slice and save the result as a new dataset you can open and annotate'
                  : 'Pick a denoise method first'
              }
              className="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium bg-sky-100 text-sky-700 hover:bg-sky-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Archive size={16} />
              Save denoised copy…
            </button>
          )}
        </>
      )}

      {mode === 'learned' && (
        <LearnedDenoiserPanel
          source={source}
          kind={kind}
          serverUri={serverUri}
          currentSlice={sliceIndex}
          nSlices={nSlices}
          renderOpts={renderOpts}
          onBake={onBake}
        />
      )}
    </div>
  );
}
