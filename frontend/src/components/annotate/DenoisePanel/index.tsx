/**
 * DenoisePanel — server-side denoising of the slice, before it is normalized.
 *
 * Unlike the sliders next to it, this is NOT a display filter. Denoising runs on
 * the raw slice in its own intensity units (noise statistics do not survive a
 * trip through the 8-bit display range), so it changes what every intensity
 * tool sees — the Threshold Brush, the Sampler's fit, magic wand, livewire — not
 * just what is drawn. It still does not change exported pixels; turning a
 * denoised view into data is what "Save denoised copy" is for.
 *
 * The slider is debounced, and results are cached server-side per
 * (slice, method, strength) — measured 1.61s -> 0.006s on a repeat — so
 * revisiting a setting is free. Two refinements are NOT here yet and are worth
 * knowing about:
 *
 *   - **Side-by-side comparison.** Denoising is judged by what it *removes*,
 *     which the current on/off switch hides: flipping between two images makes
 *     you compare from memory. A wipe overlay needs a second image layer in
 *     `AnnotationCanvas`.
 *   - **Crop while dragging.** `buildSliceUrl` already supports `denoise_crop`
 *     (measured: bilateral on a 2560² slice, 7.3s full vs 0.49s at 512px), but
 *     nothing requests it yet, so a slow method is slow on every commit.
 */
import { useCallback, useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { MagicWand, Warning } from '@phosphor-icons/react';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import { API_BASE } from '@/config';
import type { DenoiseOpts } from '@/stores/datasetStore';

/** One entry of `GET /api/denoise/methods`. */
export interface DenoiseMethodInfo {
  method: string;
  label: string;
  cost: 'cheap' | 'moderate' | 'slow';
  description: string;
  available: boolean;
  z_radius: number;
}

export interface DenoisePanelProps {
  denoise: DenoiseOpts;
  onChange: (opts: Partial<DenoiseOpts>) => void;
  /** Dataset identity, for the "Auto" suggestion (which reads this slice). */
  source: string | null;
  kind: string | null;
  serverUri: string | null;
  sliceIndex: number;
  /** Opens the bake modal — turning the preview into a real dataset. */
  onBake?: () => void;
  /** True while the slice request for the current settings is in flight. */
  busy?: boolean;
}

export default function DenoisePanel({
  denoise,
  onChange,
  source,
  kind,
  serverUri,
  sliceIndex,
  onBake,
  busy,
}: DenoisePanelProps) {
  const [autoError, setAutoError] = useState<string | null>(null);
  const [autoBusy, setAutoBusy] = useState(false);

  const { data: methods = [] } = useQuery<DenoiseMethodInfo[]>({
    queryKey: ['denoise-methods'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/denoise/methods`);
      if (!res.ok) throw new Error('Failed to load denoise methods');
      return (await res.json()).methods;
    },
    staleTime: Infinity, // capability of the server, not of the dataset
  });

  useEffect(() => setAutoError(null), [denoise.method]);

  const active = denoise.method !== 'none';
  const current = methods.find((m) => m.method === denoise.method);

  const applyAuto = useCallback(async () => {
    if (!source || !kind) return;
    setAutoBusy(true);
    setAutoError(null);
    try {
      const params = new URLSearchParams({
        source, kind, slice_index: String(sliceIndex), method: denoise.method,
      });
      if (serverUri) params.set('server_uri', serverUri);
      const res = await fetch(`${API_BASE}/api/denoise/auto?${params}`);
      if (!res.ok) throw new Error('Could not measure this slice');
      const { strength, noise_sigma: sigma } = await res.json();
      onChange({ strength });
      if (strength === 0) {
        // Saying "0.0" alone reads as a failure. It is a finding: this slice is
        // already clean, and smoothing it would only cost detail.
        setAutoError(
          `This slice looks clean (noise ≈ ${(sigma * 100).toFixed(2)}% of its range) — ` +
          'denoising it would mostly remove detail.'
        );
      }
    } catch (err) {
      setAutoError(err instanceof Error ? err.message : String(err));
    } finally {
      setAutoBusy(false);
    }
  }, [source, kind, serverUri, sliceIndex, denoise.method, onChange]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">Denoise</label>
        {busy && active && <span className="text-[10px] text-gray-500">filtering…</span>}
      </div>

      <select
        value={denoise.method}
        onChange={(e) => onChange({ method: e.target.value })}
        className="w-full rounded border border-gray-200 bg-white px-1 py-0.5 text-xs text-gray-800"
      >
        {methods.map((m) => (
          <option key={m.method} value={m.method} disabled={!m.available}>
            {m.label}
            {m.cost === 'slow' ? ' (slow)' : ''}
            {!m.available ? ' — unavailable' : ''}
          </option>
        ))}
      </select>

      {current && current.method !== 'none' && (
        <p className="text-[11px] leading-snug text-gray-600">{current.description}</p>
      )}

      {active && (
        <>
          <DebouncedSlider
            label="Strength"
            min={0}
            max={1}
            step={0.01}
            value={denoise.strength}
            onChange={(v) => onChange({ strength: v })}
          />

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={applyAuto}
              disabled={autoBusy || !source}
              className="flex items-center gap-1 rounded border border-gray-200 bg-white px-2 py-1 text-[11px] text-gray-800 hover:bg-sky-50 hover:border-sky-300 disabled:opacity-50"
              title="Measure this slice's noise level and suggest a strength"
            >
              <MagicWand size={12} />
              {autoBusy ? 'Measuring…' : 'Auto'}
            </button>
            {onBake && (
              <button
                type="button"
                onClick={onBake}
                className="rounded border border-gray-200 bg-white px-2 py-1 text-[11px] text-gray-800 hover:bg-sky-50 hover:border-sky-300"
                title="Write a denoised copy as a new dataset you can annotate and export"
              >
                Save denoised copy…
              </button>
            )}
          </div>

          {autoError && (
            <p className="flex items-start gap-1 rounded border border-amber-200 bg-amber-50 p-1.5 text-[11px] leading-snug text-amber-800">
              <Warning size={12} className="mt-0.5 shrink-0" />
              {autoError}
            </p>
          )}

          <p className="text-[10px] leading-snug text-gray-500">
            Affects the intensity tools too, not just the display. Exports still use
            the original pixels.
          </p>
        </>
      )}
    </div>
  );
}
