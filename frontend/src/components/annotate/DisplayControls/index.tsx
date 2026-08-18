/**
 * DisplayControls — brightness/contrast sliders + a min/max levels histogram, all
 * display-only via Konva filters. Does NOT affect exported pixel values — that is
 * governed by RenderOpts.
 */
import { ArrowCounterClockwise } from '@phosphor-icons/react';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import HistogramControl from '@/components/annotate/HistogramControl';
import { COLORMAP_NAMES, colormapGradient, type ColormapName } from '@/lib/colormaps';

export interface DisplayControlsProps {
  brightness: number;
  contrast: number;
  onBrightnessChange: (v: number) => void;
  onContrastChange: (v: number) => void;
  onReset: () => void;
  /** Levels histogram + window (0–255). */
  histogramBins: number[] | null;
  levelsLo: number;
  levelsHi: number;
  onLevelsChange: (lo: number, hi: number) => void;
  onLevelsReset: () => void;
  /** False-color map + gamma (display-only). */
  colormap: ColormapName;
  gamma: number;
  onColormapChange: (c: ColormapName) => void;
  onGammaChange: (g: number) => void;
  /** Nonlinear display preprocessors (display-only; do not affect export). */
  clahe: boolean;
  sharpen: boolean;
  onClaheChange: (v: boolean) => void;
  onSharpenChange: (v: boolean) => void;
  /** Gaussian pre-blur sigma in image pixels (0 = off) — denoises what the
   *  intensity-driven tools see, as well as the display. */
  blur: number;
  onBlurChange: (v: number) => void;
  /** Working resolution multiplier (1, 2, 4) for the drawing tools. */
  upscale: number;
  onUpscaleChange: (v: number) => void;
  /** Highest upscale this slice can afford before the guard clamps it. */
  maxUpscale?: number;
}

const UPSCALES = [1, 2, 4] as const;

/** Renders the brightness/contrast sliders + levels histogram with reset buttons. */
export default function DisplayControls({
  brightness,
  contrast,
  onBrightnessChange,
  onContrastChange,
  onReset,
  histogramBins,
  levelsLo,
  levelsHi,
  onLevelsChange,
  onLevelsReset,
  colormap,
  gamma,
  onColormapChange,
  onGammaChange,
  clahe,
  sharpen,
  onClaheChange,
  onSharpenChange,
  blur,
  onBlurChange,
  upscale,
  onUpscaleChange,
  maxUpscale = 4,
}: DisplayControlsProps) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Display</span>
        <button
          aria-label="Reset brightness and contrast"
          title="Reset display"
          onClick={onReset}
          className="p-0.5 rounded hover:bg-gray-100 hover:text-sky-600"
        >
          <ArrowCounterClockwise size={14} />
        </button>
      </div>

      <div className="flex flex-col gap-0.5">
        <DebouncedSlider
          label="Brightness"
          format={(v) => `${v > 0 ? '+' : ''}${v}`}
          min={-1}
          max={1}
          step={0.05}
          value={brightness}
          onChange={onBrightnessChange}
          debounceMs={0}
        />
      </div>

      <div className="flex flex-col gap-0.5">
        <DebouncedSlider
          label="Contrast"
          format={(v) => `${v > 0 ? '+' : ''}${v}`}
          min={-100}
          max={100}
          step={1}
          value={contrast}
          onChange={onContrastChange}
          debounceMs={0}
        />
      </div>

      <HistogramControl
        bins={histogramBins}
        lo={levelsLo}
        hi={levelsHi}
        onChange={onLevelsChange}
        onReset={onLevelsReset}
      />

      {/* Colormap (false-color LUT) + gamma — display only. */}
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-gray-500">Colormap</span>
        <div className="flex items-center gap-1">
          {COLORMAP_NAMES.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => onColormapChange(name)}
              title={name}
              aria-pressed={colormap === name}
              className={[
                'h-5 flex-1 rounded border',
                colormap === name ? 'border-sky-500 ring-1 ring-sky-400' : 'border-gray-200',
              ].join(' ')}
              style={{ backgroundImage: colormapGradient(name) }}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-0.5">
        <DebouncedSlider
          label="Gamma"
          format={(v) => v.toFixed(2)}
          min={0.2}
          max={3}
          step={0.05}
          value={gamma}
          onChange={onGammaChange}
          debounceMs={0}
          allowLog
        />
      </div>

      {/* Gaussian pre-blur — denoises so threshold/wand/fill see coherent regions. */}
      <div className="flex flex-col gap-0.5">
        <DebouncedSlider
          label="Blur (σ)"
          format={(v) => (v === 0 ? 'off' : v.toFixed(2))}
          min={0}
          max={5}
          step={0.25}
          value={blur}
          onChange={onBlurChange}
          // Each commit re-blurs the whole slice (and invalidates every tool
          // field), so commit on pause rather than on every tick.
          debounceMs={200}
        />
      </div>

      {/* Nonlinear enhancers (display-only; can combine: Blur → CLAHE → Sharpen). */}
      <div className="flex items-center gap-3 pt-0.5">
        <label
          className="flex items-center gap-1.5 text-[11px] text-gray-600 cursor-pointer"
          title="CLAHE — adaptive (local) contrast: equalizes each tile's histogram with a clip limit"
        >
          <input type="checkbox" checked={clahe} onChange={(e) => onClaheChange(e.target.checked)} className="accent-sky-600" />
          CLAHE
        </label>
        <label
          className="flex items-center gap-1.5 text-[11px] text-gray-600 cursor-pointer"
          title="3×3 Laplacian high-boost sharpen"
        >
          <input type="checkbox" checked={sharpen} onChange={(e) => onSharpenChange(e.target.checked)} className="accent-sky-600" />
          Sharpen
        </label>
      </div>

      {/* Working resolution — resamples the slice for the drawing tools so small
          features get more pixels to annotate against. Coordinates stay native. */}
      <div className="flex flex-col gap-1 pt-0.5">
        <span className="text-[11px] text-gray-500">Working resolution</span>
        <div role="radiogroup" aria-label="Working resolution" className="flex items-center gap-1">
          {UPSCALES.map((u) => {
            const tooBig = u > maxUpscale;
            return (
              <button
                key={u}
                type="button"
                role="radio"
                aria-checked={upscale === u}
                disabled={tooBig}
                onClick={() => onUpscaleChange(u)}
                title={tooBig
                  ? `${u}× needs more memory than this slice size allows`
                  : u === 1
                    ? 'Native resolution'
                    : `Resample ${u}× for the drawing tools — sub-pixel brush and mask detail`}
                className={[
                  'flex-1 py-1 rounded-md text-[11px] border transition-colors',
                  tooBig
                    ? 'bg-gray-50 text-gray-300 border-gray-100 cursor-not-allowed'
                    : upscale === u
                      ? 'bg-sky-600 text-white border-sky-700'
                      : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50',
                ].join(' ')}
              >
                {u}×
              </button>
            );
          })}
        </div>
        <p className="text-[10px] text-gray-500 leading-snug">
          Resamples the slice for the drawing tools so smaller features can be annotated.
          Exported pixels and annotation coordinates are unchanged.
        </p>
      </div>
    </div>
  );
}
