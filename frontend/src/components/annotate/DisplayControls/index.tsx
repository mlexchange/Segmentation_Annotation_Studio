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
}

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

      {/* Nonlinear enhancers (display-only; can combine: CLAHE → Sharpen). */}
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
    </div>
  );
}
