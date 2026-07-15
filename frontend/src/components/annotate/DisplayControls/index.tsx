/**
 * DisplayControls — brightness/contrast, CLAHE / Sharpen toggles, levels histogram.
 * Display-only. Does NOT affect exported pixel values (RenderOpts).
 */
import { ArrowCounterClockwise, CircleHalf, Sparkle } from '@phosphor-icons/react';
import DebouncedSlider from '@/components/common/DebouncedSlider';
import HistogramControl from '@/components/annotate/HistogramControl';
import { cn } from '@/lib/utils';

export interface DisplayControlsProps {
  brightness: number;
  contrast: number;
  onBrightnessChange: (v: number) => void;
  onContrastChange: (v: number) => void;
  onReset: () => void;
  histogramBins: number[] | null;
  levelsLo: number;
  levelsHi: number;
  onLevelsChange: (lo: number, hi: number) => void;
  onLevelsReset: () => void;
  clahe: boolean;
  onClaheChange: (on: boolean) => void;
  sharpen: boolean;
  onSharpenChange: (on: boolean) => void;
}

/** Renders display preprocessor toggles + brightness/contrast + levels. */
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
  clahe,
  onClaheChange,
  sharpen,
  onSharpenChange,
}: DisplayControlsProps) {
  const toggleCls = (on: boolean) =>
    cn(
      'flex items-center justify-center gap-1.5 flex-1 py-1.5 rounded-md text-xs border transition-colors',
      on
        ? 'bg-sky-600 text-white border-sky-700'
        : 'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300',
    );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">Display</span>
        <button
          type="button"
          aria-label="Reset display"
          title="Reset display"
          onClick={onReset}
          className="p-0.5 rounded hover:bg-gray-100 hover:text-sky-600"
        >
          <ArrowCounterClockwise size={14} />
        </button>
      </div>

      <div className="flex gap-1">
        <button
          type="button"
          aria-pressed={clahe}
          title="CLAHE — adaptive histogram equalization (local contrast)"
          onClick={() => onClaheChange(!clahe)}
          className={toggleCls(clahe)}
        >
          <CircleHalf size={14} weight={clahe ? 'fill' : 'regular'} />
          CLAHE
        </button>
        <button
          type="button"
          aria-pressed={sharpen}
          title="Sharpen — classic 3×3 Laplacian high-boost (fast)"
          onClick={() => onSharpenChange(!sharpen)}
          className={toggleCls(sharpen)}
        >
          <Sparkle size={14} weight={sharpen ? 'fill' : 'regular'} />
          Sharpen
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
    </div>
  );
}
