/**
 * DisplayControls — brightness/contrast sliders (display-only via Konva filters).
 * Does NOT affect exported pixel values — that is governed by RenderOpts.
 */
import { ArrowCounterClockwise } from '@phosphor-icons/react';
import DebouncedSlider from '@/components/common/DebouncedSlider';

export interface DisplayControlsProps {
  brightness: number;
  contrast: number;
  onBrightnessChange: (v: number) => void;
  onContrastChange: (v: number) => void;
  onReset: () => void;
}

/** Renders the brightness/contrast sliders with a reset button. */
export default function DisplayControls({
  brightness,
  contrast,
  onBrightnessChange,
  onContrastChange,
  onReset,
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
        />
      </div>
    </div>
  );
}
