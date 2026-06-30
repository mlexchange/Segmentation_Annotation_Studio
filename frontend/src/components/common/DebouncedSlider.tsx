/**
 * DebouncedSlider — a range input whose thumb and label update instantly, but
 * whose committed `onChange` is debounced. Use it for sliders that trigger
 * expensive work (image re-render, layer re-cache, network fetch, segmentation)
 * so dragging stays smooth and the heavy work runs once the user pauses.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';

interface DebouncedSliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  debounceMs?: number;
  /** Static label text shown before the live value. */
  label?: ReactNode;
  /** Formats the live (local) value for the label, e.g. (v) => `${v}%`. */
  format?: (value: number) => ReactNode;
  className?: string;
  labelClassName?: string;
  ariaLabel?: string;
}

/** Range input with instant local feedback and a debounced `onChange` commit. */
export default function DebouncedSlider({
  value,
  min,
  max,
  step = 1,
  onChange,
  debounceMs = 120,
  label,
  format,
  className = 'w-full',
  labelClassName = 'text-xs text-gray-500',
  ariaLabel,
}: DebouncedSliderProps) {
  const [local, setLocal] = useState(value);
  const timer = useRef<number | null>(null);

  // Track external changes (reset buttons, keyboard nav, etc.) when not dragging.
  useEffect(() => {
    setLocal(value);
  }, [value]);

  useEffect(() => () => { if (timer.current !== null) window.clearTimeout(timer.current); }, []);

  /** Update the local value immediately and (re)arm the debounced onChange commit. */
  const handle = (next: number) => {
    setLocal(next); // instant thumb + label feedback
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => onChange(next), debounceMs);
  };

  return (
    <>
      {(label !== undefined || format) && (
        <label className={labelClassName}>
          {label}
          {format ? <>{label !== undefined ? ': ' : ''}{format(local)}</> : null}
        </label>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={local}
        onChange={(e) => handle(Number(e.target.value))}
        className={className}
        aria-label={ariaLabel}
      />
    </>
  );
}
