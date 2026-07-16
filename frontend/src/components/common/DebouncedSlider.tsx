/**
 * DebouncedSlider — a range input whose thumb and label update instantly, but
 * whose committed `onChange` is debounced. Use it for sliders that trigger
 * expensive work (image re-render, layer re-cache, network fetch, segmentation)
 * so dragging stays smooth and the heavy work runs once the user pauses.
 *
 * With `allowLog`, a small "Log" checkbox toggles logarithmic response: the thumb
 * maps to the value on a log curve (`value = min·(max/min)^t`), useful for
 * wide-dynamic-range controls. The committed value is always the real value, so
 * nothing downstream changes. Falls back to linear if the range isn't all-positive.
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
  /** Show a "Log" checkbox that switches the slider to logarithmic response. */
  allowLog?: boolean;
  /** Initial state of the log toggle (only when `allowLog`). */
  defaultLog?: boolean;
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
  allowLog = false,
  defaultLog = false,
}: DebouncedSliderProps) {
  const [local, setLocal] = useState(value);
  const [logOn, setLogOn] = useState(defaultLog);
  const timer = useRef<number | null>(null);

  // Track external changes (reset buttons, keyboard nav, etc.) when not dragging.
  useEffect(() => {
    setLocal(value);
  }, [value]);

  useEffect(() => () => { if (timer.current !== null) window.clearTimeout(timer.current); }, []);

  // Log mapping only valid for an all-positive range; otherwise stay linear.
  const logUsable = allowLog && logOn && min > 0 && max > min;
  const toPos = (v: number) => (logUsable ? Math.log(v / min) / Math.log(max / min) : v);
  const fromPos = (p: number) => (logUsable ? min * Math.pow(max / min, p) : p);

  /** Update the local value immediately and (re)arm the debounced onChange commit. */
  const handle = (next: number) => {
    setLocal(next); // instant thumb + label feedback
    // debounceMs <= 0 → commit synchronously (no debounce), for cheap live updates.
    if (debounceMs <= 0) { onChange(next); return; }
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => onChange(next), debounceMs);
  };

  const showHeader = label !== undefined || format || allowLog;

  return (
    <>
      {showHeader && (
        <div className="flex items-center justify-between gap-2">
          <label className={labelClassName}>
            {label}
            {format ? <>{label !== undefined ? ': ' : ''}{format(local)}</> : null}
          </label>
          {allowLog && (
            <label
              className="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer select-none"
              title="Logarithmic slider scaling"
            >
              <input
                type="checkbox"
                checked={logOn}
                onChange={(e) => setLogOn(e.target.checked)}
                className="accent-sky-600"
              />
              Log
            </label>
          )}
        </div>
      )}
      <input
        type="range"
        min={logUsable ? 0 : min}
        max={logUsable ? 1 : max}
        step={logUsable ? 0.001 : step}
        value={toPos(local)}
        onChange={(e) => handle(fromPos(Number(e.target.value)))}
        className={className}
        aria-label={ariaLabel ?? (typeof label === 'string' ? label : undefined)}
      />
    </>
  );
}
