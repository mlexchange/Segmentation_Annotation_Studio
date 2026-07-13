/**
 * HistogramControl — a compact intensity histogram (2D plot) with a single
 * dual-knob range slider laid over the plot: drag the left/right handles (or the
 * band between them) to set the min/max window that drives a client-side levels
 * remap of the displayed image (no refetch). Knobs align with the 256-bin plot.
 */
import { useEffect, useMemo, useRef } from 'react';
import { ArrowCounterClockwise } from '@phosphor-icons/react';

export interface HistogramControlProps {
  /** 256-bin luminance histogram of the current slice, or null before load. */
  bins: number[] | null;
  lo: number;
  hi: number;
  onChange: (lo: number, hi: number) => void;
  onReset: () => void;
}

const VBW = 256;
const VBH = 60;
const MAXV = 255;

export default function HistogramControl({ bins, lo, hi, onChange, onReset }: HistogramControlProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  // Active drag target + the window state captured at press (for band drags).
  const drag = useRef<{ target: 'lo' | 'hi' | 'band'; startV: number; startLo: number; startHi: number } | null>(null);

  // Log-scaled bar heights so sparse bright tails stay visible.
  const path = useMemo(() => {
    if (!bins || bins.length === 0) return '';
    const max = Math.max(1, ...bins.map((b) => Math.log1p(b)));
    return bins
      .map((b, i) => {
        const h = (Math.log1p(b) / max) * VBH;
        return `M${i},${VBH} L${i},${VBH - h}`;
      })
      .join(' ');
  }, [bins]);

  const clampLo = (v: number) => Math.max(0, Math.min(v, hi - 1));
  const clampHi = (v: number) => Math.min(MAXV, Math.max(v, lo + 1));

  // Coalesce drag updates to one per animation frame (throttle, not debounce):
  // pointermove can fire faster than the display refresh, so we collapse bursts
  // into a single onChange per frame to keep React updates minimal + smooth.
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<[number, number] | null>(null);
  const scheduleChange = (nlo: number, nhi: number) => {
    pendingRef.current = [nlo, nhi];
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        if (pendingRef.current) onChange(pendingRef.current[0], pendingRef.current[1]);
      });
    }
  };
  useEffect(() => () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); }, []);

  /** Pointer clientX → intensity value (0–255) across the track width. */
  const xToValue = (clientX: number): number => {
    const el = trackRef.current;
    if (!el) return lo;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return lo;
    const t = (clientX - rect.left) / rect.width;
    return Math.round(Math.max(0, Math.min(1, t)) * MAXV);
  };

  const applyDrag = (clientX: number) => {
    if (!drag.current) return;
    const v = xToValue(clientX);
    if (drag.current.target === 'lo') {
      scheduleChange(clampLo(v), hi);
    } else if (drag.current.target === 'hi') {
      scheduleChange(lo, clampHi(v));
    } else {
      // Band: translate the whole window, preserving its width.
      const width = drag.current.startHi - drag.current.startLo;
      let nlo = drag.current.startLo + (v - drag.current.startV);
      nlo = Math.max(0, Math.min(nlo, MAXV - width));
      scheduleChange(nlo, nlo + width);
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const v = xToValue(e.clientX);
    const dLo = Math.abs(v - lo);
    const dHi = Math.abs(v - hi);
    // Inside the band and clearly away from both knobs → drag the window.
    let target: 'lo' | 'hi' | 'band';
    if (v > lo && v < hi && Math.min(dLo, dHi) > 6) target = 'band';
    else target = dLo <= dHi ? 'lo' : 'hi';
    drag.current = { target, startV: v, startLo: lo, startHi: hi };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    applyDrag(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (drag.current) applyDrag(e.clientX);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    drag.current = null;
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId); } catch { /* noop */ }
  };

  /** Keyboard nudge for a focused knob (±1, Shift ±10). */
  const onKnobKey = (which: 'lo' | 'hi') => (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 1;
    let delta = 0;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') delta = -step;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') delta = step;
    else return;
    e.preventDefault();
    if (which === 'lo') onChange(clampLo(lo + delta), hi);
    else onChange(lo, clampHi(hi + delta));
  };

  const pct = (v: number) => `${(v / MAXV) * 100}%`;
  const knobCls =
    'absolute top-0 h-full w-3 -translate-x-1/2 flex items-center justify-center cursor-ew-resize';
  const knobGrip = 'w-1.5 h-5 rounded-sm bg-sky-600 border border-white shadow';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-gray-500">
          Levels <span className="tabular-nums text-gray-400">{lo} – {hi}</span>
        </span>
        <button
          aria-label="Reset levels"
          title="Reset levels"
          onClick={onReset}
          className="p-0.5 rounded hover:bg-gray-100 hover:text-sky-600"
        >
          <ArrowCounterClockwise size={12} />
        </button>
      </div>

      {/* Track: histogram plot + dual-knob range slider overlaid on it. */}
      <div
        ref={trackRef}
        className="relative w-full h-12 touch-none select-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <svg
          viewBox={`0 0 ${VBW} ${VBH}`}
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full rounded bg-gray-100 border border-gray-200"
        >
          <rect x={lo} y={0} width={Math.max(0, hi - lo)} height={VBH} fill="#38bdf8" opacity={0.15} />
          <path d={path} stroke="#64748b" strokeWidth={1} fill="none" vectorEffect="non-scaling-stroke" />
          <line x1={lo} y1={0} x2={lo} y2={VBH} stroke="#0284c7" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          <line x1={hi} y1={0} x2={hi} y2={VBH} stroke="#0284c7" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        </svg>

        {/* Knob grips (crisp HTML overlay, aligned to the plot by percentage). */}
        <div
          className={knobCls}
          style={{ left: pct(lo) }}
          role="slider"
          tabIndex={0}
          aria-label="Levels minimum"
          aria-valuemin={0}
          aria-valuemax={MAXV}
          aria-valuenow={lo}
          onKeyDown={onKnobKey('lo')}
        >
          <span className={knobGrip} />
        </div>
        <div
          className={knobCls}
          style={{ left: pct(hi) }}
          role="slider"
          tabIndex={0}
          aria-label="Levels maximum"
          aria-valuemin={0}
          aria-valuemax={MAXV}
          aria-valuenow={hi}
          onKeyDown={onKnobKey('hi')}
        >
          <span className={knobGrip} />
        </div>
      </div>
    </div>
  );
}
