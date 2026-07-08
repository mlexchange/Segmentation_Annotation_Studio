/**
 * HistogramControl — a compact intensity histogram (2D plot) with a min/max
 * window that drives a client-side levels remap of the displayed image (no
 * refetch). The shaded band shows the current [lo,hi] window; two sliders set it.
 */
import { useMemo } from 'react';
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

export default function HistogramControl({ bins, lo, hi, onChange, onReset }: HistogramControlProps) {
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
  const clampHi = (v: number) => Math.min(255, Math.max(v, lo + 1));

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-gray-500">Levels (min / max)</span>
        <button
          aria-label="Reset levels"
          title="Reset levels"
          onClick={onReset}
          className="p-0.5 rounded hover:bg-gray-100 hover:text-sky-600"
        >
          <ArrowCounterClockwise size={12} />
        </button>
      </div>

      <svg
        viewBox={`0 0 ${VBW} ${VBH}`}
        preserveAspectRatio="none"
        className="w-full h-12 rounded bg-gray-100 border border-gray-200"
      >
        {/* Selected window band */}
        <rect x={lo} y={0} width={Math.max(0, hi - lo)} height={VBH} fill="#38bdf8" opacity={0.15} />
        {/* Histogram bars */}
        <path d={path} stroke="#64748b" strokeWidth={1} fill="none" vectorEffect="non-scaling-stroke" />
        {/* Handle lines */}
        <line x1={lo} y1={0} x2={lo} y2={VBH} stroke="#0284c7" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        <line x1={hi} y1={0} x2={hi} y2={VBH} stroke="#0284c7" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      </svg>

      <div className="flex items-center gap-2">
        <input
          type="range" min={0} max={255} step={1} value={lo}
          onChange={(e) => onChange(clampLo(Number(e.target.value)), hi)}
          className="flex-1 accent-sky-600"
          aria-label="Levels minimum"
        />
        <span className="w-8 text-right text-[10px] tabular-nums text-gray-500">{lo}</span>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range" min={0} max={255} step={1} value={hi}
          onChange={(e) => onChange(lo, clampHi(Number(e.target.value)))}
          className="flex-1 accent-sky-600"
          aria-label="Levels maximum"
        />
        <span className="w-8 text-right text-[10px] tabular-nums text-gray-500">{hi}</span>
      </div>
    </div>
  );
}
