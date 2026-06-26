/**
 * VersionPreviewBar — a "time-travel" overlay for scrubbing saved versions.
 *
 * Renders over the canvas. Dragging the slider previews each version on the
 * canvas (non-destructive — current work is untouched). Exit returns to the
 * live editor; Restore loads the previewed version into the editor.
 */
import { ArrowCounterClockwise, X, CaretLeft, CaretRight } from '@phosphor-icons/react';
import type { VersionMeta } from '@/hooks/useSave';
import DebouncedSlider from '@/components/common/DebouncedSlider';

interface Props {
  versions: VersionMeta[];
  /** Currently previewed version number. */
  current: number;
  /** Whether the previewed payload is still loading. */
  loading?: boolean;
  onChange: (version: number) => void;
  onExit: () => void;
  onRestore: (version: number) => void;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export default function VersionPreviewBar({
  versions,
  current,
  loading = false,
  onChange,
  onExit,
  onRestore,
}: Props) {
  if (versions.length === 0) return null;

  // Versions are stored oldest-first; the slider maps 1:1 to version numbers.
  const sorted = [...versions].sort((a, b) => a.version - b.version);
  const min = sorted[0].version;
  const max = sorted[sorted.length - 1].version;
  const meta = sorted.find((v) => v.version === current) ?? sorted[sorted.length - 1];
  const isLatest = current === max;

  const step = (dir: -1 | 1) => {
    const next = current + dir;
    if (next >= min && next <= max) onChange(next);
  };

  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 w-[min(640px,calc(100%-2rem))]">
      <div className="rounded-xl bg-gray-900/90 backdrop-blur text-white shadow-2xl border border-white/10 px-4 py-3">
        {/* Top row: label + actions */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold">
              Previewing v{meta.version}
              {isLatest && <span className="ml-1.5 text-[10px] font-medium text-sky-300">(latest)</span>}
            </span>
            <span className="text-xs text-gray-300">{formatDate(meta.saved_at)}</span>
            <span className="text-xs text-gray-400">
              {meta.shape_count} shape{meta.shape_count !== 1 ? 's' : ''}
            </span>
            {loading && <span className="text-xs text-sky-300 animate-pulse">loading…</span>}
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => onRestore(current)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-sky-600 hover:bg-sky-500 text-xs font-medium transition-colors"
              title={`Restore v${current} into the editor`}
            >
              <ArrowCounterClockwise size={14} />
              Restore this version
            </button>
            <button
              type="button"
              onClick={onExit}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/10 hover:bg-white/20 text-xs font-medium transition-colors"
              title="Exit preview"
            >
              <X size={14} />
              Exit
            </button>
          </div>
        </div>

        {/* Slider row */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => step(-1)}
            disabled={current <= min}
            className="p-1 rounded hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Older version"
          >
            <CaretLeft size={16} />
          </button>

          <span className="text-xs text-gray-400 w-8 text-right tabular-nums">v{min}</span>

          <DebouncedSlider
            min={min}
            max={max}
            step={1}
            value={current}
            onChange={onChange}
            debounceMs={150}
            className="flex-1 accent-sky-500 cursor-pointer"
            ariaLabel="Preview version"
          />

          <span className="text-xs text-gray-400 w-8 tabular-nums">v{max}</span>

          <button
            type="button"
            onClick={() => step(1)}
            disabled={current >= max}
            className="p-1 rounded hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Newer version"
          >
            <CaretRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
