/**
 * SliceNavigator — slice slider, prev/next, annotated-slice jump, negative-slice toggle.
 */
import { CaretLeft, CaretRight, WarningCircle } from '@phosphor-icons/react';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { buildSourceKey } from '@/lib/sourceKey';
import DebouncedSlider from '@/components/common/DebouncedSlider';

/** Renders slice navigation controls bound to the dataset and annotation stores. */
export default function SliceNavigator() {
  const { meta, currentSlice, source, kind, serverUri, setSlice } = useDatasetStore();
  const { byImage, negativeSlices, toggleNegativeSlice } = useAnnotationStore();

  const sourceKey = source && kind
    ? buildSourceKey(kind as 'tiled' | 'local', source, serverUri)
    : null;

  if (!meta || !sourceKey) {
    return <p className="text-xs text-gray-400">No dataset loaded.</p>;
  }

  const n = meta.nSlices;
  const slices = byImage[sourceKey] ?? {};
  const annotatedIndices = Object.keys(slices)
    .map(Number)
    .filter((i) => slices[String(i)]?.length > 0)
    .sort((a, b) => a - b);

  const isNegative = (negativeSlices[sourceKey] ?? []).includes(String(currentSlice));

  /** Steps back one slice (clamped at 0); writes to the dataset store. */
  const prev = () => setSlice(Math.max(0, currentSlice - 1));
  /** Steps forward one slice (clamped at the last slice); writes to the dataset store. */
  const next = () => setSlice(Math.min(n - 1, currentSlice + 1));

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-semibold uppercase text-gray-500 tracking-wide">
        Slice {currentSlice + 1} / {n}
      </span>

      <DebouncedSlider
        min={0}
        max={n - 1}
        value={currentSlice}
        onChange={setSlice}
        debounceMs={120}
        ariaLabel="Slice index"
      />

      <div className="flex items-center gap-1">
        <button
          aria-label="Previous slice"
          onClick={prev}
          disabled={currentSlice === 0}
          className="p-1 rounded hover:bg-gray-100 disabled:opacity-40"
        >
          <CaretLeft size={16} />
        </button>
        <button
          aria-label="Next slice"
          onClick={next}
          disabled={currentSlice === n - 1}
          className="p-1 rounded hover:bg-gray-100 disabled:opacity-40"
        >
          <CaretRight size={16} />
        </button>

        {annotatedIndices.length > 0 && (
          <select
            className="text-xs border rounded px-1 py-0.5 flex-1"
            value=""
            onChange={(e) => { if (e.target.value !== '') setSlice(Number(e.target.value)); }}
            aria-label="Jump to annotated slice"
          >
            <option value="">Jump to annotated…</option>
            {annotatedIndices.map((i) => (
              <option key={i} value={i}>
                Slice {i + 1} ({slices[String(i)].length} shapes)
              </option>
            ))}
          </select>
        )}
      </div>

      <button
        aria-pressed={isNegative}
        onClick={() => toggleNegativeSlice(sourceKey, currentSlice)}
        className={`flex items-center gap-1 text-xs px-2 py-1 rounded border transition-colors ${
          isNegative
            ? 'bg-amber-100 border-amber-400 text-amber-800'
            : 'bg-white border-gray-200 text-gray-600 hover:border-amber-300'
        }`}
        title="Mark as negative example (exported with zero annotations)"
      >
        <WarningCircle size={12} />
        {isNegative ? 'Negative example' : 'Mark as negative'}
      </button>
    </div>
  );
}
