/**
 * MaskToolsPanel — sidebar controls for cross-slice operations (copy /
 * interpolate) on the ACTIVE class. Ops round-trip the class's shapes through a
 * binary mask; see `useMaskOps`.
 */
import { useMemo, useState } from 'react';
import { Stack } from '@phosphor-icons/react';
import { useAnnotationStore } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useClassStore } from '@/stores/classStore';
import { useMaskOps } from '@/hooks/useMaskOps';

interface MaskToolsPanelProps {
  sourceKey: string | null;
  activeClassId: number | null;
}

export default function MaskToolsPanel({ sourceKey, activeClassId }: MaskToolsPanelProps) {
  const byImage = useAnnotationStore((s) => s.byImage);
  const classes = useClassStore((s) => s.classes);
  const { meta, currentSlice } = useDatasetStore();
  const { copyToNext } = useMaskOps(sourceKey, activeClassId);

  const [status, setStatus] = useState<string | null>(null);

  const slices = sourceKey ? byImage[sourceKey] ?? {} : {};
  const className = classes.find((c) => c.classId === activeClassId)?.label ?? 'the active class';

  // Active-class shape count on the current slice (enables copy).
  const hasClassHere = useMemo(
    () =>
      activeClassId !== null &&
      (slices[String(currentSlice)] ?? []).some((sh) => sh.classId === activeClassId),
    [slices, currentSlice, activeClassId],
  );

  const isVolume = (meta?.nSlices ?? 1) > 1;
  const disabled = !sourceKey || activeClassId === null;

  const btn = 'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
    'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

  if (!isVolume) return null;

  return (
    <div className="flex flex-col gap-2">
      <span className="flex items-center gap-1.5 text-xs font-semibold uppercase text-gray-500 tracking-wide">
        <Stack size={13} /> Cross-slice
      </span>
      <p className="text-[10px] text-gray-400 -mt-1">Acts on ‘{className}’.</p>

      <button
        className={btn}
        disabled={disabled || !hasClassHere}
        onClick={() => {
          const ok = copyToNext();
          setStatus(ok ? `Copied ‘${className}’ to slice ${currentSlice + 2}` : 'Nothing to copy / at last slice');
        }}
      >
        Copy ‘{className}’ → next slice
      </button>

      {status && <p className="text-[10px] text-gray-500 leading-snug">{status}</p>}
    </div>
  );
}
