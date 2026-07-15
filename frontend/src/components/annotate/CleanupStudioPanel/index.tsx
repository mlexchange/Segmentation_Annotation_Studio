/**
 * CleanupStudioPanel — size/class filter, ROI keep/delete, morphology, mask sets.
 */
import { useMemo, useState } from 'react';
import {
  Broom,
  Copy,
  Eye,
  EyeSlash,
  Stack,
  Trash,
} from '@phosphor-icons/react';
import { useClassStore } from '@/stores/classStore';
import { useToolStore } from '@/stores/toolStore';
import {
  useCleanupStudio,
  type CleanupTarget,
} from '@/hooks/useCleanupStudio';
import type { MorphCleanupOp, RoiMode, SizeFilterMode } from '@/lib/cleanupOps';
import type { MergePolicy } from '@/lib/maskSetMerge';
import type { Shape } from '@/stores/annotationStore';
import { cn } from '@/lib/utils';

interface CleanupStudioPanelProps {
  sourceKey: string | null;
  activeClassId: number | null;
}

const btn =
  'px-2 py-1 rounded border text-[11px] transition-colors disabled:opacity-40 ' +
  'bg-white text-gray-700 border-gray-200 hover:bg-sky-50 hover:border-sky-300';

/** Sidebar Cleanup Studio controls. */
export default function CleanupStudioPanel({
  sourceKey,
  activeClassId,
}: CleanupStudioPanelProps) {
  const classes = useClassStore((s) => s.classes);
  const selectedShapeIds = useToolStore((s) => s.selectedShapeIds);
  const studio = useCleanupStudio(sourceKey);

  const [target, setTarget] = useState<CleanupTarget>('mask_set');
  const [classFilter, setClassFilter] = useState<'all' | number>(
    activeClassId ?? 'all',
  );
  const [minArea, setMinArea] = useState(64);
  const [maxArea, setMaxArea] = useState(1_000_000);
  const [sizeMode, setSizeMode] = useState<SizeFilterMode>('keep');
  const [roiMode, setRoiMode] = useState<RoiMode>('keep_inside');
  const [allClassesRoi, setAllClassesRoi] = useState(false);
  const [morphOp, setMorphOp] = useState<MorphCleanupOp>('islands');
  const [morphParam, setMorphParam] = useState(50);
  const [mergePolicy, setMergePolicy] = useState<MergePolicy>('union');
  const [status, setStatus] = useState<string | null>(null);
  const [roiShape, setRoiShape] = useState<Shape | null>(null);

  const classId = classFilter === 'all' ? null : classFilter;
  const activePx = studio.activeSet?.labelMap
    ? studio.labelMapPixelCount(studio.activeSet.labelMap)
    : studio.activeSet?.shapes.length ?? 0;
  const workingOk =
    target === 'annotations'
      ? studio.sliceShapes.length > 0
      : !!studio.activeSet && (activePx > 0 || (studio.activeSet.shapes?.length ?? 0) > 0);

  const stats = useMemo(
    () => studio.sizeStats(target, classId),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional refresh on shapes
    [studio, target, classId, studio.sliceShapes, studio.activeSet],
  );

  const captureRoiFromSelection = () => {
    const shapes =
      target === 'mask_set' ? studio.activeSet?.shapes ?? [] : studio.sliceShapes;
    const selected = shapes.filter((s) => selectedShapeIds.includes(s.id));
    if (!selected.length) {
      setStatus('Select a polygon / rect / ellipse first (Select tool)');
      return;
    }
    setRoiShape(selected[0]);
    setStatus(`ROI captured: ${selected[0].kind}`);
  };

  const resolveClassForRoi = () => (allClassesRoi ? null : (activeClassId ?? classId));

  return (
    <div className="flex flex-col gap-2">
      <span className="flex items-center gap-1.5 text-xs font-semibold uppercase text-gray-500 tracking-wide">
        <Broom size={13} /> Cleanup Studio
      </span>
      <p className="text-[10px] text-gray-400 -mt-1 leading-snug">
        Edits dense pixel label maps (not polygons). Commit classifier → Active set, clean,
        then Merge → slice (vectorizes once).
      </p>

      {/* Target */}
      <div className="flex gap-1">
        {(['annotations', 'mask_set'] as const).map((t) => (
          <button
            key={t}
            type="button"
            className={cn(
              btn,
              'flex-1',
              target === t && 'bg-sky-50 border-sky-300 text-sky-800',
            )}
            onClick={() => setTarget(t)}
          >
            {t === 'annotations' ? 'Annotations' : 'Active set'}
          </button>
        ))}
      </div>

      {/* Mask sets */}
      <div className="flex flex-col gap-1 rounded-md border border-gray-100 bg-gray-50 p-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-medium text-gray-600 flex items-center gap-1">
            <Stack size={12} /> Mask sets
          </span>
          <button
            type="button"
            className={btn}
            disabled={!studio.sliceShapes.length}
            onClick={() => {
              const id = studio.copyAnnotationsToSet();
              setStatus(id ? 'Copied annotations → new set' : 'Nothing to copy');
              if (id) setTarget('mask_set');
            }}
          >
            <Copy size={12} className="inline mr-0.5" />
            From annotations
          </button>
        </div>
        {studio.sliceSets.length === 0 && (
          <p className="text-[10px] text-gray-400">No sets on this slice yet.</p>
        )}
        <ul className="flex flex-col gap-0.5 max-h-28 overflow-y-auto">
          {studio.sliceSets.map((m) => (
            <li
              key={m.id}
              className={cn(
                'flex items-center gap-1 text-[10px] rounded px-1 py-0.5',
                studio.activeSetId === m.id ? 'bg-sky-100' : 'hover:bg-white',
              )}
            >
              <button
                type="button"
                className="flex-1 text-left truncate"
                onClick={() => studio.setActiveSetId(m.id)}
                title={m.name}
              >
                {m.name}
                <span className="text-gray-400">
                  {' '}· {m.labelMap
                    ? `${studio.labelMapPixelCount(m.labelMap).toLocaleString()} px`
                    : `${m.shapes.length} sh`}{' '}· {m.origin}
                </span>
              </button>
              <button
                type="button"
                title={m.visible ? 'Hide' : 'Show'}
                onClick={() => studio.setVisible(m.id, !m.visible)}
              >
                {m.visible ? <Eye size={12} /> : <EyeSlash size={12} />}
              </button>
              <button type="button" title="Delete" onClick={() => studio.deleteSet(m.id)}>
                <Trash size={12} />
              </button>
            </li>
          ))}
        </ul>
        {studio.activeSet && (
          <input
            className="rounded border border-gray-200 px-1 py-0.5 text-[10px]"
            value={studio.activeSet.name}
            onChange={(e) => studio.renameSet(studio.activeSet!.id, e.target.value)}
          />
        )}
        <div className="flex gap-1 items-center">
          <select
            className="flex-1 rounded border border-gray-200 text-[10px] px-1 py-0.5"
            value={mergePolicy}
            onChange={(e) => setMergePolicy(e.target.value as MergePolicy)}
          >
            <option value="union">Merge: union</option>
            <option value="replace_class">Merge: replace class</option>
            <option value="delete_where_set">Merge: delete where set</option>
          </select>
          <button
            type="button"
            className={btn}
            disabled={!studio.activeSet}
            onClick={() => {
              const ok = studio.mergeActiveIntoAnnotations(mergePolicy);
              setStatus(ok ? `Merged set (${mergePolicy})` : 'Merge failed');
            }}
          >
            Merge → slice
          </button>
        </div>
      </div>

      {/* Class + size */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium text-gray-600">By size (connected components)</span>
        <select
          className="rounded border border-gray-200 text-[10px] px-1 py-0.5"
          value={classFilter === 'all' ? 'all' : String(classFilter)}
          onChange={(e) =>
            setClassFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))
          }
        >
          <option value="all">All classes</option>
          {classes.map((c) => (
            <option key={c.classId} value={c.classId}>
              {c.label}
            </option>
          ))}
        </select>
        <div className="grid grid-cols-2 gap-1 text-[10px]">
          <label className="flex flex-col gap-0.5">
            Min area (px²)
            <input
              type="number"
              min={1}
              value={minArea}
              onChange={(e) => setMinArea(Math.max(1, Number(e.target.value) || 1))}
              className="rounded border border-gray-200 px-1 py-0.5"
            />
          </label>
          <label className="flex flex-col gap-0.5">
            Max area (px²)
            <input
              type="number"
              min={1}
              value={maxArea}
              onChange={(e) => setMaxArea(Math.max(1, Number(e.target.value) || 1))}
              className="rounded border border-gray-200 px-1 py-0.5"
            />
          </label>
        </div>
        <p className="text-[9px] text-gray-400">
          {stats.count} CCs · area {Math.round(stats.min)}–{Math.round(stats.max)} px²
        </p>
        <div className="flex gap-1">
          <select
            className="rounded border border-gray-200 text-[10px] px-1"
            value={sizeMode}
            onChange={(e) => setSizeMode(e.target.value as SizeFilterMode)}
          >
            <option value="keep">Keep matching</option>
            <option value="delete">Delete matching</option>
          </select>
          <button
            type="button"
            className={cn(btn, 'flex-1')}
            disabled={!workingOk || !sourceKey}
            onClick={() => {
              const ok = studio.applySizeFilter(target, {
                classId,
                minArea,
                maxArea,
                mode: sizeMode,
              });
              setStatus(ok ? `Size filter (${sizeMode}) applied` : 'Nothing to filter');
            }}
          >
            Apply size
          </button>
        </div>
      </div>

      {/* ROI */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium text-gray-600">By region (ROI)</span>
        <p className="text-[9px] text-gray-400 leading-snug">
          Draw or select a polygon / rect / ellipse, capture it as ROI, then apply.
        </p>
        <button type="button" className={btn} onClick={captureRoiFromSelection}>
          Capture selection as ROI
        </button>
        {roiShape && (
          <p className="text-[9px] text-sky-700">ROI: {roiShape.kind} ready</p>
        )}
        <select
          className="rounded border border-gray-200 text-[10px] px-1 py-0.5"
          value={roiMode}
          onChange={(e) => setRoiMode(e.target.value as RoiMode)}
        >
          <option value="keep_inside">Keep inside</option>
          <option value="delete_inside">Delete inside</option>
          <option value="keep_outside">Keep outside</option>
          <option value="delete_outside">Delete outside</option>
        </select>
        <label className="flex items-center gap-1 text-[10px] text-gray-600">
          <input
            type="checkbox"
            checked={allClassesRoi}
            onChange={(e) => setAllClassesRoi(e.target.checked)}
          />
          Affect all classes
        </label>
        <button
          type="button"
          className={btn}
          disabled={!roiShape || !workingOk}
          onClick={() => {
            if (!roiShape) return;
            const ok = studio.applyRoi(target, {
              roi: roiShape,
              mode: roiMode,
              classId: resolveClassForRoi(),
            });
            setStatus(ok ? `ROI ${roiMode} applied` : 'ROI apply failed');
          }}
        >
          Apply ROI
        </button>
      </div>

      {/* Morphology */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium text-gray-600">Morphology</span>
        <div className="grid grid-cols-2 gap-1">
          <select
            className="rounded border border-gray-200 text-[10px] px-1 py-0.5"
            value={morphOp}
            onChange={(e) => setMorphOp(e.target.value as MorphCleanupOp)}
          >
            <option value="fill">Fill holes</option>
            <option value="islands">Remove islands</option>
            <option value="smooth">Smooth</option>
            <option value="grow">Grow</option>
            <option value="shrink">Shrink</option>
          </select>
          <input
            type="number"
            min={1}
            value={morphParam}
            title="Island min area (px²) or grow/shrink/smooth radius"
            onChange={(e) => setMorphParam(Math.max(1, Number(e.target.value) || 1))}
            className="rounded border border-gray-200 px-1 py-0.5 text-[10px]"
          />
        </div>
        <button
          type="button"
          className={btn}
          disabled={!workingOk}
          onClick={() => {
            const ok = studio.applyMorph(target, {
              op: morphOp,
              param: morphParam,
              classId: activeClassId ?? classId,
            });
            setStatus(ok ? `Morph ${morphOp} applied` : 'Morph failed');
          }}
        >
          Apply morph
        </button>
      </div>

      {status && <p className="text-[10px] text-gray-500 leading-snug">{status}</p>}
    </div>
  );
}
