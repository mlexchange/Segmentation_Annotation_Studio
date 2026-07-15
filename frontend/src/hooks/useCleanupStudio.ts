/**
 * Cleanup Studio — size / ROI / morph ops on dense label maps (pixel-accurate).
 */
import { useCallback, useMemo } from 'react';
import { useAnnotationStore, type Shape } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { useMaskSetStore } from '@/stores/maskSetStore';
import {
  applyMorphToLabelMap,
  applyRoiToLabelMap,
  componentSizeStatsLabelMap,
  filterLabelMapBySize,
  type MorphCleanupOp,
  type RoiMode,
  type SizeFilterMode,
} from '@/lib/cleanupOps';
import { mergeMaskSetIntoAnnotations, type MergePolicy, labelMapPixelCount } from '@/lib/maskSetMerge';
import { shapesToLabelMap } from '@/lib/labelMap';

export type CleanupTarget = 'annotations' | 'mask_set';

export function useCleanupStudio(sourceKey: string | null) {
  const { meta, currentSlice } = useDatasetStore();
  const byImage = useAnnotationStore((s) => s.byImage);
  const setShapes = useAnnotationStore((s) => s.setShapes);
  const {
    sets,
    activeSetId,
    updateSetLabelMap,
    addSet,
    deleteSet,
    renameSet,
    duplicateSet,
    setActiveSetId,
    setVisible,
  } = useMaskSetStore();

  const sliceShapes = useMemo(() => {
    if (!sourceKey) return [] as Shape[];
    return byImage[sourceKey]?.[String(currentSlice)] ?? [];
  }, [sourceKey, byImage, currentSlice]);

  const sliceSets = useMemo(() => {
    if (!sourceKey) return [];
    return sets.filter((m) => m.sourceKey === sourceKey && m.slice === currentSlice);
  }, [sets, sourceKey, currentSlice]);

  const activeSet = sliceSets.find((m) => m.id === activeSetId) ?? null;

  /** Resolve a dense label map for the working target (full image resolution). */
  const readLabelMap = useCallback(
    (target: CleanupTarget): { labels: Uint8Array; width: number; height: number } | null => {
      if (!meta) return null;
      const { width, height } = meta;
      if (target === 'mask_set') {
        if (!activeSet) return null;
        if (activeSet.labelMap && activeSet.width === width && activeSet.height === height) {
          return { labels: activeSet.labelMap, width, height };
        }
        // Legacy set with only shapes — rasterize once at full res
        if (activeSet.shapes.length) {
          return {
            labels: shapesToLabelMap(activeSet.shapes, width, height),
            width,
            height,
          };
        }
        return null;
      }
      if (!sliceShapes.length) return null;
      return { labels: shapesToLabelMap(sliceShapes, width, height), width, height };
    },
    [meta, activeSet, sliceShapes],
  );

  const writeLabelMap = useCallback(
    (target: CleanupTarget, labels: Uint8Array, width: number, height: number) => {
      if (!sourceKey) return false;
      if (target === 'mask_set') {
        if (!activeSet) return false;
        updateSetLabelMap(activeSet.id, labels, width, height);
        return true;
      }
      // Annotations target: promote edits into a new/updated working mask set
      // so we never force polygon round-trips mid-cleanup. Also write polygons
      // would lose fidelity — create a set instead.
      addSet({
        name: `annotations edit slice ${currentSlice + 1}`,
        sourceKey,
        slice: currentSlice,
        origin: 'manual',
        labelMap: labels,
        width,
        height,
        shapes: [],
      });
      return true;
    },
    [sourceKey, activeSet, updateSetLabelMap, addSet, currentSlice],
  );

  const sizeStats = useCallback(
    (target: CleanupTarget, classId: number | null) => {
      const buf = readLabelMap(target);
      if (!buf) return { areas: [], min: 0, max: 0, count: 0 };
      return componentSizeStatsLabelMap(buf.labels, buf.width, buf.height, classId);
    },
    [readLabelMap],
  );

  const applySizeFilter = useCallback(
    (
      target: CleanupTarget,
      opts: {
        classId: number | null;
        minArea: number;
        maxArea: number;
        mode: SizeFilterMode;
      },
    ) => {
      const buf = readLabelMap(target);
      if (!buf) return false;
      const next = filterLabelMapBySize(buf.labels, buf.width, buf.height, opts);
      return writeLabelMap(target, next, buf.width, buf.height);
    },
    [readLabelMap, writeLabelMap],
  );

  const applyRoi = useCallback(
    (
      target: CleanupTarget,
      opts: { roi: Shape; mode: RoiMode; classId: number | null },
    ) => {
      const buf = readLabelMap(target);
      if (!buf) return false;
      const next = applyRoiToLabelMap(buf.labels, buf.width, buf.height, opts);
      return writeLabelMap(target, next, buf.width, buf.height);
    },
    [readLabelMap, writeLabelMap],
  );

  const applyMorph = useCallback(
    (
      target: CleanupTarget,
      opts: { op: MorphCleanupOp; param: number; classId: number | null },
    ) => {
      const buf = readLabelMap(target);
      if (!buf) return false;
      const next = applyMorphToLabelMap(buf.labels, buf.width, buf.height, opts);
      return writeLabelMap(target, next, buf.width, buf.height);
    },
    [readLabelMap, writeLabelMap],
  );

  const createSetFromLabelMap = useCallback(
    (
      name: string,
      labelMap: Uint8Array,
      width: number,
      height: number,
      origin: 'clf' | 'manual' | 'merge' | 'import' | 'shelf',
    ) => {
      if (!sourceKey) return null;
      return addSet({
        name,
        sourceKey,
        slice: currentSlice,
        origin,
        labelMap,
        width,
        height,
        shapes: [],
      });
    },
    [sourceKey, currentSlice, addSet],
  );

  const copyAnnotationsToSet = useCallback(
    (name?: string) => {
      if (!sourceKey || !meta || !sliceShapes.length) return null;
      const labels = shapesToLabelMap(sliceShapes, meta.width, meta.height);
      return createSetFromLabelMap(
        name ?? `annotations slice ${currentSlice + 1}`,
        labels,
        meta.width,
        meta.height,
        'manual',
      );
    },
    [sourceKey, meta, sliceShapes, currentSlice, createSetFromLabelMap],
  );

  const mergeActiveIntoAnnotations = useCallback(
    (policy: MergePolicy) => {
      if (!sourceKey || !meta || !activeSet) return false;
      const next = mergeMaskSetIntoAnnotations(
        sliceShapes,
        activeSet.shapes,
        meta.width,
        meta.height,
        policy,
        activeSet.labelMap,
      );
      setShapes(sourceKey, currentSlice, next);
      return true;
    },
    [sourceKey, meta, activeSet, sliceShapes, setShapes, currentSlice],
  );

  return {
    sliceShapes,
    sliceSets,
    activeSet,
    activeSetId,
    setActiveSetId,
    setVisible,
    renameSet,
    deleteSet,
    duplicateSet,
    sizeStats,
    applySizeFilter,
    applyRoi,
    applyMorph,
    createSetFromLabelMap,
    copyAnnotationsToSet,
    mergeActiveIntoAnnotations,
    labelMapPixelCount,
    meta,
  };
}
