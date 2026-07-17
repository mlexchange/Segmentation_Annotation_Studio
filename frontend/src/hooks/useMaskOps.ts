/**
 * Mask operations that work on the ACTIVE class of the current sample by
 * round-tripping its shapes through a binary mask: cleanup (morphology) on the
 * current slice, and cross-slice copy across a volume.
 *
 * Everything reads global store state at call time (fresh, no stale closures)
 * and writes back via one-undo-step store actions.
 */
import { useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAnnotationStore, type Shape, type PolygonShape } from '@/stores/annotationStore';
import { useDatasetStore } from '@/stores/datasetStore';
import { rasterizeShapes, gridFor } from '@/lib/rasterize';
import { maskToPolygons } from '@/lib/magicwand';
import { fillHoles, removeSmallComponents, dilate, erode, smooth } from '@/lib/morphology';

export type CleanupOp = 'fill' | 'islands' | 'smooth' | 'grow' | 'shrink';

/** Polygons (image coords) → polygon shapes for a class. */
function polysToShapes(polys: number[][], classId: number): PolygonShape[] {
  return polys
    .filter((pts) => pts.length >= 6)
    .map((pts) => ({ id: uuidv4(), classId, kind: 'polygon', points: pts }));
}

export function useMaskOps(sourceKey: string | null, activeClassId: number | null) {
  const replaceClassShapesOnSlice = useAnnotationStore((s) => s.replaceClassShapesOnSlice);
  const copySliceShapes = useAnnotationStore((s) => s.copySliceShapes);

  /** Shapes of the active class on a given slice (fresh from the store). */
  const classShapesOn = useCallback(
    (slice: number): Shape[] => {
      if (!sourceKey || activeClassId === null) return [];
      const { byImage } = useAnnotationStore.getState();
      return (byImage[sourceKey]?.[String(slice)] ?? []).filter((sh) => sh.classId === activeClassId);
    },
    [sourceKey, activeClassId],
  );

  /** Apply a morphological cleanup op to the active class on the current slice. */
  const applyCleanup = useCallback(
    (op: CleanupOp, param = 2): boolean => {
      const meta = useDatasetStore.getState().meta;
      if (!sourceKey || activeClassId === null || !meta) return false;
      const slice = useDatasetStore.getState().currentSlice;
      const shapes = classShapesOn(slice);
      if (shapes.length === 0) return false;

      const { gw, gh, scale } = gridFor(meta.width, meta.height);
      let mask = rasterizeShapes(shapes, gw, gh, scale);
      switch (op) {
        case 'fill': mask = fillHoles(mask, gw, gh); break;
        case 'islands': mask = removeSmallComponents(mask, gw, gh, Math.max(1, param)); break;
        case 'smooth': mask = smooth(mask, gw, gh, Math.max(1, param)); break;
        case 'grow': mask = dilate(mask, gw, gh, Math.max(1, param)); break;
        case 'shrink': mask = erode(mask, gw, gh, Math.max(1, param)); break;
      }
      const polys = maskToPolygons(mask, gw, gh, { minRegion: 4, scale });
      replaceClassShapesOnSlice(sourceKey, slice, activeClassId, polysToShapes(polys, activeClassId));
      return true;
    },
    [sourceKey, activeClassId, classShapesOn, replaceClassShapesOnSlice],
  );

  /** Copy the active class's shapes from the current slice to the next slice. */
  const copyToNext = useCallback((): boolean => {
    const { meta, currentSlice } = useDatasetStore.getState();
    if (!sourceKey || activeClassId === null || !meta) return false;
    if (currentSlice >= meta.nSlices - 1) return false;
    if (classShapesOn(currentSlice).length === 0) return false;
    copySliceShapes(sourceKey, currentSlice, [currentSlice + 1], activeClassId);
    return true;
  }, [sourceKey, activeClassId, classShapesOn, copySliceShapes]);

  return { applyCleanup, copyToNext };
}
