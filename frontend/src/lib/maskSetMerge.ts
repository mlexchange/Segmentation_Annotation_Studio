/**
 * Merge mask-set label maps / shapes into slice annotations.
 */
import { v4 as uuidv4 } from 'uuid';
import type { Shape } from '@/stores/annotationStore';
import { gridFor, rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { classIdsInLabelMap, labelMapToShapes, shapesToLabelMap } from '@/lib/labelMap';

export type MergePolicy = 'union' | 'replace_class' | 'delete_where_set';

function shapesFromMask(
  mask: Uint8Array,
  gw: number,
  gh: number,
  scale: number,
  classId: number,
): Shape[] {
  return maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale })
    .filter((p) => p.points.length >= 6)
    .map((p) => ({
      id: uuidv4(),
      classId,
      kind: 'polygon' as const,
      points: p.points,
      ...(p.holes.length ? { holes: p.holes } : {}),
    }));
}

/**
 * Merge set into annotations under *policy*.
 * Prefer dense labelMap when present (pixel-accurate).
 */
export function mergeMaskSetIntoAnnotations(
  annotations: Shape[],
  setShapes: Shape[],
  width: number,
  height: number,
  policy: MergePolicy,
  setLabelMap?: Uint8Array | null,
): Shape[] {
  const setMap =
    setLabelMap && setLabelMap.length === width * height
      ? setLabelMap
      : setShapes.length
        ? shapesToLabelMap(setShapes, width, height)
        : null;

  if (setMap) {
    const setClasses = classIdsInLabelMap(setMap);
    if (!setClasses.length) return annotations;

    if (policy === 'replace_class') {
      const kept = annotations.filter((s) => !setClasses.includes(s.classId));
      return [...kept, ...labelMapToShapes(setMap, width, height)];
    }

    const annMap = shapesToLabelMap(annotations, width, height);
    const out = new Uint8Array(annMap);
    if (policy === 'union') {
      for (let i = 0; i < out.length; i++) {
        if (setMap[i]) out[i] = setMap[i];
      }
    } else {
      // delete_where_set: clear annotation pixels where set has that class
      for (let i = 0; i < out.length; i++) {
        const s = setMap[i];
        if (s && out[i] === s) out[i] = 0;
      }
    }
    // Preserve annotation classes not in set for union/delete semantics already handled
    return labelMapToShapes(out, width, height);
  }

  // Legacy shape-only path
  const setClasses = [...new Set(setShapes.map((s) => s.classId))];
  if (!setClasses.length) return annotations;

  if (policy === 'replace_class') {
    const kept = annotations.filter((s) => !setClasses.includes(s.classId));
    return [
      ...kept,
      ...setShapes.map((s) => ({ ...s, id: uuidv4() })),
    ];
  }

  const { gw, gh, scale } = gridFor(width, height);
  const out: Shape[] = annotations.filter((s) => !setClasses.includes(s.classId));
  for (const cid of setClasses) {
    const annGroup = annotations.filter((s) => s.classId === cid);
    const setGroup = setShapes.filter((s) => s.classId === cid);
    const annMask = annGroup.length
      ? rasterizeShapes(annGroup, gw, gh, scale)
      : new Uint8Array(gw * gh);
    const setMask = rasterizeShapes(setGroup, gw, gh, scale);
    const merged = new Uint8Array(gw * gh);
    if (policy === 'union') {
      for (let i = 0; i < merged.length; i++) merged[i] = annMask[i] || setMask[i] ? 1 : 0;
    } else {
      for (let i = 0; i < merged.length; i++) merged[i] = annMask[i] && !setMask[i] ? 1 : 0;
    }
    out.push(...shapesFromMask(merged, gw, gh, scale, cid));
  }
  return out;
}

/** Count non-zero pixels (for UI). */
export function labelMapPixelCount(labels: Uint8Array | null): number {
  if (!labels) return 0;
  let n = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i]) n++;
  return n;
}
