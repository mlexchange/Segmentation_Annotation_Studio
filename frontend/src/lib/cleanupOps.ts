/**
 * Cleanup Studio ops: connected-component size filters and ROI keep/delete.
 * Prefer *LabelMap* variants (full-res classId pixels). Shape variants remain
 * for legacy tests; Cleanup Studio uses label maps so we never round-trip
 * polygons mid-edit.
 */
import { v4 as uuidv4 } from 'uuid';
import type { PolygonShape, Shape } from '@/stores/annotationStore';
import { gridFor, rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import {
  dilate,
  erode,
  fillHoles,
  removeSmallComponents,
  smooth,
} from '@/lib/morphology';
import {
  classBinary,
  classIdsInLabelMap,
  paintClass,
} from '@/lib/labelMap';

export type SizeFilterMode = 'keep' | 'delete';
export type RoiMode = 'keep_inside' | 'delete_inside' | 'keep_outside' | 'delete_outside';
export type MorphCleanupOp = 'fill' | 'islands' | 'smooth' | 'grow' | 'shrink';

export interface LabelComponentsResult {
  labels: Int32Array;
  sizesByLabel: Record<number, number>;
  count: number;
}

/** 4-connected CC labels (1..N) and per-label pixel counts. */
export function labelComponents(
  mask: Uint8Array,
  gw: number,
  gh: number,
): LabelComponentsResult {
  const labels = new Int32Array(gw * gh);
  const sizesByLabel: Record<number, number> = {};
  let next = 1;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p] || labels[p]) continue;
    const lbl = next++;
    let size = 0;
    const stack = [p];
    labels[p] = lbl;
    while (stack.length) {
      const idx = stack.pop()!;
      size++;
      const x = idx % gw;
      const y = (idx / gw) | 0;
      const push = (j: number) => {
        if (mask[j] && !labels[j]) {
          labels[j] = lbl;
          stack.push(j);
        }
      };
      if (x > 0) push(idx - 1);
      if (x < gw - 1) push(idx + 1);
      if (y > 0) push(idx - gw);
      if (y < gh - 1) push(idx + gw);
    }
    sizesByLabel[lbl] = size;
  }
  return { labels, sizesByLabel, count: next - 1 };
}

/**
 * Keep or delete CCs whose area is in [minArea, maxArea] (grid pixels).
 * mode=keep: only in-range survive; mode=delete: in-range are removed.
 */
export function filterBySize(
  mask: Uint8Array,
  gw: number,
  gh: number,
  {
    minArea,
    maxArea,
    mode,
  }: { minArea: number; maxArea: number; mode: SizeFilterMode },
): Uint8Array {
  const { labels, sizesByLabel } = labelComponents(mask, gw, gh);
  const keepLabel = new Set<number>();
  for (const [lblStr, size] of Object.entries(sizesByLabel)) {
    const lbl = Number(lblStr);
    const inRange = size >= minArea && size <= maxArea;
    if (mode === 'keep' ? inRange : !inRange) keepLabel.add(lbl);
  }
  const out = new Uint8Array(gw * gh);
  for (let i = 0; i < out.length; i++) {
    if (labels[i] && keepLabel.has(labels[i])) out[i] = 1;
  }
  return out;
}

function maskToShapes(mask: Uint8Array, gw: number, gh: number, scale: number, classId: number): PolygonShape[] {
  const polys = maskToPolygonsWithHoles(mask, gw, gh, { minRegion: 4, scale });
  return polys
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
 * Filter CCs by image-pixel area for one class (or all if classId null).
 * Other classes are preserved unchanged.
 */
export function filterShapesByComponentSize(
  shapes: Shape[],
  width: number,
  height: number,
  {
    classId,
    minArea,
    maxArea,
    mode,
  }: {
    classId: number | null;
    minArea: number;
    maxArea: number;
    mode: SizeFilterMode;
  },
): Shape[] {
  const { gw, gh, scale } = gridFor(width, height);
  const minG = Math.max(1, Math.round(minArea / (scale * scale)));
  const maxG =
    maxArea === Infinity
      ? Infinity
      : Math.max(minG, Math.round(maxArea / (scale * scale)));

  const classIds = new Set(
    shapes.map((s) => s.classId).filter((id) => classId === null || id === classId),
  );
  const kept: Shape[] = shapes.filter((s) => classId !== null && s.classId !== classId);
  for (const cid of classIds) {
    const group = shapes.filter((s) => s.classId === cid);
    if (!group.length) continue;
    let mask = rasterizeShapes(group, gw, gh, scale);
    mask = filterBySize(mask, gw, gh, { minArea: minG, maxArea: maxG, mode });
    kept.push(...maskToShapes(mask, gw, gh, scale, cid));
  }
  return kept;
}

/** Apply ROI gate to a binary mask. */
export function applyRoiMask(
  mask: Uint8Array,
  roi: Uint8Array,
  mode: RoiMode,
): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const inRoi = !!roi[i];
    const on = !!mask[i];
    switch (mode) {
      case 'keep_inside':
        out[i] = on && inRoi ? 1 : 0;
        break;
      case 'delete_inside':
        out[i] = on && !inRoi ? 1 : 0;
        break;
      case 'keep_outside':
        out[i] = on && !inRoi ? 1 : 0;
        break;
      case 'delete_outside':
        out[i] = on && inRoi ? 1 : 0;
        break;
    }
  }
  return out;
}

/**
 * Keep/delete mask pixels relative to an ROI shape.
 * classId null = all classes.
 */
export function applyRoiToShapes(
  shapes: Shape[],
  width: number,
  height: number,
  {
    roi,
    mode,
    classId,
  }: { roi: Shape; mode: RoiMode; classId: number | null },
): Shape[] {
  const { gw, gh, scale } = gridFor(width, height);
  const roiMask = rasterizeShapes([{ ...roi, classId: roi.classId || 1 }], gw, gh, scale);
  const classIds = new Set(
    shapes.map((s) => s.classId).filter((id) => classId === null || id === classId),
  );
  const kept: Shape[] = shapes.filter((s) => classId !== null && s.classId !== classId);
  for (const cid of classIds) {
    const group = shapes.filter((s) => s.classId === cid);
    if (!group.length) continue;
    let mask = rasterizeShapes(group, gw, gh, scale);
    mask = applyRoiMask(mask, roiMask, mode);
    kept.push(...maskToShapes(mask, gw, gh, scale, cid));
  }
  return kept;
}

/** Morphology on one/all classes; returns full replacement shape list. */
export function applyMorphToShapes(
  shapes: Shape[],
  width: number,
  height: number,
  {
    op,
    param,
    classId,
  }: { op: MorphCleanupOp; param: number; classId: number | null },
): Shape[] {
  const { gw, gh, scale } = gridFor(width, height);
  const classIds = new Set(
    shapes.map((s) => s.classId).filter((id) => classId === null || id === classId),
  );
  const kept: Shape[] = shapes.filter((s) => classId !== null && s.classId !== classId);
  for (const cid of classIds) {
    const group = shapes.filter((s) => s.classId === cid);
    if (!group.length) continue;
    let mask = rasterizeShapes(group, gw, gh, scale);
    switch (op) {
      case 'fill':
        mask = fillHoles(mask, gw, gh);
        break;
      case 'islands':
        mask = removeSmallComponents(
          mask,
          gw,
          gh,
          Math.max(1, Math.round(param / (scale * scale))),
        );
        break;
      case 'smooth':
        mask = smooth(mask, gw, gh, Math.max(1, Math.round(param)));
        break;
      case 'grow':
        mask = dilate(mask, gw, gh, Math.max(1, Math.round(param / scale)));
        break;
      case 'shrink':
        mask = erode(mask, gw, gh, Math.max(1, Math.round(param / scale)));
        break;
    }
    kept.push(...maskToShapes(mask, gw, gh, scale, cid));
  }
  return kept;
}

/** Size histogram (grid px buckets) for CC areas of a class mask. */
export function componentSizeStats(
  shapes: Shape[],
  width: number,
  height: number,
  classId: number | null,
): { areas: number[]; min: number; max: number; count: number } {
  const { gw, gh, scale } = gridFor(width, height);
  const group =
    classId === null
      ? shapes
      : shapes.filter((s) => s.classId === classId);
  if (!group.length) return { areas: [], min: 0, max: 0, count: 0 };
  const mask = rasterizeShapes(group, gw, gh, scale);
  const { sizesByLabel } = labelComponents(mask, gw, gh);
  const areas = Object.values(sizesByLabel)
    .map((a) => a * scale * scale)
    .sort((a, b) => a - b);
  return {
    areas,
    min: areas[0] ?? 0,
    max: areas[areas.length - 1] ?? 0,
    count: areas.length,
  };
}

/* ---------- Full-resolution classId label-map ops (Cleanup Studio) ---------- */

function targetClassIds(labels: Uint8Array, classId: number | null): number[] {
  const all = classIdsInLabelMap(labels);
  if (classId === null) return all;
  return all.includes(classId) ? [classId] : [];
}

/** Filter CCs on a dense classId map (image pixels, no downsample). */
export function filterLabelMapBySize(
  labels: Uint8Array,
  width: number,
  height: number,
  {
    classId,
    minArea,
    maxArea,
    mode,
  }: {
    classId: number | null;
    minArea: number;
    maxArea: number;
    mode: SizeFilterMode;
  },
): Uint8Array {
  let out = new Uint8Array(labels);
  const minA = Math.max(1, Math.round(minArea));
  const maxA = maxArea === Infinity ? Infinity : Math.max(minA, Math.round(maxArea));
  for (const cid of targetClassIds(labels, classId)) {
    let mask = classBinary(out, cid);
    mask = filterBySize(mask, width, height, { minArea: minA, maxArea: maxA, mode });
    out = paintClass(out, mask, cid, { protectOthers: true });
  }
  return out;
}

/** ROI keep/delete on a dense classId map. */
export function applyRoiToLabelMap(
  labels: Uint8Array,
  width: number,
  height: number,
  {
    roi,
    mode,
    classId,
  }: { roi: Shape; mode: RoiMode; classId: number | null },
): Uint8Array {
  const roiMask = rasterizeShapes([{ ...roi, classId: 1 }], width, height, 1);
  let out = new Uint8Array(labels);
  for (const cid of targetClassIds(labels, classId)) {
    let mask = classBinary(out, cid);
    mask = applyRoiMask(mask, roiMask, mode);
    out = paintClass(out, mask, cid, { protectOthers: true });
  }
  return out;
}

/** Morphology on a dense classId map (full resolution). */
export function applyMorphToLabelMap(
  labels: Uint8Array,
  width: number,
  height: number,
  {
    op,
    param,
    classId,
  }: { op: MorphCleanupOp; param: number; classId: number | null },
): Uint8Array {
  let out = new Uint8Array(labels);
  for (const cid of targetClassIds(labels, classId)) {
    let mask = classBinary(out, cid);
    switch (op) {
      case 'fill':
        mask = fillHoles(mask, width, height);
        break;
      case 'islands':
        mask = removeSmallComponents(mask, width, height, Math.max(1, Math.round(param)));
        break;
      case 'smooth':
        mask = smooth(mask, width, height, Math.max(1, Math.round(param)));
        break;
      case 'grow':
        mask = dilate(mask, width, height, Math.max(1, Math.round(param)));
        break;
      case 'shrink':
        mask = erode(mask, width, height, Math.max(1, Math.round(param)));
        break;
    }
    out = paintClass(out, mask, cid, { protectOthers: true });
  }
  return out;
}

/** CC size stats from a dense classId map. */
export function componentSizeStatsLabelMap(
  labels: Uint8Array,
  width: number,
  height: number,
  classId: number | null,
): { areas: number[]; min: number; max: number; count: number } {
  const areas: number[] = [];
  for (const cid of targetClassIds(labels, classId)) {
    const mask = classBinary(labels, cid);
    const { sizesByLabel } = labelComponents(mask, width, height);
    for (const a of Object.values(sizesByLabel)) areas.push(a);
  }
  areas.sort((a, b) => a - b);
  return {
    areas,
    min: areas[0] ?? 0,
    max: areas[areas.length - 1] ?? 0,
    count: areas.length,
  };
}
