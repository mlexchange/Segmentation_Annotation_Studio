/**
 * Full-resolution classId label maps for Cleanup Studio (pixel-accurate buffer).
 */
import { v4 as uuidv4 } from 'uuid';
import type { Shape } from '@/stores/annotationStore';
import { rasterizeShapes } from '@/lib/rasterize';
import { maskToPolygonsWithHoles } from '@/lib/magicwand';
import { colorizeLabelMap } from '@/lib/pixelClf';

/** Rasterize shapes into an HxW classId map (scale=1, last class wins on overlap). */
export function shapesToLabelMap(
  shapes: Shape[],
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(width * height);
  const byClass = new Map<number, Shape[]>();
  for (const s of shapes) {
    const arr = byClass.get(s.classId);
    if (arr) arr.push(s);
    else byClass.set(s.classId, [s]);
  }
  for (const [cid, group] of byClass) {
    const mask = rasterizeShapes(group, width, height, 1);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) out[i] = cid;
    }
  }
  return out;
}

/** Class ids present in a label map (non-zero). */
export function classIdsInLabelMap(labels: Uint8Array): number[] {
  const seen = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (v) seen.add(v);
  }
  return [...seen].sort((a, b) => a - b);
}

/** Extract binary mask for one classId. */
export function classBinary(labels: Uint8Array, classId: number): Uint8Array {
  const out = new Uint8Array(labels.length);
  for (let i = 0; i < labels.length; i++) if (labels[i] === classId) out[i] = 1;
  return out;
}

/** Write a binary mask back as classId (optionally without overwriting other classes). */
export function paintClass(
  labels: Uint8Array,
  mask: Uint8Array,
  classId: number,
  { protectOthers = true }: { protectOthers?: boolean } = {},
): Uint8Array {
  const out = new Uint8Array(labels);
  for (let i = 0; i < out.length; i++) {
    if (out[i] === classId) out[i] = 0;
  }
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    if (protectOthers && out[i] !== 0 && out[i] !== classId) continue;
    out[i] = classId;
  }
  return out;
}

/** Vectorize a label map to polygon shapes (only at Merge). */
export function labelMapToShapes(
  labels: Uint8Array,
  width: number,
  height: number,
  { minRegion = 4 }: { minRegion?: number } = {},
): Shape[] {
  const shapes: Shape[] = [];
  for (const cid of classIdsInLabelMap(labels)) {
    const mask = classBinary(labels, cid);
    const polys = maskToPolygonsWithHoles(mask, width, height, {
      minRegion,
      scale: 1,
    });
    for (const p of polys) {
      if (p.points.length < 6) continue;
      shapes.push({
        id: uuidv4(),
        classId: cid,
        kind: 'polygon',
        points: p.points,
        ...(p.holes.length ? { holes: p.holes } : {}),
      });
    }
  }
  return shapes;
}

/** Colorize label map for canvas overlay. */
export function labelMapOverlayCanvas(
  labels: Uint8Array,
  width: number,
  height: number,
  colorByClass: Map<number, string>,
  alpha = 110,
): HTMLCanvasElement {
  return colorizeLabelMap(labels, width, height, colorByClass, alpha);
}

/** base64 encode for draft JSON. */
export function labelMapToBase64(labels: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < labels.length; i += chunk) {
    binary += String.fromCharCode(...labels.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Decode draft base64 label map. */
export function labelMapFromBase64(b64: string, expectLen?: number): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  if (expectLen !== undefined && out.length !== expectLen) {
    throw new Error(`label map length ${out.length} != expected ${expectLen}`);
  }
  return out;
}
