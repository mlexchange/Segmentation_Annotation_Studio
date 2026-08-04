/**
 * sliceAnnotationSummary — group one slice's saved annotations by class, for the
 * Browse "Annotations" column.
 *
 * A slice of a volume can hold annotations under TWO independent source keys
 * (see `isAnnotatedPath` in sourceKey.ts): the volume's key, where the slice's
 * shapes live under its slice index, and — if that slice was ever opened as a
 * standalone array — its own key, where they live under slice `"0"`. Callers
 * fetch both payloads and pass them here to be merged into one view.
 */
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

/** The `payload` object of a draft or saved-version document. */
export interface AnnotationPayload {
  classes?: AnnotationClass[];
  slices?: Record<string, Shape[]>;
  split_by_slice?: Record<string, string>;
  negative_slices?: string[];
}

/** One payload plus the slice key to read out of it. */
export interface PayloadSource {
  payload: AnnotationPayload | null;
  /** Slice key within this payload — the volume index, or `"0"` for a standalone array. */
  sliceKey: string;
}

export interface ShapeSummary {
  id: string;
  kind: Shape['kind'];
}

export interface ClassSummary {
  classId: number;
  label: string;
  color: string;
  shapes: ShapeSummary[];
}

export interface SliceAnnotationSummary {
  /** Classes that actually have shapes on this slice, in the payloads' class order. */
  classes: ClassSummary[];
  totalShapes: number;
  /** True if any payload marks this slice as a negative (deliberately empty) example. */
  isNegative: boolean;
  /** Explicit train/valid/test assignment, or null when left on auto. */
  split: string | null;
}

const EMPTY: SliceAnnotationSummary = { classes: [], totalShapes: 0, isNegative: false, split: null };

/**
 * Merge the given payloads into a per-class summary of one slice's annotations.
 *
 * Shapes are de-duplicated by id, so a slice annotated under both its volume key
 * and its own standalone key doesn't double-count. A shape whose `classId` has no
 * matching class entry is still reported, under a synthesized placeholder label —
 * dropping it would under-report what's actually on the slice.
 */
export function summarizeSliceAnnotations(sources: PayloadSource[]): SliceAnnotationSummary {
  const present = sources.filter((s): s is PayloadSource & { payload: AnnotationPayload } => s.payload != null);
  if (present.length === 0) return EMPTY;

  // First payload to define a class wins its label/color (drafts are the fresher
  // source and are passed first).
  const classById = new Map<number, AnnotationClass>();
  for (const { payload } of present) {
    for (const cls of payload.classes ?? []) {
      if (!classById.has(cls.classId)) classById.set(cls.classId, cls);
    }
  }

  const seenShapeIds = new Set<string>();
  const shapesByClass = new Map<number, ShapeSummary[]>();
  let isNegative = false;
  let split: string | null = null;

  for (const { payload, sliceKey } of present) {
    for (const shape of payload.slices?.[sliceKey] ?? []) {
      if (!shape || seenShapeIds.has(shape.id)) continue;
      seenShapeIds.add(shape.id);
      const bucket = shapesByClass.get(shape.classId);
      if (bucket) bucket.push({ id: shape.id, kind: shape.kind });
      else shapesByClass.set(shape.classId, [{ id: shape.id, kind: shape.kind }]);
    }
    if ((payload.negative_slices ?? []).includes(sliceKey)) isNegative = true;
    const assigned = payload.split_by_slice?.[sliceKey];
    if (split === null && assigned && assigned !== 'auto') split = assigned;
  }

  // Ordered by the class list so the column matches the Annotate sidebar; classes
  // with no shapes on this slice are omitted, unknown ones appended.
  const classes: ClassSummary[] = [];
  for (const [classId, cls] of classById) {
    const shapes = shapesByClass.get(classId);
    if (shapes?.length) classes.push({ classId, label: cls.label, color: cls.color, shapes });
  }
  for (const [classId, shapes] of shapesByClass) {
    if (!classById.has(classId) && shapes.length) {
      classes.push({ classId, label: `Class ${classId}`, color: '#94a3b8', shapes });
    }
  }

  return { classes, totalShapes: seenShapeIds.size, isNegative, split };
}
