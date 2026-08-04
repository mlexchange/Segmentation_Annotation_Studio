/**
 * importPredictions — remap a saved run's predicted shapes onto the
 * currently-open sample's class list before importing them as annotations.
 *
 * A run's classes are a *snapshot* from whenever it was trained — the
 * currently open sample's class list may have since been renamed, reordered,
 * or extended. Policy: match by case-insensitive label; a run class with no
 * matching label is appended to the current class list (keeping the run's
 * color); every predicted shape's classId is remapped to the resolved id.
 * Deterministic and non-destructive — existing classes/shapes are untouched.
 */
import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

export interface RunClass {
  classId: number;
  label: string;
  color: string;
  isVisible?: boolean;
}

export interface RemappedPredictions {
  /** The (possibly extended) class list to apply via classStore.setClasses. */
  classes: AnnotationClass[];
  /** Predicted shapes with classId remapped to match `classes`. */
  slices: Record<string, Shape[]>;
}

/** Remap a run's predicted shapes onto `currentClasses`, appending any of the
 *  run's classes that have no case-insensitive label match. */
export function remapPredictedShapes(
  runClasses: RunClass[],
  currentClasses: AnnotationClass[],
  predictedSlices: Record<string, Shape[]>,
): RemappedPredictions {
  const byLabel = new Map(currentClasses.map((c) => [c.label.trim().toLowerCase(), c.classId]));
  const nextClasses = [...currentClasses];
  let nextId = nextClasses.length ? Math.max(...nextClasses.map((c) => c.classId)) + 1 : 1;

  const runIdToResolvedId = new Map<number, number>();
  for (const runClass of runClasses) {
    const key = runClass.label.trim().toLowerCase();
    const existingId = byLabel.get(key);
    if (existingId !== undefined) {
      runIdToResolvedId.set(runClass.classId, existingId);
      continue;
    }
    const newId = nextId++;
    nextClasses.push({ classId: newId, label: runClass.label, color: runClass.color, isVisible: true });
    byLabel.set(key, newId);
    runIdToResolvedId.set(runClass.classId, newId);
  }

  const slices: Record<string, Shape[]> = {};
  for (const [sliceKey, shapes] of Object.entries(predictedSlices)) {
    slices[sliceKey] = shapes.map((shape) => ({
      ...shape,
      classId: runIdToResolvedId.get(shape.classId) ?? shape.classId,
    }));
  }

  return { classes: nextClasses, slices };
}
