/**
 * runCompatibility — can a saved run be *continued* (fine-tuned further) on the
 * currently-open sample's class list?
 *
 * Applying a run for inference works across taxonomies: predictions are remapped
 * by label and unmatched run classes are appended (see importPredictions.ts).
 * Continuing to TRAIN one does not. The saved segmentation head has exactly one
 * output channel per class, in the run's own class order (the backend maps
 * channel `c` to `classes[c].classId`), so the class list has to line up
 * positionally or those weights mean something different.
 *
 * Mirrors `check_resume_compatible` in backend/train_jobs.py — that remains the
 * authority; this exists so the UI can say so before the user commits to a save
 * and a request, rather than surfacing it as an error afterwards.
 */
import type { AnnotationClass } from '@/stores/classStore';

export interface RunClassLike {
  label: string;
}

const normalize = (label: string) => label.trim().toLowerCase();

/**
 * True when *runClasses* and *currentClasses* are the same labels in the same
 * order (case- and whitespace-insensitive, matching the backend).
 */
export function canContinueFineTuning(
  runClasses: RunClassLike[],
  currentClasses: AnnotationClass[],
): boolean {
  if (runClasses.length !== currentClasses.length) return false;
  return runClasses.every((rc, i) => normalize(rc.label) === normalize(currentClasses[i].label));
}

/**
 * Why a run can't be continued, phrased for display next to the button, or null
 * when it can. Deliberately concrete about both lists — "incompatible" alone
 * leaves the user with nothing to act on.
 */
export function fineTuningBlockedReason(
  runClasses: RunClassLike[],
  currentClasses: AnnotationClass[],
): string | null {
  if (canContinueFineTuning(runClasses, currentClasses)) return null;
  const runLabels = runClasses.map((c) => c.label).join(', ') || '—';
  const currentLabels = currentClasses.map((c) => c.label).join(', ') || '—';
  if (runClasses.length !== currentClasses.length) {
    return `This run was trained on ${runClasses.length} class(es) (${runLabels}) but this image has ${currentClasses.length} (${currentLabels}). Continuing would not fit its saved weights — apply it instead, or train a new run.`;
  }
  return `This run's classes (${runLabels}) don't line up with this image's (${currentLabels}). Continuing would train against the wrong classes — apply it instead, or train a new run.`;
}
