import type { AnnotationClass } from '@/stores/classStore';
import type { Shape } from '@/stores/annotationStore';

/** Semantic taxonomy signature. Display color/visibility do not change class meaning. */
function taxonomySignature(classes: AnnotationClass[]): string {
  return JSON.stringify(
    [...classes]
      .map(({ classId, label }) => ({ classId, label: label.trim() }))
      .sort((a, b) => a.classId - b.classId),
  );
}

/**
 * Validate that a merged export has one unambiguous class-id mapping and that every
 * shape references it. Returns the taxonomy to send with the export request.
 */
export function validateSharedTaxonomy(
  sourceKeys: string[],
  classesBySource: Record<string, AnnotationClass[]>,
  byImage: Record<string, Record<string, Shape[]>>,
): AnnotationClass[] {
  if (sourceKeys.length === 0) return [];

  const firstKey = sourceKeys[0];
  const first = classesBySource[firstKey];
  if (!first) throw new Error(`No recorded class taxonomy for ${firstKey}. Reopen that sample before exporting.`);
  const expected = taxonomySignature(first);

  for (const sourceKey of sourceKeys) {
    const classes = classesBySource[sourceKey];
    if (!classes) {
      throw new Error(`No recorded class taxonomy for ${sourceKey}. Reopen that sample before exporting.`);
    }
    if (taxonomySignature(classes) !== expected) {
      throw new Error('Selected samples use different class taxonomies. Export them separately or align their class IDs and labels first.');
    }

    const classIds = new Set(classes.map((cls) => cls.classId));
    for (const shapes of Object.values(byImage[sourceKey] ?? {})) {
      for (const shape of shapes) {
        if (!classIds.has(shape.classId)) {
          throw new Error(`${sourceKey} contains a shape that references missing class ${shape.classId}.`);
        }
      }
    }
  }

  return first;
}
