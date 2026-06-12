export type AnnotationFilter = 'all' | 'annotated' | 'unannotated';

export const ANNOTATION_FILTER_OPTIONS: { value: AnnotationFilter; label: string }[] = [
  { value: 'all', label: 'All samples' },
  { value: 'annotated', label: 'With annotations' },
  { value: 'unannotated', label: 'Without annotations' },
];
