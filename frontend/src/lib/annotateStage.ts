/**
 * Annotate Hub stages (sidebar routes after Connect / Browse).
 */
export type AnnotateStage = 'preprocess' | 'draw' | 'train';

const STAGE_PATHS: Record<AnnotateStage, string> = {
  preprocess: '/preprocess',
  draw: '/draw',
  train: '/train',
};

/** Map a Hub pathname to the annotate work stage. */
export function stageFromPath(pathname: string): AnnotateStage {
  if (pathname.startsWith('/train') || pathname.startsWith('/cleanup')) return 'train';
  if (pathname.startsWith('/draw')) return 'draw';
  return 'preprocess';
}

/** Hub path for a stage. */
export function pathForStage(stage: AnnotateStage): string {
  return STAGE_PATHS[stage];
}
