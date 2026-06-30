/**
 * Canonical source-key helpers.
 *
 * A "source key" uniquely identifies an annotated sample for draft storage and
 * for keying the annotation store.  All code must use these helpers so that
 * autosave keys and load keys always match.
 *
 *   tiled  → "tiled:<serverUri>:<tiledPath>"
 *   local  → "local:<relPath>"
 */

/** Builds the canonical sourceKey for a sample; tiled keys embed the server URI, local keys use the relative path. */
export function buildSourceKey(
  kind: 'tiled' | 'local',
  path: string,
  serverUri?: string | null,
): string {
  return kind === 'tiled' ? `tiled:${serverUri ?? ''}:${path}` : `local:${path}`;
}
